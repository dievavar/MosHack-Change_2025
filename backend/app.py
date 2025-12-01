from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response, JSONResponse
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import torch
from joblib import load
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from starlette.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
from backend.preprocess_test import preprocess_test_df
import logging
from enum import Enum
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "sentiment_lr_optuna.joblib"  # ← Это ФАЙЛ, не папка!

# Логируем путь
logger.info(f"Путь к модели: {MODEL_PATH}")
logger.info(f"Модель существует: {MODEL_PATH.exists()}")
logger.info(f"Это файл: {MODEL_PATH.is_file()}")


class ModelType(Enum):
    BERT = "bert"
    JOBLIB = "joblib"
    UNKNOWN = "unknown"


def detect_model_type(model_path: Path) -> ModelType:
    """
    Определяет тип модели по расширению файла или содержимому папки
    """
    if not model_path.exists():
        logger.error(f"Модель не найдена: {model_path}")
        return ModelType.UNKNOWN

    # Если это файл
    if model_path.is_file():
        extension = model_path.suffix.lower()
        if extension in ['.joblib', '.pkl', '.pickle']:
            logger.info(f"Найден файл joblib модели: {model_path}")
            return ModelType.JOBLIB
        elif extension in ['.bin', '.safetensors', '.pt', '.pth']:
            logger.info(f"Найден файл PyTorch модели: {model_path}")
            return ModelType.BERT
        else:
            logger.warning(f"Неизвестное расширение файла: {extension}")
            return ModelType.UNKNOWN

    # Если это папка (для BERT моделей)
    elif model_path.is_dir():
        # Проверяем на BERT (трансформеры)
        bert_files = ["pytorch_model.bin", "model.safetensors", "config.json", "tokenizer.json"]
        for bert_file in bert_files:
            if (model_path / bert_file).exists():
                logger.info(f"Найден BERT файл: {bert_file}")
                return ModelType.BERT

        # Проверяем на joblib в папке
        for pattern in ["*.joblib", "*.pkl"]:
            if list(model_path.glob(pattern)):
                logger.info(f"Найден joblib файл в папке")
                return ModelType.JOBLIB

    logger.warning(f"Не удалось определить тип модели в {model_path}")
    return ModelType.UNKNOWN


# Определяем тип модели
model_type = detect_model_type(MODEL_PATH)
logger.info(f"Определён тип модели: {model_type.value}")

device = None
tokenizer = None
model = None
MAX_LEN = 192
BATCH_SIZE = 32

if model_type == ModelType.BERT:
    # Настройка устройства для BERT
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logger.info(f"Используем устройство для BERT: {device}")

    # Если MODEL_PATH - это файл, а не папка с BERT
    if MODEL_PATH.is_file():
        logger.error("Для BERT модели нужна папка, а не файл!")
        raise ValueError(f"BERT модель ожидает папку, получен файл: {MODEL_PATH}")

    # Загружаем BERT модель из папки
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

        # Определяем количество классов
        config_path = MODEL_PATH / "config.json"
        if config_path.exists():
            import json

            with open(config_path, 'r') as f:
                config = json.load(f)
                num_labels = config.get("num_labels", 3)
        else:
            num_labels = 3

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH,
            num_labels=num_labels,
        )
        model.to(device)
        model.eval()
        logger.info("BERT модель успешно загружена")

    except Exception as e:
        logger.error(f"Ошибка загрузки BERT модели: {e}")
        raise

elif model_type == ModelType.JOBLIB:
    # Загружаем joblib модель из файла
    try:
        # Если MODEL_PATH - это уже файл .joblib
        if MODEL_PATH.is_file() and MODEL_PATH.suffix.lower() in ['.joblib', '.pkl']:
            model = load(MODEL_PATH)
            logger.info(f"Joblib модель загружена из файла: {MODEL_PATH.name}")

        # Если MODEL_PATH - это папка с файлом .joblib внутри
        elif MODEL_PATH.is_dir():
            joblib_files = list(MODEL_PATH.glob("*.joblib")) + list(MODEL_PATH.glob("*.pkl"))
            if not joblib_files:
                raise FileNotFoundError(f"Не найден файл .joblib или .pkl в {MODEL_PATH}")

            model_file = joblib_files[0]
            model = load(model_file)
            logger.info(f"Joblib модель загружена из: {model_file}")

        else:
            raise ValueError(f"Некорректный путь к joblib модели: {MODEL_PATH}")

        # Проверяем что модель загрузилась правильно
        logger.info(f"Тип загруженной модели: {type(model)}")
        if hasattr(model, 'classes_'):
            logger.info(f"Классы модели: {model.classes_}")

    except Exception as e:
        logger.error(f"Ошибка загрузки joblib модели: {e}")
        raise

else:
    raise ValueError(f"Неизвестный тип модели или модель не найдена в {MODEL_PATH}")


def bert_predict(texts: List[str]) -> np.ndarray:
    """
    Предсказание для BERT модели
    """
    if model_type != ModelType.BERT:
        raise ValueError("Эта функция только для BERT моделей")

    all_preds = []
    texts = list(texts)

    for start in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[start:start + BATCH_SIZE]

        enc = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = model(**enc)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

        all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_preds).astype(int)


def joblib_predict(texts: List[str]) -> np.ndarray:
    """
    Предсказание для joblib модели
    """
    if model_type != ModelType.JOBLIB:
        raise ValueError("Эта функция только для joblib моделей")

    try:
        # Если это pipeline scikit-learn
        if hasattr(model, 'predict'):
            logger.info(f"Используем метод predict для {len(texts)} текстов")
            predictions = model.predict(texts)
            logger.info(f"Получено предсказаний: {len(predictions)}")
            return predictions.astype(int)
        else:
            raise ValueError("Неподдерживаемый формат joblib модели")
    except Exception as e:
        logger.error(f"Ошибка предсказания joblib модели: {e}")
        # Для отладки
        logger.error(f"Тип модели: {type(model)}")
        logger.error(f"Атрибуты модели: {dir(model)}")
        raise


def universal_predict(texts: List[str]) -> np.ndarray:
    """
    Универсальная функция предсказания
    """
    logger.info(f"Запрос на предсказание для {len(texts)} текстов")

    if model_type == ModelType.BERT:
        result = bert_predict(texts)
    elif model_type == ModelType.JOBLIB:
        result = joblib_predict(texts)
    else:
        raise ValueError(f"Неподдерживаемый тип модели: {model_type}")

    logger.info(f"Предсказания готовы, форма: {result.shape}")
    return result


def make_id_label_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ожидаем CSV с колонками: ID, text, src.
    Возвращаем DataFrame с колонками: ID, label (предсказание модели).
    """
    required = {"ID", "text", "src"}
    missing = required - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=(
                f"В CSV должны быть колонки: {', '.join(sorted(required))}. "
                f"Отсутствуют: {', '.join(sorted(missing))}"
            ),
        )

    logger.info(f"Обработка CSV с {len(df)} строками")

    # Предобработка
    df_proc = preprocess_test_df(df)
    texts_proc = df_proc["text"].tolist()

    # Используем универсальную функцию предсказания
    preds = universal_predict(texts_proc)

    result_df = pd.DataFrame({
        "ID": df["ID"].astype(int),
        "label": preds,
    })

    logger.info(f"Результат сформирован, строк: {len(result_df)}")
    return result_df


@app.post("/api/predict-file")
async def predict_file(file: UploadFile = File(...)):
    """
    Эндпоинт для предсказания по CSV файлу
    """
    logger.info(f"Получен файл: {file.filename}")

    # Проверяем расширение
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Требуется CSV-файл (.csv)")

    # Читаем CSV
    try:
        df = pd.read_csv(file.file)
        logger.info(f"CSV прочитан успешно, строк: {len(df)}")
    except Exception as e:
        logger.error(f"Ошибка чтения CSV: {e}")
        raise HTTPException(status_code=400, detail=f"Ошибка чтения CSV: {e}")

    # Получаем предсказания
    result_df = make_id_label_df(df)

    # В CSV-строку
    csv_bytes = result_df.to_csv(index=False).encode("utf-8")

    # Отдаём как файл
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={
            "Content-Disposition": 'attachment; filename="predicted_labels.csv"'
        },
    )


@app.get("/api/model-info")
async def get_model_info():
    """
    Информация о загруженной модели
    """
    return {
        "model_type": model_type.value,
        "model_path": str(MODEL_PATH),
        "is_file": MODEL_PATH.is_file(),
        "is_dir": MODEL_PATH.is_dir(),
        "device": str(device) if device else "cpu (joblib)",
        "model_exists": MODEL_PATH.exists()
    }


@app.get("/api/test")
async def test_endpoint():
    """
    Тестовый эндпоинт для проверки работы
    """
    test_texts = ["Пример текста для тестирования", "Второй пример текста"]

    if model_type == ModelType.JOBLIB:
        try:
            # Пробуем сделать предсказание на тестовых данных
            predictions = model.predict(test_texts)
            return {
                "status": "ok",
                "model_type": model_type.value,
                "test_predictions": predictions.tolist(),
                "model_class": str(type(model))
            }
        except Exception as e:
            return {
                "status": "error",
                "model_type": model_type.value,
                "error": str(e),
                "model_class": str(type(model))
            }

    return {"status": "ok", "model_type": model_type.value}


# статика фронта
app.mount("/", StaticFiles(directory="frontend", html=True), name="static")
