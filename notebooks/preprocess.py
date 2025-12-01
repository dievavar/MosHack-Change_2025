import os
import re
import random
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_TRAIN_PATH   = "../data/ТОНАЛЬНОСТЬ/train.csv"
TRAIN_SUPER_PATH = "../data/processed/train_super.csv"
VAL_SUPER_PATH   = "../data/processed/val_super.csv"

VAL_SIZE     = 0.1
RANDOM_STATE = 42

# максимум примеров на один источник (src)
MAX_SAMPLES_PER_SRC = 40000

# максимум длины текста в символах; режем ближе к max_length модели (~256-320 токенов)
MAX_TEXT_LEN = 900

MAX_TOTAL_SAMPLES = None

# сколько максимум аугментаций на один исходный текст класса (мягкие)
MAX_AUG_PER_SAMPLE = 1

# ------------------ регулярки для очистки базовые ------------------ #
URL_RE        = re.compile(r"(https?://\S+|www\.\S+)")
EMAIL_RE      = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
PHONE_RE      = re.compile(r"\b(\+?\d[\d\-\s]{6,}\d)\b")
WHITESPACE_RE = re.compile(r"\s+")

# сильно повторяющиеся знаки
STRONG_EXCL_RE  = re.compile(r"!{3,}")
STRONG_QUEST_RE = re.compile(r"\?{3,}")

# сжатие повторяющихся знаков пунктуации до 3
MULTI_PUNCT_RE = re.compile(r"([!?.,])\1{2,}")

# растянутые буквы (рус/англ)
STRETCH_RE = re.compile(r"([A-Za-zА-Яа-яЁё])\1{2,}")

# рейтинги формата "2 из 10", "8/10", а также 5-балльные и звёзды
RATING_RE1   = re.compile(r"\b(\d{1,2})\s*из\s*10\b")
RATING_RE2   = re.compile(r"\b(\d{1,2})\s*/\s*10\b")
RATING5_RE1  = re.compile(r"\b(\d)\s*из\s*5\b")
RATING5_RE2  = re.compile(r"\b(\d)\s*/\s*5\b")
STARS_RE     = re.compile(r"[★⭐]{2,5}")

# латиница для детекции английского
LATIN_RE = re.compile(r"[A-Za-z]")

# эмодзи-группы (можно расширять по мере надобности)
EMOJI_MAP = [
    (re.compile(r"[😊🙂😀😁😄😃]"), " [SMILE] "),
    (re.compile(r"[😂🤣]"), " [LAUGH] "),
    (re.compile(r"[😡🤬😠]"), " [ANGER] "),
    (re.compile(r"[😢😭]"), " [CRY] "),
    (re.compile(r"[😍❤️💕💖💘💓]"), " [LOVE] "),
    (re.compile(r"[🤮🤢]"), " [DISGUST] "),
    (re.compile(r"[🙈🙉🙊]"), " [FACEPALM] "),
    (re.compile(r"[😎🤩]"), " [COOL] "),
    (re.compile(r"[✌👌👍👎]"), " [GESTURE] "),
    (re.compile(r"[🍷🍸🍺🥂🍻]"), " [DRINK] "),
]

# сленг/брань -> токены (можно постепенно расширять по анализу корпуса)
SLANG_MAP = [
    (re.compile(r"\bфуф+ел\w*\b", re.IGNORECASE), " [BAD_QUALITY] "),
    (re.compile(r"\bговн\w*\b", re.IGNORECASE), " [BAD_STRONG] "),
    (re.compile(r"\bотстой\b", re.IGNORECASE), " [BAD] "),
    (re.compile(r"\bжесть\b", re.IGNORECASE), " [WOW] "),
    (re.compile(r"\bах+\*+\b", re.IGNORECASE), " [STRONG] "),
    (re.compile(r"\bёкарныйбабай\b", re.IGNORECASE), " [EXCLAIM] "),
]

# разбиение на предложения
SENT_SPLIT_RE = re.compile(r'(?<=[.!?…])\s+')

# “ударные” токены и слова, по которым будем выбирать важные предложения
KEY_TOKENS = [
    "[SMILE]", "[LAUGH]", "[ANGER]", "[CRY]", "[LOVE]", "[DISGUST]",
    "[FACEPALM]", "[COOL]", "[GESTURE]", "[DRINK]",
    "[LOW_SCORE]", "[MID_SCORE]", "[HIGH_SCORE]",
    "[BAD]", "[BAD_STRONG]", "[BAD_QUALITY]",
    "[WOW]", "[EXCLAIM]", "[STRONG_EXCL]", "[STRONG_QUEST]",
]

KEY_WORD_PATTERNS = [
    re.compile(r"\bужас\w*\b", re.IGNORECASE),
    re.compile(r"\bотстой\w*\b", re.IGNORECASE),
    re.compile(r"\bшикарн\w*\b", re.IGNORECASE),
    re.compile(r"\bпрекрасн\w*\b", re.IGNORECASE),
    re.compile(r"\bрекомендую\b", re.IGNORECASE),
    re.compile(r"\bне рекомендую\b", re.IGNORECASE),
    re.compile(r"\bне советую\b", re.IGNORECASE),
    re.compile(r"\bфуф+ел\w*\b", re.IGNORECASE),
    re.compile(r"\bговн\w*\b", re.IGNORECASE),
]


def normalize_emojis(text: str) -> str:
    """Заменяем эмодзи на смысловые токены."""
    for pattern, token in EMOJI_MAP:
        text = pattern.sub(token, text)
    return text


def normalize_stretched(text: str) -> str:
    """Сжимаем чрезмерно растянутые буквы: крууутой -> круутой."""
    return STRETCH_RE.sub(r"\1\1", text)


def apply_slang_map(text: str) -> str:
    """Заменяем сленг/брань на токены."""
    for pattern, token in SLANG_MAP:
        text = pattern.sub(token, text)
    return text


def _rating_token(val: int, low=4, high=7,
                  token_low="[LOW_SCORE]",
                  token_mid="[MID_SCORE]",
                  token_high="[HIGH_SCORE]"):
    if val <= low:
        return token_low
    if val >= high:
        return token_high
    return token_mid


def mark_ratings(text: str) -> str:
    """
    Находим конструкции вида "2 из 10", "8/10", а также 5-балльные и звёзды.
    Добавляем маркеры [LOW_SCORE]/[MID_SCORE]/[HIGH_SCORE].
    """
    def _replacer(match, scale=10):
        try:
            val = int(match.group(1))
        except Exception:
            return match.group(0)
        if scale == 5:
            val = val * 2
        token = _rating_token(val)
        return f"{token} {match.group(0)}"

    text = RATING_RE1.sub(lambda m: _replacer(m, scale=10), text)
    text = RATING_RE2.sub(lambda m: _replacer(m, scale=10), text)
    text = RATING5_RE1.sub(lambda m: _replacer(m, scale=5), text)
    text = RATING5_RE2.sub(lambda m: _replacer(m, scale=5), text)

    def _stars_replacer(match):
        count = len(match.group(0))
        val = count * 2
        token = _rating_token(val, low=4, high=7)
        return f"{token} {match.group(0)}"

    text = STARS_RE.sub(_stars_replacer, text)
    return text


def mark_caps(text: str) -> str:
    """
    Если значительная часть слов в ВЕРХНЕМ РЕГИСТРЕ, добавляем токен [ALL_CAPS].
    """
    words = text.split()
    if not words:
        return text
    caps_words = [w for w in words if len(w) > 3 and w.isupper()]
    if caps_words and len(caps_words) / len(words) >= 0.3:
        return "[ALL_CAPS] " + text
    return text


def mark_english(text: str) -> str:
    """
    Если в тексте заметная доля латиницы — помечаем [EN].
    """
    latin_chars = LATIN_RE.findall(text)
    if len(latin_chars) >= 10:
        return "[EN] " + text
    return text


def clean_text(text: str) -> str:
    """
    Продвинутый препроцессинг:
      - убираем невидимые символы,
      - HTML,
      - URL/EMAIL/PHONE -> токены,
      - эмодзи -> токены,
      - маркеры рейтингов (в т.ч. 5-балльные/звёзды),
      - сильная пунктуация -> токены,
      - нормализуем растянутые буквы,
      - сленг -> токены,
      - помечаем капс и английский,
      - нормализуем пробелы.
    """
    if not isinstance(text, str):
        return ""

    text = "".join(ch for ch in text if ch.isprintable())
    text = re.sub(r"<[^>]+>", " ", text)
    text = URL_RE.sub(" [URL] ", text)
    text = EMAIL_RE.sub(" [EMAIL] ", text)
    text = PHONE_RE.sub(" [PHONE] ", text)
    text = normalize_emojis(text)
    text = mark_ratings(text)
    text = STRONG_EXCL_RE.sub(" [STRONG_EXCL] ", text)
    text = STRONG_QUEST_RE.sub(" [STRONG_QUEST] ", text)
    text = MULTI_PUNCT_RE.sub(r"\1\1\1", text)
    text = normalize_stretched(text)
    text = apply_slang_map(text)
    text = mark_caps(text)
    text = mark_english(text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


# --------- укорочение длинных текстов по предложениям --------- #

def sentence_has_signal(sent: str) -> bool:
    """Проверяем, есть ли в предложении 'ударные' токены/слова."""
    for tok in KEY_TOKENS:
        if tok in sent:
            return True
    for pat in KEY_WORD_PATTERNS:
        if pat.search(sent):
            return True
    return False


def shorten_text_by_sentences(text: str,
                              max_len: int = MAX_TEXT_LEN,
                              max_first_sent: int = 3) -> str:
    """
    Если текст длиннее max_len:
      1) Берём первые max_first_sent предложений,
      2) Добавляем 'ударные' предложения (с эмоциями/оценками),
      3) Собираем обратно, не выходя за max_len.
    """
    if len(text) <= max_len:
        return text

    sentences = SENT_SPLIT_RE.split(text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return text[:max_len]

    selected = []
    for s in sentences[:max_first_sent]:
        selected.append(s)

    for s in sentences[max_first_sent:]:
        if sentence_has_signal(s) and s not in selected:
            selected.append(s)

    result = ""
    for s in selected:
        candidate = (result + " " + s).strip() if result else s
        if len(candidate) > max_len:
            break
        result = candidate

    if not result:
        return text[:max_len]

    return result


# ------------------ простые аугментации ------------------ #

def random_delete(words, p=0.05):
    """Мягкий word dropout."""
    if len(words) <= 3:
        return words
    new_words = []
    for w in words:
        if random.random() > p:
            new_words.append(w)
    if not new_words:
        return words
    return new_words


def random_swap(words, n_swaps=1, prob=0.2):
    """Редкая перестановка слов."""
    if len(words) <= 3 or random.random() > prob:
        return words
    words = words[:]
    for _ in range(n_swaps):
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
    return words


def augment_text(text: str, n_aug: int = 1):
    """Генерирует n_aug аугментированных версий текста."""
    words = text.split()
    aug_texts = []
    for _ in range(n_aug):
        tmp = words[:]
        tmp = random_delete(tmp, p=0.05)
        tmp = random_swap(tmp, n_swaps=1, prob=0.2)
        aug_texts.append(" ".join(tmp))
    return aug_texts



def main():
    if not os.path.exists(RAW_TRAIN_PATH):
        raise FileNotFoundError(f"Не найден {RAW_TRAIN_PATH}")

    random.seed(RANDOM_STATE)

    print(f"Читаем {RAW_TRAIN_PATH}...")
    df = pd.read_csv(RAW_TRAIN_PATH)

    for col in ["text", "label"]:
        if col not in df.columns:
            raise ValueError(f"В train.csv нет колонки '{col}'")
    if "src" not in df.columns:
        print("⚠️ Колонки 'src' нет, доменные токены добавлены не будут.")

    print("Размер исходного датасета:", df.shape)
    print("\nРаспределение классов (сырые):")
    print(df["label"].value_counts().sort_index())

    df = df.dropna(subset=["text", "label"]).copy()

    before = len(df)
    df = df.drop_duplicates(subset=["text", "label"])
    print("Удалено полных дубликатов:", before - len(df))

    print("\nОчистка текста (продвинутая)...")
    df["text"] = df["text"].astype(str).apply(clean_text)

    before = len(df)
    df = df[df["text"].str.len() > 0]
    print("Удалено пустых после очистки:", before - len(df))

    long_before = (df["text"].str.len() > MAX_TEXT_LEN).sum()
    print(f"\nСлишком длинных текстов (> {MAX_TEXT_LEN} символов) до укорочения:", long_before)
    if long_before > 0:
        df["text"] = df["text"].apply(lambda t: shorten_text_by_sentences(t, max_len=MAX_TEXT_LEN))
        long_after = (df["text"].str.len() > MAX_TEXT_LEN).sum()
        print(f"Слишком длинных текстов после укорочения (ещё > {MAX_TEXT_LEN}):", long_after)

    if MAX_SAMPLES_PER_SRC is not None and "src" in df.columns:
        print(f"\nОграничиваем максимум {MAX_SAMPLES_PER_SRC} примеров на каждое значение src...")
        df = df.groupby("src", group_keys=False).apply(
            lambda g: g.sample(n=min(len(g), MAX_SAMPLES_PER_SRC), random_state=RANDOM_STATE)
        )
        print("Размер после downsampling по src:", df.shape)
        print("Распределение по src:")
        print(df["src"].value_counts())
    else:
        print("\nDownsampling по src пропущен (либо нет колонки 'src', либо MAX_SAMPLES_PER_SRC=None).")

    if MAX_TOTAL_SAMPLES is not None and len(df) > MAX_TOTAL_SAMPLES:
        print(f"\nОбщий размер df = {len(df)}, режем до {MAX_TOTAL_SAMPLES} строк (стратифицированно по label)...")
        label_counts = df["label"].value_counts()
        total = len(df)
        samples_per_label = {}
        for label, cnt in label_counts.items():
            n = int(MAX_TOTAL_SAMPLES * cnt / total)
            n = max(1, n)
            samples_per_label[label] = min(n, cnt)
        parts = []
        for label, n in samples_per_label.items():
            part = df[df["label"] == label].sample(n=n, random_state=RANDOM_STATE)
            parts.append(part)
        df = pd.concat(parts, ignore_index=True)
        print("Размер после общего лимита:", df.shape)
        print("Распределение классов после общего лимита:")
        print(df["label"].value_counts().sort_index())
    else:
        print("\nОбщий лимит по объёму данных не применён (либо MAX_TOTAL_SAMPLES=None, либо данных и так мало).")

    lengths = df["text"].str.len()
    print("\nСтатистика длины текста (символы) ПОСЛЕ укорочения/фильтрации:")
    print(lengths.describe(percentiles=[0.1, 0.5, 0.9, 0.95, 0.99]))

    print(f"\nДелаем stratified split (val={VAL_SIZE})...")
    train_df, val_df = train_test_split(
        df,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    print("Train shape (до аугментации):", train_df.shape)
    print("Val shape:", val_df.shape)
    print("\nРаспределение классов в train (до аугм.):")
    print(train_df["label"].value_counts(normalize=True).sort_index())

    print("\nЗапускаем аугментации для балансировки классов (мягкие)...")
    label_counts = train_df["label"].value_counts().sort_index()
    print("Классы до аугментации:", label_counts.to_dict())
    max_count = label_counts.max()

    augmented_rows = []
    for label, count in label_counts.items():
        deficit = max_count - count
        if deficit <= 0:
            continue
        class_df = train_df[train_df["label"] == label]
        est_aug_per_sample = min(MAX_AUG_PER_SAMPLE, max(1, deficit // max(1, len(class_df))))
        print(f"\nКласс {label}: нужно добавить ~{deficit} примеров, будем делать до {est_aug_per_sample} аугм./текст")
        added = 0
        for row in class_df.itertuples(index=False):
            if added >= deficit:
                break
            base_text = row.text
            base_src = getattr(row, "src", None) if "src" in class_df.columns else None
            aug_list = augment_text(base_text, n_aug=est_aug_per_sample)
            for aug_t in aug_list:
                if added >= deficit:
                    break
                new_row = {"text": aug_t, "label": label}
                if base_src is not None:
                    new_row["src"] = base_src
                augmented_rows.append(new_row)
                added += 1
        print(f"Фактически добавлено для класса {label}: {added}")

    if augmented_rows:
        aug_df = pd.DataFrame(augmented_rows)
        print("\nРазмер аугментированных данных:", aug_df.shape)
        train_df = pd.concat([train_df, aug_df], ignore_index=True)
    else:
        print("\nАугментации не потребовались (классы уже сбалансированы).")

    print("\nРаспределение классов в train (ПОСЛЕ аугм.):")
    print(train_df["label"].value_counts().sort_index())
    print(train_df["label"].value_counts(normalize=True).sort_index())

    if "src" in train_df.columns:
        def add_src_token(row):
            src = str(row["src"]).strip().lower()
            if not src:
                return row["text"]
            token = "[SRC_" + src.replace(" ", "_") + "]"
            return f"{token} {row['text']}"
        train_df["text"] = train_df.apply(add_src_token, axis=1)
        if "src" in val_df.columns:
            val_df["text"] = val_df.apply(add_src_token, axis=1)
        print("\nДобавлены доменные токены [SRC_xxx] в начало текста.")
    else:
        print("\nДоменные токены пропущены (нет src).")

    train_df = train_df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    train_df.to_csv(TRAIN_SUPER_PATH, index=False)
    val_df.to_csv(VAL_SUPER_PATH, index=False)

    total_rows = len(train_df) + len(val_df)
    print(f"\nИТОГО после всех фильтров и аугментаций:")
    print(f"  train_super: {len(train_df)} строк")
    print(f"  val_super:   {len(val_df)} строк")
    print(f"  ВСЕГО:       {total_rows} строк")
    print(f"\nСохранено:\n  {TRAIN_SUPER_PATH}\n  {VAL_SUPER_PATH}")


if __name__ == "__main__":
    main()
