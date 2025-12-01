import random
import re
import pandas as pd

VAL_SIZE     = 0.1
RANDOM_STATE = 42
MAX_SAMPLES_PER_SRC = 20000
MAX_TEXT_LEN = 1200
# MAX_TOTAL_SAMPLES = 120000
MAX_AUG_PER_SAMPLE = 2

#регулярки для очистки
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

# рейтинги формата "2 из 10", "8/10"
RATING_RE1 = re.compile(r"\b(\d{1,2})\s*из\s*10\b")
RATING_RE2 = re.compile(r"\b(\d{1,2})\s*/\s*10\b")

# латиница для детекции английского
LATIN_RE = re.compile(r"[A-Za-z]")

# эмодзи-группы 
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


def mark_ratings(text: str) -> str:
    """
    Находим конструкции вида "2 из 10", "8/10" и добавляем маркеры
    [LOW_SCORE] / [MID_SCORE] / [HIGH_SCORE].
    """
    def _rating_replacer(match, low=4, high=7,
                         token_low="[LOW_SCORE]",
                         token_mid="[MID_SCORE]",
                         token_high="[HIGH_SCORE]"):
        try:
            val = int(match.group(1))
        except Exception:
            return match.group(0)
        if val <= low:
            token = token_low
        elif val >= high:
            token = token_high
        else:
            token = token_mid
        return f"{token} {match.group(0)}"

    text = RATING_RE1.sub(_rating_replacer, text)
    text = RATING_RE2.sub(_rating_replacer, text)
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
      - маркеры рейтингов,
      - сильная пунктуация -> токены,
      - нормализуем растянутые буквы,
      - сленг -> токены,
      - помечаем капс и английский,
      - нормализуем пробелы.
    """
    if not isinstance(text, str):
        return ""

    # убираем непечатные символы
    text = "".join(ch for ch in text if ch.isprintable())

    # HTML
    text = re.sub(r"<[^>]+>", " ", text)

    # URL / EMAIL / PHONE
    text = URL_RE.sub(" [URL] ", text)
    text = EMAIL_RE.sub(" [EMAIL] ", text)
    text = PHONE_RE.sub(" [PHONE] ", text)

    # эмодзи -> токены
    text = normalize_emojis(text)

    # рейтинги вида "2 из 10", "8/10"
    text = mark_ratings(text)

    # сильные последовательности знаков препинания -> токены
    text = STRONG_EXCL_RE.sub(" [STRONG_EXCL] ", text)
    text = STRONG_QUEST_RE.sub(" [STRONG_QUEST] ", text)

    # сжимаем чрезмерные повторы пунктуации до 3
    text = MULTI_PUNCT_RE.sub(r"\1\1\1", text)

    # нормализация растянутых букв
    text = normalize_stretched(text)

    # сленг / брань -> токены
    text = apply_slang_map(text)

    # маркер капса
    text = mark_caps(text)

    # маркер английского
    text = mark_english(text)

    # финальная нормализация пробелов
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


#укорочение длинных текстов по предложениям

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

    # режем на предложения
    sentences = SENT_SPLIT_RE.split(text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        # fallback — просто обрезать по длине
        return text[:max_len]

    # сначала берём первые N предложений
    selected = []
    for s in sentences[:max_first_sent]:
        selected.append(s)

    # затем добавляем "ударные" (но не дублируем)
    for s in sentences[max_first_sent:]:
        if sentence_has_signal(s) and s not in selected:
            selected.append(s)

    # собираем, не превышая max_len
    result = ""
    for s in selected:
        candidate = (result + " " + s).strip() if result else s
        if len(candidate) > max_len:
            break
        result = candidate

    if not result:
        # если вдруг ничего не собралось — жёстко обрежем
        return text[:max_len]

    return result





def add_src_token_if_exists(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляем доменный токен [SRC_xxx] в начало текста,
    если в датафрейме есть колонка 'src'.
    """
    df = df.copy()

    if "src" not in df.columns:
        print("В test.csv нет колонки 'src' — доменные токены не добавляем.")
        return df

    def add_src_token(row):
        src = str(row["src"]).strip().lower()
        if not src:
            return row["text"]
        token = "[SRC_" + src.replace(" ", "_") + "]"
        return f"{token} {row['text']}"

    df["text"] = df.apply(add_src_token, axis=1)
    print("Добавлены доменные токены [SRC_xxx] в начало текста (test).")
    return df


def preprocess_test_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Полноценный препроцессинг test:
      - text -> str, заполняем NaN
      - clean_text (как в train_super)
      - укорачивание длинных текстов shorten_text_by_sentences
      - добавление [SRC_xxx] (если есть src)
    НИЧЕГО НЕ РЕЖЕМ по количеству, НЕ делаем аугментаций, НЕ меняем ID.
    """
    df = df.copy()

    if "text" not in df.columns:
        raise ValueError("В test.csv нет колонки 'text'")

    # приводим к строкам, NaN -> ""
    df["text"] = df["text"].fillna("").astype(str)

    # очистка (та же, что для train)
    print("Очистка текста (clean_text) для test...")
    df["text"] = df["text"].apply(clean_text)

    # текст может стать пустым — для теста строки НЕ выбрасываем,
    # просто логируем, чтобы понимать масштаб бедствия
    empty_after_clean = (df["text"].str.len() == 0).sum()

    # укорачиваем длинные тексты
    if MAX_TEXT_LEN is not None:
        long_cnt = (df["text"].str.len() > MAX_TEXT_LEN).sum()
        if long_cnt > 0:
            df["text"] = df["text"].apply(
                lambda t: shorten_text_by_sentences(t, max_len=MAX_TEXT_LEN)
            )
            long_cnt_after = (df["text"].str.len() > MAX_TEXT_LEN).sum()

    # доменные токены [SRC_xxx]
    df = add_src_token_if_exists(df)
    return df


