import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TextClassificationPipeline,
)

IN_PATH  = Path("data/airpods_clean.csv")
OUT_PATH = Path("data/airpods_scored.csv")

# Модель
CHECKPOINT = "sismetanin/rubert-ru-sentiment-RuReviews"

# Параметры инференса
BATCH_SIZE  = int(os.getenv("BATCH_SIZE", 16))
MAX_LENGTH  = int(os.getenv("MAX_LENGTH", 512))

# Глобальный кэш пайплайна (прогревается один раз)
_PIPE: TextClassificationPipeline | None = None

# Отображение меток
LABEL_MAPPING = {
    "LABEL_0": "neutral",
    "LABEL_1": "negative",
    "LABEL_2": "positive",
}


def pick_device() -> int:
    """Возвращает 0 если есть CUDA, иначе -1 (CPU)."""
    return 0 if torch.cuda.is_available() else -1


def build_pipeline() -> TextClassificationPipeline:
    """Создаёт пайплайн (без глобального кэша)."""
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT)
    device = pick_device()
    pipe = TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True,
        device=device,
    )
    return pipe


def ensure_model() -> TextClassificationPipeline:
    """
    Прогрев и возврат единственного экземпляра пайплайна.
    Вызывается из Dockerfile и при старте приложения.
    """
    global _PIPE
    if _PIPE is None:
        _PIPE = build_pipeline()
    return _PIPE


def normalize_scores(score_list):
    """
    score_list: [{'label': 'LABEL_0', 'score': 0.98}, ...]
    -> (pred_label, p_positive, p_negative)
    """
    mapped = {
        LABEL_MAPPING.get(item["label"], item["label"]): float(item["score"])
        for item in score_list
    }
    # на всякий случай — заполним отсутствующие
    for key in ("negative", "neutral", "positive"):
        mapped.setdefault(key, 0.0)

    pred_label = max(mapped.items(), key=lambda kv: kv[1])[0]
    p_positive = mapped["positive"]
    p_negative = mapped["negative"]
    return pred_label, p_positive, p_negative


def _score_texts(texts: list[str]) -> tuple[list[str], list[float], list[float]]:
    """Скоры для списка текстов батчами — использует прогретый пайплайн."""
    pipe = ensure_model()

    pred_labels, pos_probs, neg_probs = [], [], []
    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Scoring"):
        batch = texts[start : start + BATCH_SIZE]
        results = pipe(
            batch,
            truncation=True,
            max_length=MAX_LENGTH,
            batch_size=BATCH_SIZE,
        )
        for scores in results:
            lbl, p_pos, p_neg = normalize_scores(scores)
            pred_labels.append(lbl)
            pos_probs.append(p_pos)
            neg_probs.append(p_neg)
    return pred_labels, pos_probs, neg_probs


def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(
            f"Нет входного файла {IN_PATH}. Сначала запусти preprocess."
        )

    df = pd.read_csv(IN_PATH)
    if "text" not in df.columns:
        raise ValueError("В файле нет колонки 'text'")

    texts = df["text"].astype(str).fillna("").tolist()
    lbls, p_pos, p_neg = _score_texts(texts)

    df["pred_label"] = lbls
    df["p_positive"] = p_pos
    df["p_negative"] = p_neg

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"[OK] saved {OUT_PATH}  rows={len(df)}")


def score_sentiments(input_path: str, output_path: str = "data/airpods_scored.csv") -> str:
    """
    Унифицированный интерфейс: читает CSV после препроцессинга,
    прогоняет модель, сохраняет scored CSV и возвращает путь.
    """
    global IN_PATH, OUT_PATH
    IN_PATH = Path(input_path)
    OUT_PATH = Path(output_path)
    main()
    return str(OUT_PATH.resolve())


def analyze_sentiment(input_path: str, output_path: str = "data/airpods_scored.csv") -> str:
    """Старая обёртка для совместимости."""
    return score_sentiments(input_path, output_path)


if __name__ == "__main__":
    # Локальный прогон
    ensure_model()
    main()
