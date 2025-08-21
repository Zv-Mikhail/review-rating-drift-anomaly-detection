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

CHECKPOINT = "sismetanin/rubert-ru-sentiment-RuReviews"

BATCH_SIZE = int(os.getenv("BATCH_SIZE", 16))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", 512))

# Меток для этой модели
LABEL_MAPPING = {
    "LABEL_0": "neutral",
    "LABEL_1": "negative",
    "LABEL_2": "positive"
}

def pick_device():
    if torch.cuda.is_available():
        return 0
    return -1  # CPU

def build_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT)
    device = pick_device()
    pipe = TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True,
        device=device
    )
    return pipe

def normalize_scores(score_list):
    """
    score_list: [{'label': 'LABEL_0', 'score': 0.98}, ...]
    -> (pred_label, p_positive, p_negative)
    """
    mapped = {LABEL_MAPPING.get(item["label"], item["label"]): float(item["score"])
              for item in score_list}

    
    for key in ("negative", "neutral", "positive"):
        mapped.setdefault(key, 0.0)

    pred_label = max(mapped.items(), key=lambda kv: kv[1])[0]
    p_positive = mapped["positive"]
    p_negative = mapped["negative"]
    return pred_label, p_positive, p_negative

def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Нет входного файла {IN_PATH}. Сначала запусти preprocess_airpods.py")

    df = pd.read_csv(IN_PATH)
    if "text" not in df.columns:
        raise ValueError("В файле нет колонки 'text'")

    pipe = build_pipeline()

    texts = df["text"].astype(str).fillna("").tolist()
    pred_labels, pos_probs, neg_probs = [], [], []

    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Scoring"):
        batch = texts[start:start + BATCH_SIZE]
        results = pipe(batch, truncation=True, max_length=MAX_LENGTH, batch_size=BATCH_SIZE)
        for scores in results:
            lbl, p_pos, p_neg = normalize_scores(scores)
            pred_labels.append(lbl)
            pos_probs.append(p_pos)
            neg_probs.append(p_neg)

    df["pred_label"] = pred_labels
    df["p_positive"] = pos_probs
    df["p_negative"] = neg_probs

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"[OK] saved {OUT_PATH}  rows={len(df)}")
    # print(df[["text","pred_label","p_positive","p_negative"]].head(5))

def score_sentiments(input_path: str, output_path: str = "data/airpods_scored.csv") -> str:
    """
    Унифицированный интерфейс: читает CSV после препроцессинга,
    прогоняет модель, сохраняет scored CSV.
    Возвращает путь к сохранённому файлу.
    """
    global IN_PATH, OUT_PATH

    IN_PATH = Path(input_path)
    OUT_PATH = Path(output_path)
    main()
    return str(OUT_PATH.resolve())


def analyze_sentiment(input_path: str, output_path: str = "data/airpods_scored.csv") -> str:
    """
    Старая обёртка для совместимости.
    Делает то же самое, что score_sentiments().
    """
    return score_sentiments(input_path, output_path)

if __name__ == "__main__":
    main()
