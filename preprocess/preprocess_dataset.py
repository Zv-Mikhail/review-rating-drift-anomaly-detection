import re
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd


IN_PRIMARY = Path("datasets/air_pods5.xlsx")
IN_FALLBACK = Path("datasets/pods.xlsx")
OUT_DIR = Path("data")
OUT_PATH = OUT_DIR / "airpods_clean.csv"


def load_source() -> pd.DataFrame:
    src = IN_PRIMARY if IN_PRIMARY.exists() else IN_FALLBACK
    if not src.exists():
        raise FileNotFoundError(f"Нет входного файла: {IN_PRIMARY} (или {IN_FALLBACK})")
    df = pd.read_excel(src)
    
    if "review_date" in df.columns:
        df = df[df["review_date"] != "Не найдено"].copy()
    return df

# чистка disadvantages: «минусов нет» -> NaN ---
EMPTY_CONS = [
    r"^нет$",
    r"^нету$",
    r"^-+$",
    r"^(?:нет(?:у)?\s+минусов?)$",
    r"^(?:все|всё)\s+(?:ок|отлично)$",
    r"^ничего$",
    r"^без\s+минус(?:ов)?$",
    r"^минус(?:ов|ы)?\s+нет(?:у)?$",
    r"^не\s+(?:наш[её]л|нашл[аи])$",
    r"^не\s+встретил[аи]$",
    r"^ещ[её]\s+не\s+нашл[аи]$",
    r"^(?:недостатков|минус(?:ов)?)\s+пока\s+не\s+увидел[аи]$",
    r"^минус(?:ы)?\s+ещ[её]\s+не\s+нашл[аи]$",
]
EMPTY_CONS_RE = re.compile(r"^(?:" + r"|".join(EMPTY_CONS) + r")$", flags=re.IGNORECASE)

def clean_disadvantages(text: str) -> str:
    if not isinstance(text, str):
        return np.nan
    t = text.strip()
    if not t:
        return np.nan
    if EMPTY_CONS_RE.match(t):
        return np.nan
    return t


def compose_text(row: pd.Series) -> str:
    parts = []
    comm = row.get("comment")
    if isinstance(comm, str) and comm.strip():
        parts.append(comm.strip())

    adv = row.get("advantages")
    if isinstance(adv, str) and adv.strip():
        parts.append(adv.strip())

    cons = row.get("cons_clean")
    if isinstance(cons, str) and cons.strip():
        parts.append(cons.strip())

    return ". ".join(parts)


MONTHS = {
    "января": 1, "февраля": 2, "марта": 3,
    "апреля": 4, "мая": 5, "июня": 6,
    "июля": 7, "августа": 8, "сентября": 9,
    "октября": 10, "ноября": 11, "декабря": 12,
}

def parse_date_manual(s, default_year=None):
    if s is None or not isinstance(s, str):
        return pd.NaT
    s = s.strip()
    if not s:
        return pd.NaT

    today = datetime.now().date()
    lower = s.lower()
    if lower.startswith("сегодня"):
        return pd.Timestamp(today)
    if lower.startswith("вчера"):
        return pd.Timestamp(today - timedelta(days=1))

    # берём часть до запятой/точки по центру (убираем время/«дополнен»)
    part = re.split(r"[,··]", s, 1)[0].strip()

    m = re.search(r"(20\d{2})", part)
    if m:
        year = int(m.group(1))
        part = part.replace(m.group(1), "").strip()
    else:
        year = default_year or today.year

    try:
        day_str, mon_str = part.split()
        day = int(day_str)
        month = MONTHS[mon_str.lower()]
    except Exception:
        return pd.NaT

    try:
        return pd.Timestamp(year=year, month=month, day=day)
    except ValueError:
        return pd.NaT


def main():
    df = load_source()

    # cons_clean
    if "disadvantages" not in df.columns:
        df["disadvantages"] = np.nan
    df["cons_clean"] = df["disadvantages"].apply(clean_disadvantages)

    # text
    if "advantages" not in df.columns: df["advantages"] = np.nan
    if "comment" not in df.columns: df["comment"] = np.nan
    df["text"] = df.apply(compose_text, axis=1)

    # финальные колонки
    if "review_date" not in df.columns:
        raise ValueError("В файле нет колонки 'review_date'")
    if "rating" not in df.columns:
        raise ValueError("В файле нет колонки 'rating'")

    df_final = df[["text", "review_date", "rating"]].copy().reset_index(drop=True)
    df_final = df_final.rename(columns={"review_date": "date"})

    # парсим дату и фильтруем c мая 2025
    df_final["date_parsed"] = df_final["date"].apply(lambda x: parse_date_manual(x))
    df_final = df_final.dropna(subset=["date_parsed"]).reset_index(drop=True)
    df_final = df_final[(df_final["date_parsed"].dt.year >= 2025) & (df_final["date_parsed"].dt.month >= 5)].reset_index(drop=True)
    df_final = df_final.drop(columns=["date"]).rename(columns={"date_parsed": "date"})

    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUT_PATH, index=False)
    print(f"[OK] rows: {len(df_final)}  -> {OUT_PATH}")

    # показать первые строки в консоли
    with pd.option_context('display.max_colwidth', 120):
        print(df_final.head(5))

if __name__ == "__main__":
    main()
