import re
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

# Настройки (удобно менять в одном месте)
# Сколько последних дней оставлять. None → не ограничивать (все даты)
LAST_DAYS: int | None = None
TRIM_ACTIVE_PERIOD: bool = False
TRIM_PARAMS = dict(max_gap_days=21, min_span_days=60, min_weekly_reviews=0)

#пути по умолчанию
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


EMPTY_CONS = [
    r"^нет$", r"^нету$", r"^-+$",
    r"^(?:нет(?:у)?\s+минусов?)$",
    r"^(?:все|всё)\s+(?:ок|отлично)$",
    r"^ничего$", r"^без\s+минус(?:ов)?$",
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


def trim_active_period(df: pd.DataFrame,
                       *,
                       max_gap_days: int = 21,
                       min_span_days: int = 60,
                       min_weekly_reviews: int = 0) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.sort_values("date").reset_index(drop=True)
    uniq = pd.Series(sorted(df["date"].dt.normalize().unique()))
    if len(uniq) == 0:
        return df

    last = uniq.iloc[-1]
    span_start = uniq.iloc[-1]

    for i in range(len(uniq) - 2, -1, -1):
        gap = (last - uniq.iloc[i]).days
        if gap > max_gap_days and (last - span_start).days >= min_span_days:
            span_start = uniq.iloc[i + 1]
            break
        span_start = uniq.iloc[i]

    if (last - span_start).days < min_span_days:
        target_start = last - pd.Timedelta(days=min_span_days)
        span_start = uniq[uniq >= target_start].min() if (uniq >= target_start).any() else uniq.min()

    df_tail = df[df["date"].dt.normalize() >= span_start].copy()

    if min_weekly_reviews > 0 and not df_tail.empty:
        cnt = (
            df_tail.set_index("date").assign(_one=1)["_one"]
                   .resample("D").sum()
                   .rolling(7, min_periods=1).sum()
        )
        active_dates = cnt[cnt >= min_weekly_reviews].index
        if len(active_dates) > 0:
            first_active = active_dates.min()
            df_tail = df_tail[df_tail["date"] >= first_active]

    return df_tail.reset_index(drop=True)


def main():
    df = load_source()

    # cons_clean
    if "disadvantages" not in df.columns:
        df["disadvantages"] = np.nan
    df["cons_clean"] = df["disadvantages"].apply(clean_disadvantages)

    # text
    if "advantages" not in df.columns:
        df["advantages"] = np.nan
    if "comment" not in df.columns:
        df["comment"] = np.nan
    df["text"] = df.apply(compose_text, axis=1)

    # финальные колонки
    if "review_date" not in df.columns:
        raise ValueError("В файле нет колонки 'review_date'")
    if "rating" not in df.columns:
        raise ValueError("В файле нет колонки 'rating'")

    df_final = df[["text", "review_date", "rating"]].copy().reset_index(drop=True)
    df_final = df_final.rename(columns={"review_date": "date"})

    # ---- парсим дату (БЕЗ жёсткого фильтра «с мая 2025») ----
    df_final["date_parsed"] = df_final["date"].apply(parse_date_manual)
    df_final = df_final.dropna(subset=["date_parsed"]).reset_index(drop=True)
    df_final = df_final.drop(columns=["date"]).rename(columns={"date_parsed": "date"})

    # ---- ограничение по окну последних N дней (если задано) ----
    if LAST_DAYS is not None and not df_final.empty:
        last_date = df_final["date"].max().normalize()
        cutoff = last_date - pd.Timedelta(days=LAST_DAYS)
        df_final = df_final[df_final["date"] >= cutoff].reset_index(drop=True)

    # ---- опциональная обрезка «активного периода» ----
    if TRIM_ACTIVE_PERIOD:
        df_final = trim_active_period(df_final, **TRIM_PARAMS)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUT_PATH, index=False)
    print(f"[OK] rows: {len(df_final)}  -> {OUT_PATH}")

    with pd.option_context('display.max_colwidth', 120):
        print(df_final.head(5))


def preprocess_data(input_path: str, output_path: str = "data/airpods_clean.csv") -> str:
    """
    Унифицированный интерфейс: читает Excel/CSV, чистит и сохраняет в output_path.
    Возвращает путь к сохранённому файлу.
    """
    global IN_PRIMARY, IN_FALLBACK, OUT_PATH, OUT_DIR
    IN_PRIMARY = Path(input_path)
    IN_FALLBACK = Path(input_path)
    OUT_PATH = Path(output_path)
    OUT_DIR = OUT_PATH.parent
    main()
    return str(OUT_PATH.resolve())

if __name__ == "__main__":
    main()
