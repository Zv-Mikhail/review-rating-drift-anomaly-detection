from pathlib import Path
from dataclasses import dataclass
import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp


DATA_PATH = Path("data/airpods_scored.csv")  


def load_scored_df(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Нет файла {path}. Сначала подготовь данные.")
    df = pd.read_csv(path, parse_dates=["date"])
    need = {"date", "rating", "p_negative"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"В данных нет колонок: {missing}")
    df = df.dropna(subset=["date"]).copy()
    return df

def build_daily_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Дневные n, avg_rating, neg_share + 3/7-дневные SMA для neg_share."""
    g = df.groupby(df["date"].dt.date)
    daily = pd.DataFrame({
        "n": g["rating"].count(),
        "avg_rating": g["rating"].mean(),
        "neg_share": g["p_negative"].mean(),
    })
    
    idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D").date
    daily = daily.reindex(idx)
    daily.index.name = "day"
    daily = daily.reset_index().rename(columns={"index": "day"})
    # сглаживание для neg_share
    daily["neg_sma3"] = daily["neg_share"].rolling(3, min_periods=1).mean()
    daily["neg_sma7"] = daily["neg_share"].rolling(7, min_periods=1).mean()
    # флаг малой выборки
    daily["low_n_day"] = (daily["n"].fillna(0) < 3).astype(int)
    return daily

def build_weekly_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    """Недельные агрегаты на основе daily (понедельник — старт)."""
    tmp = daily.copy()
    tmp["day_dt"] = pd.to_datetime(tmp["day"])
    tmp["week_start"] = tmp["day_dt"].dt.to_period("W-MON").apply(lambda r: r.start_time.date())
    gw = tmp.groupby("week_start", as_index=False).agg(
        n=("n", "sum"),
        avg_rating=("avg_rating", "mean"),
        neg_share=("neg_share", "mean"),
    )
    gw["low_n_week"] = (gw["n"].fillna(0) < 10).astype(int)
    return gw

@dataclass
class Summary:
    date_from: str
    date_to: str
    total_reviews: int
    avg_rating_overall: float
    avg_rating_last_day: float
    last_day: str

def compute_summary(df: pd.DataFrame) -> Summary:
    date_from = str(df["date"].min().date())
    date_to   = str(df["date"].max().date())
    total_reviews = int(len(df))
    avg_overall = float(df["rating"].mean())

    last_day = df["date"].dt.date.max()
    avg_last_day = float(df.loc[df["date"].dt.date == last_day, "rating"].mean())

    return Summary(
        date_from=date_from,
        date_to=date_to,
        total_reviews=total_reviews,
        avg_rating_overall=avg_overall,
        avg_rating_last_day=avg_last_day,
        last_day=str(last_day),
    )

def save_metrics_csv(daily: pd.DataFrame, weekly: pd.DataFrame, out_dir: Path = Path("out")):
    out_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out_dir / "metrics_daily.csv", index=False)
    weekly.to_csv(out_dir / "metrics_weekly.csv", index=False)



def check_negative_day_universal_pneg(
    check_date,
    df: pd.DataFrame,
    alpha: float = 0.05,
    baseline_days: int | None = None,
    min_reviews: int = 5,
    volume_ratio: float = 0.3,
):
    """Проверка “негативного дня” по средней p_negative (t-тест одной выборки)."""
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    d = pd.to_datetime(check_date).normalize()

    if "p_negative" not in data.columns:
        return {"ok": False, "message": "Нет колонки 'p_negative'."}

    before = data[data["date"] < d]
    if baseline_days is not None:
        before = before[before["date"] >= d - pd.Timedelta(days=baseline_days)]
    if before.empty:
        return {"ok": False, "message": f"Нет данных до {d.date()} для фона."}

    p0 = float(before["p_negative"].mean())
    avg_n_before = float(before.groupby("date").size().mean())

    today = data[data["date"] == d]
    n = int(len(today))
    if n == 0:
        return {"ok": False, "message": f"Нет отзывов за {d.date()}."}

    p_day = float(today["p_negative"].mean())

    stat, p_val = ttest_1samp(today["p_negative"], popmean=p0)
    p_val = float(p_val)

    is_negative = (p_val < alpha) and (p_day > p0)
    low_volume = (n < min_reviews) or (n < avg_n_before * volume_ratio)

    if is_negative:
        msg = (
            f"День {d.date()} считается негативным — "
            f"средний p_negative {p_day:.1%} против обычного {p0:.1%} "
            f"(p-value={p_val:.3f})."
        )
        if low_volume:
            msg += f" Возможно, связано с малым количеством отзывов ({n} vs средний {avg_n_before:.1f})."
    else:
        msg = (
            f"День {d.date()} не считается негативным — "
            f"средний p_negative {p_day:.1%} против обычного {p0:.1%} "
            f"(p-value={p_val:.3f})."
        )

    return {
        "ok": True,
        "date": str(d.date()),
        "n": n,
        "p_day": p_day,
        "p0": p0,
        "p_value": p_val,
        "is_negative": bool(is_negative),
        "low_volume": bool(low_volume),
        "avg_n_before": avg_n_before,
        "message": msg,
    }

def detect_negative_drift(
    df: pd.DataFrame,
    window: int = 7,
    end_date: pd.Timestamp | str | None = pd.Timestamp.now().floor("D"),
    alpha: float = 0.05,
):
    """
    Drift по p_negative: сравниваем текущее окно (window дней, заканчивается end_date)
    с предыдущим через наклон линейной регрессии и t-статистику (односторонняя).
    Ожидаются колонки 'date' и 'p_negative'. Возвращает строку.
    """

    try:
        from scipy.stats import t as student_t
    except Exception as e:
        return f"scipy не установлен/недоступен: {e}"

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.floor("D")
    daily = (
        df.groupby("date")["p_negative"]
        .mean()
        .rename("avg_p_negative")
        .reset_index()
        .sort_values("date")
    )

    if len(daily) < 2 * window:
        return f"Недостаточно данных: нужно как минимум {2*window} дней, есть {len(daily)}."

    full_idx = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    daily = daily.set_index("date").reindex(full_idx).rename_axis("date").reset_index()

    if end_date is None:
        curr_end = daily["date"].dropna().max()
    else:
        curr_end = pd.to_datetime(end_date).floor("D")
        if curr_end < daily["date"].min() or curr_end > daily["date"].max():
            return (
                f"end_date={curr_end.date()} вне доступного диапазона "
                f"{daily['date'].min().date()} - {daily['date'].max().date()}."
            )

    curr_start = curr_end - pd.Timedelta(days=window - 1)
    prev_end = curr_start - pd.Timedelta(days=1)
    prev_start = prev_end - pd.Timedelta(days=window - 1)

    if prev_start < daily["date"].min():
        return (
            f"Недостаточно предшествующих данных для предыдущего окна: "
            f"нужно {2*window} подряд дней до {curr_end.date()}."
        )

    curr_window = daily[(daily["date"] >= curr_start) & (daily["date"] <= curr_end)].copy()
    prev_window = daily[(daily["date"] >= prev_start) & (daily["date"] <= prev_end)].copy()

    if len(curr_window) < window or len(prev_window) < window:
        return (
            f"Окна неполные: требуются {window} дней в каждом, а есть "
            f"текущих {len(curr_window)}, предыдущих {len(prev_window)}."
        )

    if curr_window["avg_p_negative"].isna().any() or prev_window["avg_p_negative"].isna().any():
        return "В окнах есть пропуски значений, невозможно корректно оценить drift."

    # Линейная регрессия на текущем окне
    
    y = curr_window["avg_p_negative"].to_numpy()
    n = len(y)
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    y_mean = y.mean()
    Sxx = np.sum((x - x_mean) ** 2)
    Sxy = np.sum((x - x_mean) * (y - y_mean))
    slope = Sxy / Sxx if Sxx != 0 else 0.0
    intercept = y_mean - slope * x_mean
    y_hat = intercept + slope * x
    residuals = y - y_hat
    rss = np.sum(residuals ** 2)
    mse = rss / (n - 2) if n - 2 > 0 else 0.0
    se_slope = np.sqrt(mse / Sxx) if Sxx != 0 and mse >= 0 else np.nan

    if se_slope and not np.isnan(se_slope) and se_slope > 0:
        t_stat = slope / se_slope
        dfree = n - 2
        p_one_sided = 1 - student_t.cdf(t_stat, dfree) if slope > 0 else 1.0
    else:
        p_one_sided = np.nan

    mean_prev = prev_window["avg_p_negative"].mean()
    mean_curr = curr_window["avg_p_negative"].mean()
    if mean_prev != 0:
        pct_change = (mean_curr - mean_prev) / mean_prev * 100
        sign = "+" if pct_change >= 0 else "-"
    else:
        pct_change = float("inf")
        sign = "+"

    current_interval = f"с {curr_start.date()} по {curr_end.date()}"
    drift_detected = (slope > 0) and (not np.isnan(p_one_sided)) and (p_one_sided < alpha)

    slope_str = f"{slope:.4f}"
    p_str = f"{p_one_sided:.3f}" if not np.isnan(p_one_sided) else "n/a"
    mean_curr_str = f"{mean_curr:.3f}"
    pct_str = f"{sign}{abs(pct_change):.1f}%"

    if drift_detected:
        return (
            f"За период {current_interval} выявлены отклонения.\n"
            f"Среднее значение: {mean_curr_str}. "
            f"Изменение к прошлому окну: {pct_str}. "
            f"Наклон={slope_str}, p={p_str}."
        )
    else:
        return (
            f"За период {current_interval} существенных отклонений нет.\n"
            f"Среднее значение наклона: {mean_curr_str}. "
            f"Изменение к прошлому окну: {pct_str}."
        )

def last_day_alert_bad_and_rating_down(df: pd.DataFrame, alpha: float = 0.05) -> tuple[bool, str]:
    """
    Возвращает (flag, message):
    - flag=True, если последний день аномален по p_negative (t-тест из check_negative_day_universal_pneg)
      И средний рейтинг в последний день ниже, чем в предыдущий день.
    - message — человекочитаемое описание.
    """
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    if data.empty:
        return False, "Нет данных."

    last_day = data["date"].max()
    prev_day = last_day - pd.Timedelta(days=1)

    res = check_negative_day_universal_pneg(last_day, data, alpha=alpha)
    if not res.get("ok", False) or not res.get("is_negative", False):
        return False, "Последний день не аномален по p_negative."

    if "rating" not in data.columns:
        return True, f"Последний день аномален по p_negative ({last_day.date()}), но нет колонки rating для проверки падения."

    r_last = data.loc[data["date"] == last_day, "rating"].mean()
    r_prev = data.loc[data["date"] == prev_day, "rating"].mean()
    if pd.isna(r_prev):
        return True, f"Последний день аномален по p_negative ({last_day.date()}), пред. дня нет для сравнения рейтинга."

    if r_last < r_prev:
        return True, f"Последний день {last_day.date()} аномален по p_negative и рейтинг снизился ({r_last:.2f} vs {r_prev:.2f})."
    else:
        return False, f"Последний день {last_day.date()} аномален по p_negative, но рейтинг не снизился ({r_last:.2f} vs {r_prev:.2f})."


@dataclass
class DailyWeeklyStats:
    last_day: str
    last_day_reviews: int
    last_day_neg_share: float          # средний p_negative за день
    week_start: str
    week_end: str
    week_reviews: int
    week_neg_expected: float           # сумма p_negative за 7 дней (ожидаемое число «негативов»)
    week_neg_share: float              # доля негатива за 7 дней (sum pneg / count)

def compute_daily_week_stats(df: pd.DataFrame, days: int = 7) -> DailyWeeklyStats:
    """Считает:
      - долю негатива за последний день (avg p_negative)
      - за последние `days` дней: всего отзывов, «ожидаемое» число негативов (sum p_negative) и их долю.
    """
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    if data.empty:
        return DailyWeeklyStats("-", 0, 0.0, "-", "-", 0, 0.0, 0.0)

    last_day_ts = data["date"].max()
    last_day = str(last_day_ts.date())

    day_df = data[data["date"] == last_day_ts]
    day_n = int(len(day_df))
    day_neg_share = float(day_df["p_negative"].mean()) if day_n > 0 else 0.0

    week_start_ts = last_day_ts - pd.Timedelta(days=days-1)
    week_df = data[(data["date"] >= week_start_ts) & (data["date"] <= last_day_ts)]
    week_n = int(len(week_df))
    week_neg_exp = float(week_df["p_negative"].sum()) if week_n > 0 else 0.0
    week_neg_share = float(week_neg_exp / week_n) if week_n > 0 else 0.0

    return DailyWeeklyStats(
        last_day=last_day,
        last_day_reviews=day_n,
        last_day_neg_share=day_neg_share,
        week_start=str(week_start_ts.date()),
        week_end=last_day,
        week_reviews=week_n,
        week_neg_expected=week_neg_exp,
        week_neg_share=week_neg_share,
    )
