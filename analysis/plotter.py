# plotter.py
from io import BytesIO
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # без GUI
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def make_pnegative_plot_png(df: pd.DataFrame) -> bytes:
    """
    Принимает сырой df (date, p_negative), строит дневной ряд и SMA(3/7),
    возвращает PNG как bytes.
    """
    ts = (
        df.set_index("date")["p_negative"]
          .resample("D").mean()
          .reset_index(name="avg_p_negative")
    )

    if ts.empty or ts["avg_p_negative"].isna().all():
        fig, ax = plt.subplots(figsize=(8, 2.5))
        ax.text(0.5, 0.5, "Нет данных для графика", ha="center", va="center")
        ax.axis("off")
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    ts["sma3"] = ts["avg_p_negative"].rolling(3, min_periods=1).mean()
    ts["sma7"] = ts["avg_p_negative"].rolling(7, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(ts["date"], ts["avg_p_negative"], marker="o", linestyle="-",
            markersize=4, alpha=0.3, label="Дневное среднее")
    ax.plot(ts["date"], ts["sma3"], linewidth=2, label="Скользящее среднее (3 дня)")
    ax.plot(ts["date"], ts["sma7"], linewidth=2, label="Скользящее среднее (7 дней)")

    
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax.tick_params(axis="x", rotation=45)

    # вертикальные линии строго по понедельникам
    first = pd.to_datetime(ts["date"].min()).normalize()
    last  = pd.to_datetime(ts["date"].max()).normalize()
    monday = first - pd.Timedelta(days=first.weekday())
    week_marks = pd.date_range(monday, last + pd.Timedelta(days=7), freq="W-MON")
    for d in week_marks:
        ax.axvline(d, color="0.85", linestyle="--", linewidth=1, alpha=0.6)

    ax.set_title("Динамика негативных отзывов (p_negative)")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Среднее p_negative")
    max_val = ts[["avg_p_negative", "sma3", "sma7"]].max().max()
    ax.set_ylim(-0.02, min(1, max_val + 0.1))
    ax.legend()
    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return buf.read()
