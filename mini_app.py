from __future__ import annotations
import time
from typing import Optional
from fastapi import FastAPI, Response, Query, Request
from fastapi.responses import HTMLResponse
from analysis import (
    load_scored_df,
    build_daily_metrics,
    build_weekly_metrics,
    compute_summary,
    check_negative_day_universal_pneg,
    detect_negative_drift,
    last_day_alert_bad_and_rating_down,
    compute_daily_week_stats,
    make_pnegative_plot_png,
)


app = FastAPI(title="Mini Review Dashboard")


@app.get("/plot.png")
def plot_png():
    df = load_scored_df()
    png = make_pnegative_plot_png(df)
    return Response(content=png, media_type="image/png")


@app.get("/api/check")
def api_check(date: str = Query(..., description="YYYY-MM-DD")):
    df = load_scored_df()
    neg = check_negative_day_universal_pneg(date, df)
    drift = detect_negative_drift(df, window=7, end_date=date, alpha=0.05)
    return {"negative_day": neg, "drift": drift}


@app.get("/", response_class=HTMLResponse)
def index(request: Request, date: Optional[str] = None):
    df = load_scored_df()

    s = compute_summary(df)
    dw = compute_daily_week_stats(df, days=7)
    week_neg_count = round(dw.week_neg_expected)

    flag, badge_msg = last_day_alert_bad_and_rating_down(df, alpha=0.05)
    badge_html = f'<div class="badge">⚠️ {badge_msg}</div>' if flag else ""

    result_block = ""
    if date:
        try:
            neg = check_negative_day_universal_pneg(date, df)
            drift = detect_negative_drift(df, window=7, end_date=date, alpha=0.05)
            neg_msg = neg.get("message", "Ошибка проверки.")
            result_block = f"""
              <div class="card">
                <div class="title">Результаты проверок за {date}</div>
                <div style="font-size:16px; margin-bottom:6px;"><b>Негативный день:</b> {neg_msg}</div>
                <div style="font-size:16px;"><b>Drift (окно 7 дней):</b> {drift}</div>
              </div>
            """
        except Exception as e:
            result_block = f"""
              <div class="card" style="border-color:#fca5a5; background:#fef2f2;">
                <div class="title" style="color:#b91c1c;">Ошибка проверки даты</div>
                <div style="font-size:15px; color:#7f1d1d;">{type(e).__name__}: {e}</div>
              </div>
            """

    bust = int(time.time())

    html = f"""
    <html>
      <head>
        <meta charset="utf-8" />
        <title>Mini Review Dashboard</title>
        <style>
          :root {{ --border:#e5e7eb; }}
          body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin:24px; }}
          .wrap {{ max-width:1200px; margin:0 auto; }}
          /* --- GRID 3 столбца --- */
          .grid {{
            display:grid;
            grid-template-columns: repeat(3, minmax(260px, 1fr));
            gap:16px;
            margin-bottom:16px;
          }}
          .span2 {{ grid-column: span 2; }}
          .card {{
            padding:14px 16px; border:1px solid var(--border); border-radius:12px;
            box-shadow:0 1px 3px rgba(0,0,0,.04); background:#fff;
          }}
          .title {{ font-size:14px; color:#6b7280; margin-bottom:6px; line-height:1.25; }}
          .value {{ font-size:28px; font-weight:700; }}
          .value-row {{ font-size:22px; font-weight:600; line-height:1.25; }}
          .value-strong {{ font-size:28px; font-weight:800; }}
          .muted {{ color:#6b7280; font-size:13px; }}
          .nowrap {{ white-space:nowrap; }}
          .form {{ margin:12px 0 16px 0; display:flex; gap:8px; align-items:center; flex-wrap:wrap; }}
          input[type=date] {{ padding:8px 10px; border:1px solid var(--border); border-radius:8px; }}
          button {{ padding:8px 14px; border-radius:8px; border:1px solid var(--border); background:#fff; cursor:pointer; }}
          button:hover {{ background:#f9fafb; }}
          img.plot {{ max-width:100%; border:1px solid var(--border); border-radius:8px; }}
          .badge {{ padding:10px 12px; border-radius:10px; background:#FEF3C7; color:#92400E; border:1px solid #FDE68A; margin:8px 0; display:inline-block; }}
          h2, h3 {{ margin:8px 0 12px; }}
        </style>
      </head>
      <body>
        <div class="wrap">
          <h2>Обзор по отзывам</h2>

          <!-- ЕДИНАЯ GRID-СЕТКА: 1 2 3 / 4(=span2) 5 -->
          <div class="grid">
            <!-- 1 -->
            <div class="card">
              <div class="title">Средний рейтинг за весь период</div>
              <div class="value">{s.avg_rating_overall:.2f}</div>
            </div>

            <!-- 2 (без переноса строки + дата не переносится) -->
            <div class="card">
              <div class="title">
                Средний рейтинг за последний день <span class="muted nowrap">({s.last_day})</span>
              </div>
              <div class="value">{s.avg_rating_last_day:.2f}</div>
            </div>

            <!-- 3 (та же ширина, т.к. 3 равных столбца — перестал быть «слишком широким») -->
            <div class="card">
              <div class="title">Всего отзывов</div>
              <div class="value">{s.total_reviews}</div>
            </div>

            <!-- 4: тянется на две колонки -->
            <div class="card span2">
              <div class="title">За последние 7 дней ({dw.week_start} — {dw.week_end})</div>
              <div class="value-row">
                <span class="value-strong">{dw.week_reviews}</span> отзывов,
                из которых ≈ <span class="value-strong">{week_neg_count}</span> негативных
              </div>
            </div>

            <!-- 5: третья колонка -->
            <div class="card">
              <div class="title">Средний негатив за неделю</div>
              <div class="value">{dw.week_neg_share:.1%}</div>
            </div>
          </div>

          {badge_html}

          <h3>Проверка даты</h3>
          <form class="form" method="get" action="/">
            <label for="date">Дата:</label>
            <input type="date" id="date" name="date" />
            <button type="submit">Проверить</button>
            <a href="/" style="margin-left:8px;">сброс</a>
          </form>

          {result_block}

          <h3>p_negative по дням</h3>
          <img class="plot" src="/plot.png?cb={bust}" alt="p_negative plot" />
        </div>
      </body>
    </html>
    """
    return HTMLResponse(html)
