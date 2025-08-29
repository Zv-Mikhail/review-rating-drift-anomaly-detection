from __future__ import annotations
import time
from typing import Optional, Dict, Any
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, Response, Query, Request, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from analysis import (
    compute_summary,
    check_negative_day_universal_pneg,
    detect_negative_drift,
    last_day_alert_bad_and_rating_down,
    compute_daily_week_stats,
    make_pnegative_plot_png,
)

app = FastAPI(title="Mini Review Dashboard")

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
FINAL_CSV = DATA_DIR / "reviews_scored.csv"        # единый итоговый файл

# --- параметры регулярности/фильтра ---
MIN_REVIEWS_PER_DAY = 3
REGULARITY_WINDOW_DAYS = 90        
REGULARITY_MIN_RATIO = 0.25        

def _cleanup_old_files(folder: Path, keep: int = 0, pattern: str = "*") -> None:
    folder.mkdir(parents=True, exist_ok=True)
    files = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    for f in files[keep:]:
        try:
            if f.is_file():
                f.unlink()
        except Exception:
            pass

def _load_scored_df() -> Optional[pd.DataFrame]:
    try:
        if FINAL_CSV.exists():
            return pd.read_csv(FINAL_CSV, parse_dates=["date"])
    except Exception as e:
        print(f"[load_scored_df] read error: {e}")
        return None
    return None

def _preprocess_xlsx_to_clean_tmp(xlsx_path: str) -> str:
    import preprocess.preprocess_dataset as pp
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp_clean = DATA_DIR / "_tmp_clean.csv"
    if hasattr(pp, "preprocess_data"):
        pp.preprocess_data(str(xlsx_path), str(tmp_clean))
    else:
        xlsx_abs = Path(xlsx_path).resolve()
        pp.IN_PRIMARY = xlsx_abs
        pp.IN_FALLBACK = xlsx_abs
        pp.OUT_DIR = DATA_DIR
        pp.OUT_PATH = tmp_clean
        pp.main()
    return str(tmp_clean.resolve())

def _score_sentiment_and_write_final(clean_csv_path: str) -> str:
    import preprocess.sentiment_analysis as sa
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _cleanup_old_files(DATA_DIR, keep=0, pattern="clean_*.csv")
    _cleanup_old_files(DATA_DIR, keep=0, pattern="scored_*.csv")

    if hasattr(sa, "analyze_sentiment"):
        sa.analyze_sentiment(str(clean_csv_path), str(FINAL_CSV))
    else:
        sa.IN_PATH = Path(clean_csv_path).resolve()
        sa.OUT_PATH = FINAL_CSV
        sa.main()
    return str(FINAL_CSV.resolve())

def _run_full_pipeline_from_xlsx(xlsx_path: Path) -> Dict[str, Any]:
    tmp_clean = _preprocess_xlsx_to_clean_tmp(str(xlsx_path))
    try:
        scored_csv = _score_sentiment_and_write_final(tmp_clean)
    finally:
        try:
            Path(tmp_clean).unlink(missing_ok=True)
        except Exception:
            pass
    return {"scored_csv": scored_csv}

def _regularity_and_filter(df: pd.DataFrame) -> tuple[bool, pd.DataFrame, float]:
    if df.empty or "date" not in df.columns:
        return False, df, 0.0
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    per_day = d.groupby("date").size()
    full_idx = pd.date_range(per_day.index.min(), per_day.index.max(), freq="D")
    per_day_full = per_day.reindex(full_idx, fill_value=0)
    if len(per_day_full) == 0:
        return False, d, 0.0
    last_date = per_day_full.index.max()
    cutoff = last_date - pd.Timedelta(days=REGULARITY_WINDOW_DAYS)
    win = per_day_full[per_day_full.index >= cutoff]
    if len(win) == 0:
        return False, d, 0.0
    dense_ratio = float((win >= MIN_REVIEWS_PER_DAY).sum() / len(win))
    is_regular = dense_ratio >= REGULARITY_MIN_RATIO
    good_days = set(per_day_full[per_day_full >= MIN_REVIEWS_PER_DAY].index.date)
    d_filtered = d[d["date"].dt.date.isin(good_days)].copy()
    return is_regular, d_filtered, dense_ratio

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="Нужен .xlsx файл")
    tmp_path = DATA_DIR / f"uploaded_{int(time.time())}.xlsx"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with tmp_path.open("wb") as f:
        content = await file.read()
        f.write(content)
    try:
        paths = _run_full_pipeline_from_xlsx(tmp_path)
        df_new = _load_scored_df()
        last_date = None
        if df_new is not None and not df_new.empty:
            last_date = pd.to_datetime(df_new["date"]).max().date().isoformat()
        return {
            "status": "ok",
            "scored_csv": paths["scored_csv"],
            "last_date": last_date,
            "message": "ОК: данные из файла обработаны и сохранены."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)

@app.get("/plot.png")
def plot_png():
    df = _load_scored_df()
    if df is None or df.empty:
        from io import BytesIO
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4, 1))
        ax.axis("off")
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", transparent=True)
        plt.close(fig)
        buf.seek(0)
        return Response(content=buf.read(), media_type="image/png")
    is_regular, df_filtered, _ = _regularity_and_filter(df)
    if not is_regular or df_filtered.empty:
        from io import BytesIO
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4, 1))
        ax.axis("off")
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", transparent=True)
        plt.close(fig)
        buf.seek(0)
        return Response(content=buf.read(), media_type="image/png")
    png = make_pnegative_plot_png(df_filtered)
    return Response(content=png, media_type="image/png")

@app.get("/", response_class=HTMLResponse)
def index(request: Request, date: Optional[str] = None):
    df = _load_scored_df()
    if df is None or df.empty:
        return HTMLResponse("""
        <html>
                            
      <head>
        <style>
          body {
            font-family: Arial, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            background-color: #fafafa;
          }
          .container {
            text-align: center;
            background: white;
            padding: 30px 40px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
          }
          h2 {
            margin-bottom: 20px;
          }
                            
          #upload-status {
            display: block;
            margin-top: 15px;
            color: grey;
            font-size: 14px;
          }
        </style>
      </head>
      <body>
        <div class="container">
          <h2>Загрузить файл с отзывами (.xlsx)</h2>
          <div>
            <label for="file">Файл:</label>
            <input type="file" id="file" />
            <button id="btn-upload">Загрузить</button>
            <span id="upload-status"></span>
          </div>
        </div>

        <script>
        async function uploadReviewsFile() {
          const fileInput = document.getElementById('file');
          const statusEl = document.getElementById('upload-status');
          if (!fileInput.files.length) {
            statusEl.textContent = 'Выберите .xlsx файл.';
            return;
          }
          const formData = new FormData();
          formData.append('file', fileInput.files[0]);
          try {
            statusEl.textContent = 'Загрузка...';
            const res = await fetch('/api/upload', { method: 'POST', body: formData });
            const data = await res.json();
            if (!res.ok) {
              statusEl.textContent = 'Ошибка: ' + (data.detail || res.status);
              return;
            }
            statusEl.textContent = 'Файл успешно обработан!';
            setTimeout(() => { window.location.href = '/'; }, 1500);
          } catch (e) {
            statusEl.textContent = 'Сетевая ошибка.';
          }
        }

        document.addEventListener('DOMContentLoaded', function () {
          const btn = document.getElementById('btn-upload');
          if (btn) btn.addEventListener('click', uploadReviewsFile);
        });
        </script>
      </body>
    </html>
        """)
    # --- дальше всё как было (карточки, график, проверки) ---
    # оставляем твой рендеринг summary / plots
    # ...



    s = compute_summary(df)
    dw = compute_daily_week_stats(df, days=7)
    week_neg_count = round(dw.week_neg_expected)

    flag, badge_msg = last_day_alert_bad_and_rating_down(df, alpha=0.05)
    badge_html = f'<div class="badge">⚠️ {badge_msg}</div>' if flag else ""

    is_regular, _df_dense, dense_ratio = _regularity_and_filter(df)
    
    last_date_available = None
    try:
        if df is not None and not df.empty and "date" in df.columns:
            last_date_available = pd.to_datetime(df["date"]).max().date().isoformat()
    except Exception:
        last_date_available = None

    if is_regular and date is None and last_date_available:
        date = last_date_available

    sparse_notice_html = ""
    if not is_regular:
        sparse_notice_html = f"""
          <div class="card" style="border-color:#e5e7eb; background:#fafafa;">
            <div class="title" style="color:#374151;">
              Недостаточно данных для детальной проверки
            </div>
            <div style="font-size:15px; color:#4b5563;">
              Мало отзывов или низкая дневная плотность публикаций за последние {REGULARITY_WINDOW_DAYS} дней.
              Доля «плотных» дней (≥{MIN_REVIEWS_PER_DAY} отзыва/день): {dense_ratio:.0%}.
            </div>
          </div>
        """

    result_block = ""
    date_section_html = ""
    if is_regular:
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
        date_section_html = f"""
          <h3>Проверка даты</h3>
          <form class="form" method="get" action="/">
            <label for="date">Дата:</label>
            <input type="date" id="date" name="date" />
            <button type="submit">Проверить</button>
            <a href="/" style="margin-left:8px;">сброс</a>
          </form>
          {result_block}
        """

    # --- блок "график" (только если данных достаточно) ---
    bust = int(time.time())
    plot_section_html = ""
    if is_regular:
        plot_section_html = f"""
          <h3>Доля негатива по дням</h3>
          <img class="plot" src="/plot.png?cb={bust}" alt="p_negative plot" />
        """

    # --- HTML ---
    html = f"""
    <html>
      <head>
        <meta charset="utf-8" />
        <title>Mini Review Dashboard</title>
        <style>
          :root {{ --border:#e5e7eb; }}
          body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin:24px; }}
          .wrap {{ max-width:1200px; margin:0 auto; }}
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
          input[type=date], input[type=text] {{ padding:8px 10px; border:1px solid var(--border); border-radius:8px; }}
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

          
        <h3>Загрузить файл с отзывами (.xlsx)</h3>
        <form class="form" id="upload-form" enctype="multipart/form-data" onsubmit="return false;">
        <label for="file">Файл:</label>
        <input type="file" id="file" name="file" accept=".xlsx" />
        <button id="btn-upload" type="button">Загрузить</button>
        <span id="upload-status" class="muted" style="margin-left:8px;"></span>
        </form>

        <script>
        async function uploadReviewsFile() {{
            const fileInput = document.getElementById('file');
            const statusEl = document.getElementById('upload-status');

            if (!fileInput.files.length) {{
            statusEl.textContent = 'Выберите .xlsx файл.';
            return;
            }}

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            try {{
            statusEl.textContent = 'Загрузка...';
            const res = await fetch('/api/upload', {{ method: 'POST', body: formData }});
            const data = await res.json();

            if (!res.ok) {{
                statusEl.textContent = 'Ошибка: ' + (data.detail || res.status);
                return;
            }}

            statusEl.textContent = 'OK: ' + (data.message || 'Файл обработан');
            setTimeout(() => {{ window.location.href = '/'; }}, 600);
            }} catch (e) {{
            statusEl.textContent = 'Сетевая ошибка.';
            }}
        }}

        document.addEventListener('DOMContentLoaded', function () {{
            const btn = document.getElementById('btn-upload');
            if (btn) btn.addEventListener('click', uploadReviewsFile);
        }});
        </script>


          <script>
            async function startParse() {{
              const inputEl = document.getElementById('article_id');
              const statusEl = document.getElementById('parse-status');
              const article = (inputEl.value || '').trim();

              statusEl.textContent = '';
              if (!article) {{
                statusEl.textContent = 'Введите артикул.';
                return;
              }}

              try {{
                statusEl.textContent = 'Отправка...';
                const res = await fetch('/api/parse/start', {{
                  method: 'POST',
                  headers: {{'Content-Type': 'application/json'}},
                  body: JSON.stringify({{ article_id: article }})
                }});

                const data = await res.json();
                if (!res.ok) {{
                  statusEl.textContent = "Ошибка: " + (data.detail || res.status);
                  return;
                }}
                statusEl.textContent = "OK: " + data.message;
                // Откроем страницу сразу с последним днём
                setTimeout(() => {{
                  const u2 = new URL(window.location.href);
                  if (data.last_date) {{
                    u2.searchParams.set('date', data.last_date);
                  }} else {{
                    u2.searchParams.delete('date');
                  }}
                  u2.searchParams.set('cb', Date.now().toString());
                  window.location.href = u2.toString();
                }}, 500);
              }} catch (e) {{
                statusEl.textContent = 'Сетевая ошибка.';
              }}
            }}

            document.addEventListener('DOMContentLoaded', function () {{
              const btn = document.getElementById('btn-parse');
              if (btn) btn.addEventListener('click', startParse);
            }});
          </script>

          <!-- Основные карточки -->
          <div class="grid">
            <div class="card">
              <div class="title">Средний рейтинг за весь период</div>
              <div class="value">{s.avg_rating_overall:.2f}</div>
            </div>

            <div class="card">
              <div class="title">
                Средний рейтинг за последний день <span class="muted nowrap">({s.last_day})</span>
              </div>
              <div class="value">{s.avg_rating_last_day:.2f}</div>
            </div>

            <div class="card">
              <div class="title">Всего отзывов</div>
              <div class="value">{s.total_reviews}</div>
            </div>

            <div class="card span2">
              <div class="title">За последние 7 дней ({dw.week_start} — {dw.week_end})</div>
              <div class="value-row">
                <span class="value-strong">{dw.week_reviews}</span> отзывов,
                из которых ≈ <span class="value-strong">{week_neg_count}</span> негативных
              </div>
            </div>

            <div class="card">
              <div class="title">Средний негатив за неделю</div>
              <div class="value">{dw.week_neg_share:.1%}</div>
            </div>
          </div>

          {sparse_notice_html}

          {badge_html}

          {date_section_html}

          {plot_section_html}
        </div>
      </body>
    </html>
    """
    return HTMLResponse(html)