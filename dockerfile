FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# базовые либы для numpy/pandas/transformers (минимум для slim)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libjpeg62-turbo libpng16-16 libfreetype6 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r requirements.txt

# кладём исходники
COPY . /app

# ---------- прогрев модели ----------
# во время сборки скачиваем и кешируем веса HF (в $HF_HOME),
# чтобы на первом запросе ничего не подтягивать из сети
RUN python - <<'PY'
from preprocess import sentiment_analysis as sa
try:
    sa.ensure_model()
    print("[Docker build] sentiment model preloaded to cache")
except Exception as e:
    print("[Docker build] preload failed:", repr(e))
PY
# ------------------------------------

EXPOSE 3000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3000"]
