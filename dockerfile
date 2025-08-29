FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# актуальные названия пакетов для Debian trixie
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libjpeg62-turbo libpng16-16 libfreetype6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r requirements.txt

COPY . /app

EXPOSE 3000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3000"]
