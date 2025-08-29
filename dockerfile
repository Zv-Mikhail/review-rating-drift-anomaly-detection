FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# только реально нужные системные либы
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libjpeg62-turbo libpng16-16 libfreetype6 \
 && rm -rf /var/lib/apt/lists/*

# зависимости проекта (БЕЗ torch в requirements.txt)
COPY requirements.txt /app/requirements.txt
RUN python -m pip install -U pip setuptools wheel \
 && pip install -r requirements.txt

# Ставим CPU-версию torch ЯВНО и БЕЗ зависимостей (чтобы не притащить nvidia-*)
RUN pip install --no-deps --index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.1

# На всякий случай — вымести, если что-то из nvidia-* уже прилетело
RUN pip freeze | grep -i '^nvidia-' | cut -d'=' -f1 | xargs -r pip uninstall -y && \
    rm -rf /usr/local/lib/python3.11/site-packages/nvidia || true && \
    rm -rf /usr/local/lib/python3.11/dist-packages/nvidia || true && \
    find /usr/local/lib/python3.11/site-packages -maxdepth 2 -iname "*cuda*" -type f -delete || true

# код
COPY . /app

# непривилегированный пользователь (по желанию)
RUN useradd -m appuser
USER appuser

EXPOSE 3000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3000"]
