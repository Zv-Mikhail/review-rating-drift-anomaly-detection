FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Только реально нужные системные либы
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libjpeg62-turbo libpng16-16 libfreetype6 \
 && rm -rf /var/lib/apt/lists/*

# Ставим обычные зависимости проекта
COPY requirements.txt /app/requirements.txt
RUN python -m pip install -U pip setuptools wheel \
 && pip install -r requirements.txt

# ЖЁСТКО ставим CPU-версию PyTorch
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.1

# Код
COPY . /app

# Непривилегированный пользователь (по желанию)
RUN useradd -m appuser
USER appuser

EXPOSE 3000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3000"]
