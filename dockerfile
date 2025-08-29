FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Минимум системных либ (оставь только нужные)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libjpeg62-turbo libpng16-16 libfreetype6 \
 && rm -rf /var/lib/apt/lists/*

# 1) Ставим обычные зависимости проекта
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install -r requirements.txt

# 2) ЖЁСТКО ставим CPU-версию PyTorch (без CUDA)
# Удали torchvision/torchaudio если не нужны
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.1 torchvision torchaudio

# Код
COPY . /app

EXPOSE 3000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3000"]
