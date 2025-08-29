FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# только нужные либы — оставь как есть
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libjpeg62-turbo libpng16-16 libfreetype6 \
 && rm -rf /var/lib/apt/lists/*

# ВАЖНО: в requirements.txt НЕТ строки "torch==..."
COPY requirements.txt /app/requirements.txt
RUN python -m pip install -U pip setuptools wheel \
 && pip install -r requirements.txt

# Ставим CPU-версию torch (без CUDA)
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.3.1

# Страховка: снесём любые случайно приехавшие nvidia-* пакеты
RUN python - <<'PY'
import pkgutil, subprocess
pkgs=[m.name for m in pkgutil.iter_modules() if m.name.startswith('nvidia')]
if pkgs:
    subprocess.run(['pip','uninstall','-y',*pkgs], check=False)
print("Removed nvidia pkgs:", pkgs)
PY

# Проверка: убеждаемся, что CUDA нет
RUN python - <<'PY'
import os, torch
assert not os.path.isdir('/usr/local/lib/python3.11/site-packages/nvidia'), "Found CUDA nvidia/* in site-packages"
assert not torch.cuda.is_available(), "CUDA unexpectedly available"
print("CPU build OK")
PY

COPY . /app

# (необязательно, но полезно)
RUN useradd -m appuser
USER appuser

EXPOSE 3000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3000"]
