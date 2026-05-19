# Lenta Tech — веб-сервис распознавания ценников (GPU/CUDA).
#
# База: официальный pytorch-образ с CUDA — torch+cudnn уже собраны под GPU,
# не тратим время сборки и не рискуем ABI-несовместимостью.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # пакет — это mlcore/ (там лежит `src/`)
    PYTHONPATH=/app/mlcore \
    # кэш HF — на volume (см. docker-compose), чтобы не качать VLM повторно
    HF_HOME=/root/.cache/huggingface \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

# Системные зависимости: ffmpeg (декод видео), libGL/glib (opencv).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Сначала только requirements — кэш слоя pip переживает правки кода.
COPY webapp/requirements-webapp.txt /app/webapp/requirements-webapp.txt
RUN pip install --upgrade pip \
    && pip install -r /app/webapp/requirements-webapp.txt

# Боевой ML-пакет + веб-слой.
COPY mlcore /app/mlcore
COPY webapp /app/webapp

EXPOSE 7860

# Веса (best.pt / LoRA) монтируются как volume в /models в рантайме —
# в образ НЕ зашиваются (см. docker-compose.yml и models/README.md).
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=5 \
    CMD curl -fsS http://localhost:7860/ >/dev/null || exit 1

CMD ["python", "/app/webapp/app.py"]
