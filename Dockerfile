FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
        libjpeg62-turbo-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU-only torch to keep the image small; drop --index-url to use the
# pinned versions in requirements.txt verbatim if you need GPU support.
RUN pip install --no-cache-dir -r requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu

COPY src/ src/
COPY app/ app/
COPY monitoring/ monitoring/
COPY artifacts/model.pt artifacts/model.pt

ENV MODEL_PATH=artifacts/model.pt
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
