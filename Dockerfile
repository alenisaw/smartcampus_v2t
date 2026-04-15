# Dockerfile
# Container image for the SmartCampus V2T demo stack.
FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONUTF8=1
ENV PYTHONFAULTHANDLER=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_ROOT_USER_ACTION=ignore

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
COPY requirements-faiss.txt /app/requirements-faiss.txt
RUN pip install --upgrade pip \
    && pip install -r /app/requirements.txt \
    && pip install -r /app/requirements-faiss.txt


FROM base AS runtime

ENV SMARTCAMPUS_PROFILE=main
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV XDG_CACHE_HOME=/app/.cache
ENV TOKENIZERS_PARALLELISM=false
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

COPY app /app/app
COPY backend /app/backend
COPY configs /app/configs
COPY docker /app/docker
COPY docs /app/docs
COPY scripts /app/scripts
COPY src /app/src
COPY README.md /app/README.md
COPY requirements.txt /app/requirements.txt
COPY requirements-faiss.txt /app/requirements-faiss.txt

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD ["python", "/app/docker/healthcheck_http.py", "http://127.0.0.1:8000/healthz", "--expect-json-ok"]

CMD ["python", "-m", "uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
