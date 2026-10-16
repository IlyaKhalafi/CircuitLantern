ARG PYTHON_BASE=python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1
FROM ${PYTHON_BASE}
ARG TORCH_VERSION=2.14.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HOME=/data/huggingface HF_HUB_CACHE=/data/hub LANTERN_DATA_DIR=/data
COPY requirements.docker.txt ./
RUN python3 -m pip install --no-cache-dir "torch==${TORCH_VERSION}" --index-url ${TORCH_INDEX_URL} \
    && python3 -m pip install --no-cache-dir -r requirements.docker.txt
ARG PYTHON_BASE
LABEL org.opencontainers.image.base.name="${PYTHON_BASE}" \
      io.circuitlantern.torch.version="${TORCH_VERSION}" \
      io.circuitlantern.torch.index="${TORCH_INDEX_URL}"
RUN if ! command -v cc >/dev/null 2>&1; then apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*; fi
COPY pyproject.toml ./
COPY circuit_lantern ./circuit_lantern
RUN python3 -m pip install --no-cache-dir --no-deps . \
    && python3 -c "import torch, transformers, fastapi, bitsandbytes; from circuit_lantern.main import app; assert app"
ENTRYPOINT []
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health',timeout=3)"
CMD ["python3", "-m", "uvicorn", "circuit_lantern.main:app", "--host", "0.0.0.0", "--port", "8000"]
