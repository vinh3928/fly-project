FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY python/pyproject.toml python/README.md /app/python/
COPY python/flybrain /app/python/flybrain
COPY python/scripts /app/python/scripts
COPY python/configs /app/python/configs
COPY python/tests /app/python/tests

WORKDIR /app/python
RUN pip install --no-cache-dir -e ".[dev]"

ENV PYTHONUNBUFFERED=1
CMD ["pytest", "-q"]
