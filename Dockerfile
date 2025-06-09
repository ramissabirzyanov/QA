FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    curl \
    python3-dev \
    libopenblas-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry

ENV POETRY_VIRTUALENVS_CREATE=false

ENV LLAMA_CPP_PYTHON_FORCE_CMAKE=1
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=off"

WORKDIR /bot

COPY pyproject.toml poetry.lock ./

RUN poetry install --only main --no-root

COPY . /bot
