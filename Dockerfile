FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Установка системных зависимостей для сборки
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    curl \
    python3-dev \
    libopenblas-dev \
    g++ \
    libffi-dev \
    libssl-dev \
    pkg-config \
    libblas-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Установка poetry
RUN pip install --no-cache-dir poetry

# Poetry без виртуальных окружений
ENV POETRY_VIRTUALENVS_CREATE=false

# Флаги для сборки llama-cpp-python
ENV LLAMA_CPP_PYTHON_FORCE_CMAKE=1
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=off -DCMAKE_VERBOSE_MAKEFILE=ON"
ENV NINJA_ARGS="-v"

WORKDIR /bot

# Копируем зависимости проекта
COPY pyproject.toml poetry.lock ./

# Устанавливаем только основные зависимости (без llama-cpp-python)
RUN poetry install --only main --no-root
RUN pip wheel --no-cache-dir --use-pep517 "llama-cpp-python==0.3.9"

# Устанавливаем собранный wheel
RUN pip install --no-cache-dir llama-cpp-python-0.3.9-*.whl

# Копируем остальной код
COPY . /bot
