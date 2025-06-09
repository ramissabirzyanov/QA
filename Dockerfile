FROM python:3.11-slim-bullseye

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir poetry

ENV POETRY_VIRTUALENVS_CREATE=false 

WORKDIR /bot

COPY pyproject.toml poetry.lock ./

RUN poetry install --only main --no-root

COPY . /bot