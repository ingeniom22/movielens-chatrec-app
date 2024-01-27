FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install poetry && \
  poetry config virtualenvs.create false && \
  poetry install --no-interaction --no-ansi --only main

CMD exec uvicorn app.main:app --host 0.0.0.0 --port $PORT