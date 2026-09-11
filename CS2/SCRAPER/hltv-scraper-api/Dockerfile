FROM python:3.13.7-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt requirements.lock.txt ./
COPY wheels ./wheels
COPY scripts/verify_flasgger_wheel.py ./scripts/verify_flasgger_wheel.py

RUN python scripts/verify_flasgger_wheel.py && \
    python -m pip install --no-cache-dir --no-deps --only-binary=:all: \
        --require-hashes --find-links ./wheels -r requirements.lock.txt && \
    python -m pip check

COPY . .

CMD ["python", "app.py"]
