FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV ST_MODEL_PATH=/app/data/models/paraphrase-multilingual-MiniLM-L6-v2

COPY requirements.txt /app/
RUN pip install --no-cache-dir --default-timeout=120 --retries 5 -r requirements.txt

COPY api /app/api
COPY core /app/core
COPY web /app/web
COPY data /app/data

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
