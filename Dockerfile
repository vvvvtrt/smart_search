FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY api /app/api
COPY core /app/core
COPY web /app/web
COPY data /app/data
RUN python /app/core/build_db.py

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
