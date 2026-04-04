# Tender Search Agent

## Запуск локально

```bash
pip install -r requirements.txt
PYTHONPATH=. uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Открой `http://localhost:8000` — там минимальный UI.

## API

- `GET /health`
- `GET /companies?kind=customer|supplier`
- `POST /search`

Пример `POST /search`:

```json
{
  "user_query": "флешка 16 гб",
  "user_profile": {
    "favorite_categories": ["Usb-накопители твердотельные (флеш-драйвы)"],
    "purchase_history": ["1222958"],
    "recent_actions": ["флеш накопитель smartbuy"]
  },
  "company": { "type": "customer", "name": "..." }
  "page": 1,
  "limit": 5
}
```

## Запуск в Docker

```bash
docker compose up --build
```

При первом запуске база `search.db` создаётся из CSV внутри контейнера, если файл ещё не существует.
Чтобы ускорить пересборки, можно заранее положить `data/search.db` — он будет подхвачен через volume `./data:/app/data`.

## CatBoost персонализация

1) Убедиться, что создана БД (если запуск вне Docker):
```bash
python /Users/mikhail/Documents/PythonProject/hack_tender_1/core/build_db.py
```

2) Собрать датасет:
```bash
python /Users/mikhail/Documents/PythonProject/hack_tender_1/core/build_dataset.py --max-positives 200000 --neg-per-pos 5
```

3) Обучить модель:
```bash
python /Users/mikhail/Documents/PythonProject/hack_tender_1/core/train_model.py --iters 200 --depth 6 --lr 0.1
```

3) Положить модель в `data/model.cbm` (автоматически создаётся скриптом).

CatBoost используется только если модель существует. Иначе работает эвристика.

## Производительность

- CatBoost использует все ядра по умолчанию (через `CATBOOST_THREADS`).
- Для локального старта можно задать:
```bash
export CATBOOST_THREADS=10
```

Данные читаются из `./data`. Контейнер монтирует их в `/app/data` (read‑only) и хранит кэш в volume `cache`.
