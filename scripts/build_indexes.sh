#!/usr/bin/env bash
set -euo pipefail

python -m core.indexer build_faiss
python -m core.indexer build_cf
