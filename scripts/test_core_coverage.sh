#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${PYENV_BIN:-}" ]]; then
  TEST_RUNNER=("$PYENV_BIN" exec python)
elif command -v pyenv >/dev/null 2>&1; then
  TEST_RUNNER=(pyenv exec python)
else
  TEST_RUNNER=(python3)
fi

# Coverage follows the preferred seahorse namespace.
"${TEST_RUNNER[@]}" -m pytest -q \
  tests/test_reranker.py \
  tests/test_retrieval.py \
  --cov=seahorse.reranker \
  --cov=seahorse.retrieval \
  --cov-report=term-missing \
  --cov-fail-under=85
