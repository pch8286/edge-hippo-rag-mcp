import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.search_proof_candidates import _load_candidates


def test_load_candidates_returns_list(tmp_path: Path):
    payload = [
        {
            "fixture_id": "demo",
            "docs": [{"id": "a", "text": "hello"}],
            "query": "question",
            "target_doc_id": "a",
            "target_substring": "hello",
        }
    ]
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = _load_candidates(path)

    assert isinstance(loaded, list)
    assert loaded[0]["fixture_id"] == "demo"
