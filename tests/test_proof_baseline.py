import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.proof_baseline import _extract_first_passage_text, _load_fixture
def test_extract_first_passage_text():
    raw = (
        "Found 1 seed entities.\n\nRelevant Passages:\n\n"
        "--- [Score: 0.9123] ---\n"
        "ZRAM keeps compressed pages in RAM before swapping to disk.\n"
    )
    assert (
        _extract_first_passage_text(raw)
        == "ZRAM keeps compressed pages in RAM before swapping to disk."
    )


def test_load_fixture_requires_expected_keys(tmp_path: Path):
    fixture = {
        "fixture_id": "demo",
        "docs": [{"id": "a", "text": "hello"}],
        "query": "question",
        "target_doc_id": "a",
        "target_substring": "hello",
    }
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(fixture), encoding="utf-8")

    loaded = _load_fixture(path)

    assert loaded["fixture_id"] == "demo"
    assert loaded["docs"][0]["id"] == "a"
