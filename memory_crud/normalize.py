import re
import unicodedata

_CONTROL_RE = re.compile(r"[\x00-\x1F\x7F]")
_SPACE_RE = re.compile(r"\s+")


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value or "")
    normalized = _CONTROL_RE.sub(" ", normalized)
    normalized = _SPACE_RE.sub(" ", normalized).strip()
    return normalized.casefold()


def canonicalize_scope(scope: str) -> str:
    normalized = normalize_text(scope)
    normalized = normalized.replace(":", "_")
    return normalized or "global"


def canonicalize_key(key: str) -> str:
    return normalize_text(key)


def is_phrase_candidate(text: str) -> bool:
    if not (2 <= len(text) <= 64):
        return False
    # Discard if the token has no alphabetic characters (numbers/symbols only).
    return any(ch.isalpha() for ch in text)


def supports_substring_like(text: str) -> bool:
    return len(text) >= 4
