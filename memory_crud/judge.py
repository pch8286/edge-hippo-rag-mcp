from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .normalize import canonicalize_key, canonicalize_scope


@dataclass
class DecisionValidation:
    valid: bool
    action: str
    reason: Optional[str]
    decision: Dict[str, Any]


def split_mem_key(mem_key: str) -> Tuple[str, str]:
    if ":" not in mem_key:
        return ("global", mem_key)
    scope, canonical = mem_key.split(":", 1)
    return (scope, canonical)


def build_mem_key(scope: str, canonical_key: str) -> str:
    return f"{canonicalize_scope(scope)}:{canonical_key}"


def build_known_keys_payload(
    candidates: Iterable[Dict[str, Any]],
    max_keys: int = 32,
    max_prefixes: int = 12,
) -> Dict[str, List[str]]:
    full_keys: List[str] = []
    prefixes: List[str] = []
    seen_key = set()
    seen_prefix = set()
    for candidate in candidates:
        mem_key = str(candidate.get("mem_key") or "").strip()
        if not mem_key:
            continue
        if mem_key not in seen_key:
            seen_key.add(mem_key)
            full_keys.append(mem_key)
        _, canonical = split_mem_key(mem_key)
        prefix = canonical.split(".", 1)[0].split("/", 1)[0].split("_", 1)[0]
        if prefix and prefix not in seen_prefix:
            seen_prefix.add(prefix)
            prefixes.append(prefix)
        if len(full_keys) >= max_keys and len(prefixes) >= max_prefixes:
            break
    return {"known_keys": full_keys[:max_keys], "known_prefixes": prefixes[:max_prefixes]}


def validate_decision(
    decision: Dict[str, Any],
    candidates: Optional[Iterable[Dict[str, Any]]] = None,
) -> DecisionValidation:
    payload = dict(decision or {})
    action = str(payload.get("action", "noop")).casefold()
    if action not in {"create", "update", "delete", "noop"}:
        return DecisionValidation(False, "noop", "unsupported_action", payload)

    if action == "noop":
        return DecisionValidation(True, "noop", None, payload)

    memory = payload.get("memory") or {}
    key = canonicalize_key(str(memory.get("key") or ""))
    if not key:
        return DecisionValidation(False, "noop", "missing_key", payload)

    if action == "create":
        init = memory.get("init") or {}
        if "trust" not in init or "strength" not in init:
            return DecisionValidation(False, "noop", "missing_init_values", payload)
        return DecisionValidation(True, action, None, payload)

    candidates = list(candidates or [])
    if any("node_id" not in candidate for candidate in candidates):
        return DecisionValidation(False, "noop", "candidate_missing_node_id", payload)

    target_node_id = payload.get("target_node_id")
    if target_node_id not in {candidate["node_id"] for candidate in candidates}:
        return DecisionValidation(False, "noop", "target_not_in_candidates", payload)

    affected_keys = payload.get("affected_keys") or []
    if not affected_keys:
        return DecisionValidation(False, "noop", "empty_affected_keys", payload)

    if action == "update":
        selected = next(candidate for candidate in candidates if candidate["node_id"] == target_node_id)
        existing_scope = str(selected.get("scope") or split_mem_key(str(selected.get("mem_key") or ""))[0])
        existing_key = split_mem_key(str(selected.get("mem_key") or ""))[1]
        if canonicalize_scope(str(memory.get("scope") or existing_scope)) != canonicalize_scope(existing_scope):
            return DecisionValidation(False, "noop", "immutable_scope_violation", payload)
        if key != existing_key:
            return DecisionValidation(False, "noop", "immutable_key_violation", payload)

    return DecisionValidation(True, action, None, payload)
