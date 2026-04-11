JUDGE_OUTPUT_GUIDE = """
Decision JSON v3 contract:
- action: one of create/update/delete/noop
- memory.scope + memory.key are always required for create/update
- create MUST include memory.init.trust and memory.init.strength
- update/delete MUST include target_node_id and affected_keys

Prompt defaults:
- provenance and kind defaults are guidance only
- runtime code must not recalculate provenance/kind defaults
""".strip()
