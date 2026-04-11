"""Compatibility wrapper for edge_hippo.cli."""

from edge_hippo.cli import main

if __name__ == "__main__":
    main()
else:
    from ._compat import alias_module as _alias_module

    _alias_module(__name__, "edge_hippo.cli")
