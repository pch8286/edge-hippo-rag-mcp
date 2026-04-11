"""Compatibility wrapper for edge_hippo.server."""

from edge_hippo.server import main

if __name__ == "__main__":
    main()
else:
    from ._compat import alias_module as _alias_module

    _alias_module(__name__, "edge_hippo.server")
