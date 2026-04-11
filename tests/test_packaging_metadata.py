"""Packaging metadata tests for the Seahorse rebrand."""

from pathlib import Path
import json
import tomllib


ROOT = Path(__file__).resolve().parent.parent


def _load_pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def test_project_name_and_entrypoints_prefer_seahorse():
    project = _load_pyproject()["project"]
    scripts = project["scripts"]
    dependencies = set(project["dependencies"])

    assert project["name"] == "seahorse-rag-mcp"
    assert project["readme"] == "README.md"
    assert project["license"]["file"] == "LICENSE"
    assert "pydantic-settings>=2.0.0" in dependencies
    assert scripts["seahorse"] == "seahorse.cli:main"
    assert scripts["seahorse-server"] == "seahorse.server:main"

    # Legacy command names stay installed, but they now resolve through the
    # preferred public seahorse namespace.
    assert scripts["pam"] == "seahorse.cli:main"
    assert scripts["pam-server"] == "seahorse.server:main"
    assert scripts["hippo"] == "seahorse.cli:main"
    assert scripts["edge-hippo-server"] == "seahorse.server:main"


def test_package_discovery_keeps_legacy_internal_modules():
    package_find = _load_pyproject()["tool"]["setuptools"]["packages"]["find"]
    assert package_find["include"] == ["seahorse*", "edge_hippo*", "memory_crud*"]


def test_openclaw_preset_prefers_seahorse_server():
    preset = json.loads((ROOT / "presets" / "openclaw.json").read_text())
    assert preset["name"] == "seahorse-rag-mcp"
    assert preset["server"]["args"][1] == "git+https://github.com/pch8286/seahorse-rag-mcp.git"
    assert preset["server"]["args"][-1] == "seahorse-server"
