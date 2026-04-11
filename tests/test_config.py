from edge_hippo.config import Settings


def test_settings_ignore_unrelated_env_entries(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "TAVILY_API_KEY=test-key\n"
        "UNRELATED_RELEASE_FLAG=true\n",
        encoding="utf-8",
    )

    settings = Settings()

    assert settings.DATA_DIR.exists()
