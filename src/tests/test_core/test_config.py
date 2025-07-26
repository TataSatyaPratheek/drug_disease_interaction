import pytest
from src.utils.config import AppConfig

def test_config_loading_default():
    """Tests that the config loads with default values."""
    config = AppConfig()
    assert config.database.neo4j_uri == "bolt://localhost:7687"
    # Compare as string to handle AnyHttpUrl normalization
    assert str(config.ollama.url) == "http://localhost:11434/"
    assert "primary_model" in config.hardware["model_recommendations"]

def test_config_loading_from_env(monkeypatch):
    """Tests overriding config with environment variables."""
    monkeypatch.setenv("DDI_DB_NEO4J_URI", "bolt://test-neo4j:7687")
    monkeypatch.setenv("DDI_OLLAMA_URL", "http://test-ollama:11434")

    config = AppConfig()
    assert config.database.neo4j_uri == "bolt://test-neo4j:7687"
    assert str(config.ollama.url) == "http://test-ollama:11434/"
