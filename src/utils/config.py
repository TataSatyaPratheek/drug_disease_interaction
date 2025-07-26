
# src/utils/config.py - ENHANCED VERSION

# src/utils/config.py - ENHANCED
from pydantic_settings import BaseSettings
from pydantic import Field, validator, AnyHttpUrl
import json
from typing import Dict, Any, Optional
from pathlib import Path

class DatabaseConfig(BaseSettings):
    neo4j_uri: str = Field("bolt://localhost:7687", description="Neo4j Bolt URI")
    neo4j_user: str = Field("neo4j", description="Neo4j username")
    neo4j_password: str = Field("123lol123", description="Neo4j password", repr=False)
    weaviate_url: AnyHttpUrl = Field("http://localhost:8080", description="Weaviate instance URL")

    class Config:
        env_prefix = "DDI_DB_"

class OllamaConfig(BaseSettings):
    url: AnyHttpUrl = Field("http://localhost:11434", description="Ollama API base URL")
    request_timeout: int = Field(120, description="Request timeout in seconds")
    
    class Config:
        env_prefix = "DDI_OLLAMA_"


class AppConfig(BaseSettings):
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    hardware: Dict[str, Any] = {}

    def __init__(self, **values: Any):
        super().__init__(**values)
        self._load_hardware_config()

    def _load_hardware_config(self):
        hardware_config_path = Path("config/hardware_optimized.json")
        if hardware_config_path.exists():
            with open(hardware_config_path) as f:
                self.hardware = json.load(f)
        else:
            # Provide a sensible default if the file is missing
            self.hardware = {
                "threading_config": {"max_workers": 4},
                "model_recommendations": {"primary_model": "phi3:mini"},
                "llm_config": {"temperature": 0.1}
            }

# Global settings instance
settings = AppConfig()
