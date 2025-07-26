# src/utils/config.py - DON'T REINVENT CONFIG MANAGEMENT
from pydantic_settings import BaseSettings
from pydantic import Field
import json
from typing import Dict, Any

class DatabaseConfig(BaseSettings):
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="123lol123")
    weaviate_url: str = Field(default="http://localhost:8080")
    
    class Config:
        env_file = ".env.optimized"

class AppConfig(BaseSettings):
    # Load your existing hardware optimization
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with open("config/hardware_optimized.json") as f:
            self.hardware = json.load(f)
    
    database: DatabaseConfig = DatabaseConfig()
    hardware: Dict[str, Any] = {}

# Global settings
settings = AppConfig()
