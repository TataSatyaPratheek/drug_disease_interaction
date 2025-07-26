
# src/utils/config.py - ENHANCED VERSION
from pydantic_settings import BaseSettings
from pydantic import Field, validator
import json
from typing import Dict, Any, Optional
from pathlib import Path

class DatabaseConfig(BaseSettings):
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="123lol123")
    weaviate_url: str = Field(default="http://localhost:8080")
    
    class Config:
        env_file = ".env.optimized"
        env_prefix = "DDI_"

class LoggingConfig(BaseSettings):
    level: str = Field(default="INFO")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_path: Optional[str] = Field(default="logs/hybrid_rag.log")

class APIConfig(BaseSettings):
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    workers: int = Field(default=1)
    reload: bool = Field(default=False)

class AppConfig(BaseSettings):
    database: DatabaseConfig = DatabaseConfig()
    logging: LoggingConfig = LoggingConfig() 
    api: APIConfig = APIConfig()
    hardware: Dict[str, Any] = {}
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load hardware optimization
        hardware_config_path = Path("config/hardware_optimized.json")
        if hardware_config_path.exists():
            with open(hardware_config_path) as f:
                self.hardware = json.load(f)
    
    @validator('hardware')
    def validate_hardware_config(cls, v):
        required_keys = ['threading_config', 'model_recommendations', 'llm_config']
        for key in required_keys:
            if key not in v:
                raise ValueError(f"Missing required hardware config key: {key}")
        return v

# Global settings
settings = AppConfig()
