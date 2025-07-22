# src/ddi/utils/config.py
import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration manager for the project"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration
        
        Args:
            config_path: Path to config YAML file (optional)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = {}
        
        # Load default config
        self._load_default_config()
        
        # Load config from file if provided
        if config_path:
            self._load_from_file(config_path)
        
        # Override with environment variables
        self._load_from_env()
        
    def _load_default_config(self) -> None:
        """Load default configuration"""
        self.config = {
            "data": {
                "raw_dir": "data/raw",
                "processed_dir": "data/processed",
                "external_dir": "data/external",
                "graph_dir": "data/graph"
            },
            "drugbank": {
                "xml_path": "data/raw/full_database/full_database.xml",
                "vocabulary_path": "data/raw/open_data/drugbank_all_drugbank_vocabulary.csv",
                "structures_path": "data/raw/structures/drugbank_all_structures.sdf",
                "external_links_dir": "data/raw/external_links"
            },
            "models": {
                "save_dir": "models",
                "batch_size": 64,
                "hidden_dim": 256,
                "embedding_dim": 128,
                "num_layers": 2,
                "dropout": 0.1,
                "learning_rate": 0.001,
                "weight_decay": 1e-5,
                "epochs": 100,
                "patience": 10
            },
            "api": {
                "host": "127.0.0.1",
                "port": 8000,
                "debug": True
            }
        }
        
    def _load_from_file(self, config_path: str) -> None:
        """Load configuration from YAML file
        
        Args:
            config_path: Path to config YAML file
        """
        try:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                
            # Update config with file values
            self._update_nested_dict(self.config, file_config)
            self.logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            self.logger.error(f"Error loading config from {config_path}: {str(e)}")
            
    def _load_from_env(self) -> None:
        """Load configuration from environment variables"""
        # Check for data directories
        if os.environ.get('DDI_DATA_RAW_DIR'):
            self.config['data']['raw_dir'] = os.environ.get('DDI_DATA_RAW_DIR')
            
        if os.environ.get('DDI_DATA_PROCESSED_DIR'):
            self.config['data']['processed_dir'] = os.environ.get('DDI_DATA_PROCESSED_DIR')
            
        if os.environ.get('DDI_DATA_GRAPH_DIR'):
            self.config['data']['graph_dir'] = os.environ.get('DDI_DATA_GRAPH_DIR')
            
        # Check for model parameters
        if os.environ.get('DDI_MODEL_BATCH_SIZE'):
            self.config['models']['batch_size'] = int(os.environ.get('DDI_MODEL_BATCH_SIZE'))
            
        if os.environ.get('DDI_MODEL_LEARNING_RATE'):
            self.config['models']['learning_rate'] = float(os.environ.get('DDI_MODEL_LEARNING_RATE'))
            
        # Check for API settings
        if os.environ.get('DDI_API_HOST'):
            self.config['api']['host'] = os.environ.get('DDI_API_HOST')
            
        if os.environ.get('DDI_API_PORT'):
            self.config['api']['port'] = int(os.environ.get('DDI_API_PORT'))
            
        if os.environ.get('DDI_API_DEBUG'):
            self.config['api']['debug'] = os.environ.get('DDI_API_DEBUG').lower() == 'true'
    
    def _update_nested_dict(self, d: Dict, u: Dict) -> Dict:
        """Recursively update nested dictionary
        
        Args:
            d: Dictionary to update
            u: Dictionary with updates
            
        Returns:
            Updated dictionary
        """
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = self._update_nested_dict(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value
        
        Args:
            key: Configuration key (dot-separated for nested keys)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value
        
        Args:
            key: Configuration key (dot-separated for nested keys)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
            
        config[keys[-1]] = value
        
    def get_path(self, key: str, create: bool = False) -> Path:
        """Get path from configuration and optionally create directory
        
        Args:
            key: Configuration key for path
            create: Whether to create directory if it doesn't exist
            
        Returns:
            Path object
        """
        path_str = self.get(key)
        if not path_str:
            raise ValueError(f"Path not found for key: {key}")
            
        path = Path(path_str)
        
        if create:
            path.mkdir(parents=True, exist_ok=True)
            
        return path
    
    def save(self, config_path: str) -> None:
        """Save configuration to YAML file
        
        Args:
            config_path: Path to save config YAML file
        """
        try:
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
                
            self.logger.info(f"Saved configuration to {config_path}")
        except Exception as e:
            self.logger.error(f"Error saving config to {config_path}: {str(e)}")


# Create global config instance
config = Config()


