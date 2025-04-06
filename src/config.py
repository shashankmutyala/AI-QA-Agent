import os
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CrawlerConfig:
    max_depth: int = 3
    max_pages: int = 100
    rate_limit: float = 0.5
    timeout: int = 10
    max_workers: int = 5
    user_agent: str = 'Mozilla/5.0 (compatible; HelpWebsiteQAAgent/1.0)'
    respect_robots_txt: bool = True
    follow_redirects: bool = True
    headers: Dict[str, str] = field(default_factory=lambda: {
        'User-Agent': 'Mozilla/5.0 (compatible; HelpWebsiteQAAgent/1.0)'
    })


@dataclass
class IndexingConfig:
    chunk_size: int = 512
    chunk_overlap: int = 128
    embedding_model: str = "all-MiniLM-L6-v2"
    min_chunk_size: int = 50
    vector_dimension: int = 384
    batch_size: int = 32


@dataclass
class QueryConfig:
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k: int = 5
    confidence_threshold: float = 0.7
    min_query_length: int = 3
    max_query_length: int = 500
    require_question_format: bool = False


@dataclass
class OutputConfig:
    show_sources: bool = True
    show_confidence: bool = True
    max_sources: int = 3
    min_confidence_for_source: float = 0.5
    default_format: str = "terminal"


@dataclass
class StorageConfig:
    storage_dir: str = "data/storage"
    index_name: str = "help_website_index"
    use_cache: bool = True
    cache_dir: str = "data/cache"
    cache_expiration: int = 86400


@dataclass
class AppConfig:
    crawler: CrawlerConfig = field(default_factory=CrawlerConfig)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    debug: bool = False
    log_level: str = "INFO"


def load_config(config_path: Optional[str] = None) -> AppConfig:
    config = AppConfig()
    if config_path:
        try:
            path = Path(config_path)
            if path.exists():
                with open(path, 'r') as f:
                    config_data = json.load(f)
                for section in ['crawler', 'indexing', 'query', 'output', 'storage']:
                    if section in config_data:
                        setattr(config, section, globals()[f"{section.capitalize()}Config"](**config_data[section]))
                logger.info(f"Loaded configuration from {config_path}")
            else:
                logger.warning(f"Configuration file {config_path} not found. Using defaults.")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file {config_path}: {str(e)}")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
    _update_from_env(config)
    _configure_logging(config.log_level, config.debug)
    return config


def _update_from_env(config: AppConfig) -> None:
    env_mappings = {
        'CRAWLER_MAX_DEPTH': 'crawler.max_depth',
        'CRAWLER_MAX_PAGES': 'crawler.max_pages',
        'INDEXING_MODEL': 'indexing.embedding_model',
        'QUERY_CONFIDENCE_THRESHOLD': 'query.confidence_threshold',
        'STORAGE_DIR': 'storage.storage_dir',
    }
    for env_key, attr_path in env_mappings.items():
        if value := os.environ.get(env_key):
            section, attr = attr_path.split('.')
            setattr(getattr(config, section), attr, value)


def _configure_logging(log_level: str, debug: bool) -> None:
    level = logging.DEBUG if debug else getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def save_config(config: AppConfig, config_path: str) -> None:
    try:
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        config_dict = asdict(config)
        config_dict['metadata'] = {
            'saved_at': datetime.now().isoformat(),
            'environment': os.getenv('APP_ENV', 'development'),
        }
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Configuration saved to {config_path}")
    except Exception as e:
        logger.error(f"Error saving configuration: {str(e)}")