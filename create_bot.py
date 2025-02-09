import logging
import pathlib

import decouple
from logging import config as logging_config

def get_env_config() -> decouple.Config:
    """
    Creates and returns a Config object based on the environment setting.
    It uses .env.dev for development and .env for production.
    """
    env_files = {
        "DEVELOPMENT": ".env.dev",
        "PRODUCTION": ".env",
    }

    app_dir_path = pathlib.Path(__file__).resolve().parent
    env_file_name = ".env"
    file_path = app_dir_path / env_file_name

    if not file_path.is_file():
        raise FileNotFoundError(f"Environment file not found: {file_path}")

    return decouple.Config(decouple.RepositoryEnv(file_path))

env_config = get_env_config()

def get_logger(name) -> logging.Logger:
    logging_config.fileConfig("file_config_local.ini")
    LOGGER_LEVEL = env_config.get('LOGGER_LEVEL')
    logging.basicConfig(level=LOGGER_LEVEL)

    return logging.getLogger(name)
