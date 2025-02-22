import logging.config
import os
import pathlib
import sqlalchemy
import decouple

ENVIRONMENT = os.getenv("ENVIRONMENT", default="DEVELOPMENT")

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
    env_file_name = env_files.get(ENVIRONMENT, ".env.dev")
    file_path = app_dir_path / env_file_name

    if not file_path.is_file():
        raise FileNotFoundError(f"Environment file not found: {file_path}")

    return decouple.Config(decouple.RepositoryEnv(file_path))

env_config = get_env_config()

logging.config.fileConfig(fname=pathlib.Path(__file__).resolve().parent / 'logging.ini',
                          disable_existing_loggers=False)
logging.getLogger('aiogram.dispatcher').propagate = False
logging.getLogger('aiogram.event').propagate = False
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) + "/appdb/"
os.path.join(BASE_DIR, 'local.db')
db_string = 'sqlite:///' + os.path.join(BASE_DIR, 'local.db')
db = sqlalchemy.create_engine(
    db_string,
    **(
        dict(pool_recycle=900, pool_size=100, max_overflow=3)
    )
)
superusers = [int(superuser_id) for superuser_id in env_config.get('SUPERUSERS').split(',')]
START_MENU_TEXT = env_config.get('START_MENU_TEXT')
