import logging.config

import pathlib

import openai
import sqlalchemy
import decouple
from sqlalchemy_vectorstores.tokenizers import JiebaTokenize
from sqlalchemy_vectorstores import SqliteDatabase, SqliteVectorStore

def get_env_config() -> decouple.Config:
    """
    Creates and returns a Config object based on the environment setting.
    It uses .env.dev for development and .env for production.
    """
    app_dir_path = pathlib.Path(__file__).resolve().parent
    env_file_name = ".env"
    file_path = app_dir_path / env_file_name

    if not file_path.is_file():
        raise FileNotFoundError(f"Environment file not found: {file_path}")

    return decouple.Config(decouple.RepositoryEnv(file_path))

env_config = get_env_config()

logging.config.fileConfig(fname=pathlib.Path(__file__).resolve().parent / 'logging.ini',
                          disable_existing_loggers=False)
logging.getLogger('aiogram.dispatcher').propagate = False
logging.getLogger('aiogram.event').propagate = False
db_string = 'sqlite:///local.db'
db = sqlalchemy.create_engine(
    db_string,
    **(
        dict(pool_recycle=900, pool_size=100, max_overflow=3)
    )
)
client = openai.Client(api_key=env_config.get('OPEN_AI_TOKEN'))

def embed_func(text: str) -> list[float]:
    return client.embeddings.create(
        input=text,
        model=env_config.get('EMBEDDING_MODEL'),
    ).data[0].embedding

sqlite_db = SqliteDatabase(db_string, fts_tokenizers={"jieba": JiebaTokenize()}, echo=False)
vs = SqliteVectorStore(sqlite_db, dim=1536, embedding_func=embed_func, fts_tokenize="jieba")

superusers = [int(superuser_id) for superuser_id in env_config.get('SUPERUSERS').split(',')]
