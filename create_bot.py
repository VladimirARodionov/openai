import logging.config

import pathlib

import openai
import sqlalchemy
import typing as t
import decouple
from sqlalchemy_vectorstores.databases import sa_types
from sqlalchemy_vectorstores.tokenizers import JiebaTokenize

class SqliteVector(sqlalchemy.TypeDecorator):
    '''
    a simple sqlalchemy column type representing embeddings in sqlite
    '''
    impl = sqlalchemy.LargeBinary
    cache_ok = True

    def __init__(self, *args: t.Any, dim: int = 1536, **kwargs: t.Any):
        super().__init__(*args, **kwargs)
        self._dim = dim

    def process_bind_param(self, value: t.List[float | int] | None, dialect: sqlalchemy.Dialect) -> bytes:
        from sqlite_vec import serialize_float32

        if value is not None:
            assert len(value) == self._dim, f"the embedding dimension({len(value)}) is not equal to ({self._dim}) in database."
            return serialize_float32(value)

    def process_result_value(self, value: bytes | None, dialect: sqlalchemy.Dialect) -> t.List[float]:
        import struct

        if value is not None:
            return list(struct.unpack(f"{self._dim}f", value))


sa_types.SqliteVector = SqliteVector

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
