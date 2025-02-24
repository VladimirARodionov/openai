import datetime

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, BIGINT, TIMESTAMP, MetaData, VARCHAR, BOOLEAN, TEXT

Base = declarative_base()

convention = {
    'all_column_names': lambda constraint, table: '_'.join([
        column.name for column in constraint.columns.values()
    ]),
    'ix': 'ix__%(table_name)s__%(all_column_names)s',
    'uq': 'uq__%(table_name)s__%(all_column_names)s',
    'ck': 'ck__%(table_name)s__%(constraint_name)s',
    'fk': (
        'fk__%(table_name)s__%(all_column_names)s__'
        '%(referred_table_name)s'
    ),
    'pk': 'pk__%(table_name)s'
}
metadata = MetaData(naming_convention=convention)


# Список пользователей
class User(Base):
    __tablename__ = 'user'
    metadata = metadata
    id = Column(BIGINT, primary_key=True)
    name = Column(VARCHAR, nullable=True)
    search_from_inet = Column(BOOLEAN, nullable=False, default=False)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, default=datetime.datetime.now())
    updated_at = Column(TIMESTAMP(timezone=True), nullable=False, default=datetime.datetime.now(), onupdate=datetime.datetime.now())

    def __repr__(self) -> str:
        return f"User(id={self.id!r}, name={self.name!r})"

    def __str__(self) -> str:
        return f"{self.id!r} [{self.name!r}]"

class History(Base):
    __tablename__ = 'history'
    metadata = metadata
    id = Column(BIGINT, primary_key=True)
    user_id = Column(BIGINT, nullable=False)
    search_text = Column(TEXT, nullable=False)
    answer_text = Column(TEXT, nullable=False)
    is_error = Column(BOOLEAN, nullable=False, default=False)
    search_type = Column(VARCHAR, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, default=datetime.datetime.now())
    updated_at = Column(TIMESTAMP(timezone=True), nullable=False, default=datetime.datetime.now(), onupdate=datetime.datetime.now())