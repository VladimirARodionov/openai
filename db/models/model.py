import datetime

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, BIGINT, TIMESTAMP, MetaData, VARCHAR

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
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, default=datetime.datetime.now())
    updated_at = Column(TIMESTAMP(timezone=True), nullable=False, default=datetime.datetime.now(), onupdate=datetime.datetime.now())

    def __repr__(self) -> str:
        return f"User(id={self.id!r}"

    def __str__(self) -> str:
        return f"{self.id!r} [{self.name!r}]"
