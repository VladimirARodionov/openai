import logging

from sqlalchemy.orm import Session

from create_bot import db, superusers
from db.models.model import User

logger = logging.getLogger(__name__)


def is_superuser(user_id: int) -> bool:
    return user_id in superusers


def check_user(user_id: int) -> bool:
    session = Session(db)
    with session.begin():
        user_info = session.get(User, user_id)
        if user_info:
            return True
        else:
            return False

