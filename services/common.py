import logging

from sqlalchemy.orm import Session

from create_bot import db, superusers
from db.models.model import User
from locale_config import i18n

logger = logging.getLogger(__name__)


def is_superuser(user_id: int) -> bool:
    return user_id in superusers


def is_allowed(user) -> bool:
    session = Session(db)
    try:
        user_info = session.get(User, user.id)
        if user_info:
            if not user_info.name and user.username and not user_info.name == user.username:
                user_info.name = user.username
                session.add(user_info)
                session.commit()
            return True
        else:
            if is_superuser(user.id):
                return True
        return False
    except Exception:
        logger.exception('Ошибка при проверке пользователя')
        session.rollback()
        if is_superuser(user.id):
            return True
        return False
    finally:
        session.close()

def convert_user_id(user_id_string: str) -> int:
    try:
        return int(user_id_string)
    except ValueError:
        return -1

def add_user(user_id_string: str) -> bool:
    user_id = convert_user_id(user_id_string)
    if user_id == -1:
        return False
    session = Session(db)
    try:
        existing_user = session.query(User).filter(User.id == user_id).first()
        if not existing_user:
            user = User(id=user_id)
            session.add(user)
            session.commit()
            return True
    except Exception:
        logger.exception('Ошибка при добавлении пользователя')
        session.rollback()
        return False
    finally:
        session.close()
    return False


def delete_user(user_id_string: str) -> bool:
    user_id = convert_user_id(user_id_string)
    if user_id == -1:
        return False
    session = Session(db)
    try:
        existing_user = session.query(User).filter(User.id == user_id).first()
        if existing_user:
            session.delete(existing_user)
            session.commit()
            return True
    except Exception:
        logger.exception('Ошибка при удалении пользователя')
        session.rollback()
        return False
    finally:
        session.close()
    return False


def list_users() -> str:
    session = Session(db)
    try:
        users = session.query(User).all()
        if not users:
            return i18n.format_value("show_users_text_empty")
        res = "\n".join([f"<code><b>{user.id}</b></code> [{user.name}]" for user in users])
        return res
    except Exception:
        logger.exception('Ошибка при выводе списка пользователей')
        session.rollback()
        return ''
    finally:
        session.close()


def get_profile(user_from_msg):
    text = i18n.format_value("get_profile_text", {"username": user_from_msg.username or 'Не указан', "id": str(user_from_msg.id)})
    session = Session(db)
    try:
        user = session.get(User, user_from_msg.id)
        if not user:
            return text + "\n" + i18n.format_value("user_not_found")
        inet = i18n.format_value("toggle_inet_on") if user.search_from_inet else i18n.format_value("toggle_inet_off")
        text = f"{text}\n {inet}"
        return text
    except Exception:
        logger.exception('Ошибка при выводе списка пользователей')
        session.rollback()
        return text
    finally:
        session.close()


def toggle_inet(user_id: int):
    session = Session(db)
    try:
        user = session.get(User, user_id)
        if not user:
            return i18n.format_value("user_not_found")
        user.search_from_inet = not user.search_from_inet
        session.add(user)
        session.commit()
        return i18n.format_value("toggle_inet_on") if user.search_from_inet else i18n.format_value("toggle_inet_off")
    except Exception:
        logger.exception('Ошибка при изменении режима поиска из интернета')
        session.rollback()
    finally:
        session.close()


def add_superusers():
    session = Session(db)
    try:
        for user_id in superusers:
            user = session.get(User, user_id)
            if not user:
                user = User(id=user_id)
                session.add(user)
        session.commit()
    except Exception:
        logger.exception('Ошибка при выводе списка пользователей')
        session.rollback()
    finally:
        session.close()

def get_search_from_inet(user_id) -> bool:
    session = Session(db)
    try:
        user = session.get(User, user_id)
        if not user:
            return i18n.format_value("user_not_found")
        search_from_inet = user.search_from_inet
    except Exception:
        logger.exception('Ошибка при проверке режима поиска из интернета')
        session.rollback()
    finally:
        session.close()
