from aiogram.types import KeyboardButton, ReplyKeyboardMarkup, ReplyKeyboardRemove

from locale_config import i18n
from services.common import is_superuser


def main_kb(user_telegram_id: int):
    kb_list = [[]]
    if is_superuser(user_telegram_id):
        kb_list.append([KeyboardButton(text=i18n.format_value("add_user_menu")),
                        KeyboardButton(text=i18n.format_value("show_users_menu")),
                        KeyboardButton(text=i18n.format_value("delete_user_menu"))])
    if kb_list:
        return ReplyKeyboardMarkup(
            keyboard=kb_list,
            resize_keyboard=True,
            one_time_keyboard=True,
            input_field_placeholder=i18n.format_value("use_menu"))
    else:
        return ReplyKeyboardRemove()

def back_kb():
    kb_list = [[KeyboardButton(text=i18n.format_value("back_menu"))]]
    return ReplyKeyboardMarkup(
        keyboard=kb_list,
        resize_keyboard=True,
        one_time_keyboard=True,
        input_field_placeholder=i18n.format_value("use_menu")
    )
