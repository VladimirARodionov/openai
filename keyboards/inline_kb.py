from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from locale_config import i18n

def get_inline_kb() -> InlineKeyboardMarkup:
    simple_response_button = InlineKeyboardButton(
        text=i18n.format_value("simple_response_menu"),
        callback_data="simple_response")
    detailed_report_button = InlineKeyboardButton(
        text=i18n.format_value("detailed_report_menu"),
        callback_data="detailed_report")
    return InlineKeyboardMarkup(inline_keyboard=[[simple_response_button, detailed_report_button]])
