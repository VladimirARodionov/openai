from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from locale_config import i18n

def get_inline_kb():
    """Создает инлайн клавиатуру для выбора формата ответа"""
    buttons = [
        [
            InlineKeyboardButton(
                text="Обычный ответ",
                callback_data="simple_response"
            ),
            InlineKeyboardButton(
                text=i18n.format_value("detailed_report_menu"),
                callback_data="detailed_report"
            )
        ]
    ]
    keyboard = InlineKeyboardMarkup(inline_keyboard=buttons)
    return keyboard

def get_users_pagination_kb(page: int, total_pages: int):
    """
    Создает инлайн клавиатуру для пагинации списка пользователей
    
    Args:
        page: Текущая страница (начиная с 0)
        total_pages: Общее количество страниц
        
    Returns:
        InlineKeyboardMarkup: Клавиатура с кнопками навигации
    """
    keyboard = []
    nav_row = []
    
    # Кнопка "Назад"
    if page > 0:
        nav_row.append(InlineKeyboardButton(
            text=i18n.format_value("users_nav_prev"),
            callback_data=f"users_page:{page-1}"
        ))
    
    # Кнопка "Вперед"
    if page < total_pages - 1:
        nav_row.append(InlineKeyboardButton(
            text=i18n.format_value("users_nav_next"),
            callback_data=f"users_page:{page+1}"
        ))
    
    if nav_row:
        keyboard.append(nav_row)
    
    # Добавляем кнопку возврата в главное меню
    keyboard.append([
        InlineKeyboardButton(
            text=i18n.format_value("users_nav_back_to_menu"),
            callback_data="users_back_to_menu"
        )
    ])
    
    return InlineKeyboardMarkup(inline_keyboard=keyboard)
