import logging
import asyncio
import html

from aiogram import Router, F
from aiogram.filters import CommandStart, Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State, default_state
from aiogram.types import Message, CallbackQuery

from create_bot import START_MENU_TEXT
from filters.filter import IsAllowed, IsSuperUser
from keyboards.inline_kb import get_inline_kb, get_users_pagination_kb
from keyboards.kbs import back_kb, main_kb
from locale_config import i18n
from services.common import add_user, delete_user, get_profile, toggle_inet, add_superusers, get_users_paginated
from services.embedding import EmbeddingsSearch

router = Router()
logger = logging.getLogger(__name__)

add_superusers()
searcher = None

# Константа для количества пользователей на странице
USERS_PER_PAGE = 20

def _get_searcher():
    global searcher
    if searcher is None:
        searcher = EmbeddingsSearch()
    return searcher

class FSMAddUser(StatesGroup):
    wait_user = State()


class FSMDeleteUser(StatesGroup):
    wait_user = State()


class FSMSelectResponseFormat(StatesGroup):
    select_response_format = State()


# хендлер команды старт
@router.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(START_MENU_TEXT, reply_markup=main_kb(message.from_user.id))


@router.message(Command('stop'))
async def cmd_stop(message: Message):
    pass

@router.message(Command('profile'))
async def cmd_profile(message: Message):
    await message.answer(text=get_profile(message.from_user), reply_markup=main_kb(message.from_user.id))


@router.message(F.text.contains(i18n.format_value("back_menu_text")))
async def cmd_back(message: Message, state: FSMContext):
    await state.clear()
    await message.answer(i18n.format_value("back_text"), reply_markup=main_kb(message.from_user.id))


@router.message(F.text.endswith(i18n.format_value("toggle_inet_text")), IsAllowed())
async def toggle_inet_user(message: Message):
    res = toggle_inet(message.from_user.id)
    await message.answer(res, reply_markup=main_kb(message.from_user.id))


@router.message(F.text.endswith(i18n.format_value("show_users_text")), IsSuperUser())
async def cmd_show_users(message: Message):
    """Показать список пользователей с пагинацией"""
    # Получаем список пользователей и отображаем первую страницу
    await show_users_page(message, page=0)

async def show_users_page(message: Message, page=0, edit=False):
    """Показать определенную страницу списка пользователей"""
    # Получаем общее количество пользователей и список для текущей страницы
    total_users, users = get_users_paginated(page, USERS_PER_PAGE)
    
    if not users:
        text = i18n.format_value("show_users_text_empty")
        if edit:
            await message.edit_text(text, reply_markup=main_kb(message.from_user.id))
        else:
            await message.answer(text, reply_markup=main_kb(message.from_user.id))
        return
    
    # Формируем текст сообщения
    total_pages = (total_users - 1) // USERS_PER_PAGE + 1
    text = i18n.format_value("show_users_page_header", {
        "current": page + 1,
        "total": total_pages,
        "count": total_users
    }) + "\n\n"
    
    text += "\n".join([f"<code><b>{user.id}</b></code> [{('@'+user.name) if user.name else ''}]" for user in users])
    
    # Создаем клавиатуру для навигации
    markup = get_users_pagination_kb(page, total_pages)
    
    # Отправляем или редактируем сообщение
    if edit and hasattr(message, 'edit_text'):
        await message.edit_text(text, reply_markup=markup)
    else:
        await message.answer(text, reply_markup=markup)

@router.callback_query(lambda c: c.data.startswith("users_page:"))
async def process_users_page(callback_query: CallbackQuery):
    """Обработка навигации по страницам пользователей"""
    await callback_query.answer()
    
    # Извлекаем номер страницы из callback_data
    page = int(callback_query.data.split(":")[1])
    
    # Показываем запрошенную страницу
    await show_users_page(callback_query.message, page, edit=True)

@router.callback_query(lambda c: c.data == "users_back_to_menu")
async def process_users_back_to_menu(callback_query: CallbackQuery):
    """Возврат в главное меню из списка пользователей"""
    await callback_query.answer()
    await callback_query.message.edit_text(
        i18n.format_value("back_text"),
        reply_markup=None
    )

@router.message(F.text.endswith(i18n.format_value("add_user_text")), IsSuperUser())
async def cmd_add_user(message: Message, state: FSMContext):
    await state.set_state(FSMAddUser.wait_user)
    await message.answer(i18n.format_value("add_user_prompt"), reply_markup=back_kb())


@router.message(F.text.endswith(i18n.format_value("delete_user_text")), IsSuperUser())
async def cmd_delete_user(message: Message, state: FSMContext):
    await state.set_state(FSMDeleteUser.wait_user)
    await message.answer(i18n.format_value("delete_user_prompt"), reply_markup=back_kb())


@router.message(IsSuperUser(), StateFilter(FSMAddUser.wait_user))
async def cmd_add_user_set(message: Message, state: FSMContext):
    await process_add_user(message, state)


@router.message(IsSuperUser(), StateFilter(FSMDeleteUser.wait_user))
async def cmd_delete_user_set(message: Message, state: FSMContext):
    await process_delete_user(message, state)


async def process_add_user(message: Message, state: FSMContext):
    if add_user(message.text):
        await state.clear()
        await message.answer(i18n.format_value("success_add_user_text"), reply_markup=main_kb(message.from_user.id))
    else:
        await state.clear()
        await message.answer(i18n.format_value("not_success_add_user_text"), reply_markup=main_kb(message.from_user.id))


async def process_delete_user(message: Message, state: FSMContext):
    if delete_user(message.text):
        await state.clear()
        await message.answer(i18n.format_value("success_delete_user_text"), reply_markup=main_kb(message.from_user.id))
    else:
        await state.clear()
        await message.answer(i18n.format_value("not_success_delete_user_text"), reply_markup=main_kb(message.from_user.id))


@router.message(Command('load_from_dir'), IsSuperUser())
async def load_from_dir(message: Message):
    searcher = _get_searcher()
    response = searcher.load_documents_from_directory('load', message.from_user.id)
    await message.answer(response, reply_markup=main_kb(message.from_user.id))


@router.message(Command('clear_database'), IsSuperUser())
async def clear_database(message: Message):
    searcher = _get_searcher()
    response = searcher.clear_database()
    await message.answer(response, reply_markup=main_kb(message.from_user.id))


@router.message(F.text, IsAllowed(), StateFilter(default_state))
async def chat_with_gpt(message, state: FSMContext):
    await state.update_data(name=message.text)
    await state.set_state(FSMSelectResponseFormat.select_response_format)
    await message.reply(i18n.format_value("response_format_text"), reply_markup=get_inline_kb())

@router.message(F.text, IsAllowed(), StateFilter(FSMSelectResponseFormat.select_response_format))
async def chat_with_gpt_again(message, state: FSMContext):
    await state.clear()
    await state.update_data(name=message.text)
    await state.set_state(FSMSelectResponseFormat.select_response_format)
    await message.reply(i18n.format_value("response_format_text"), reply_markup=get_inline_kb())

async def print_parts(response:str, callback_query: CallbackQuery):
    # Разбиваем отчет на части по маркерам
    parts = response.split("\n\n")

    # Отправляем каждую часть отдельным сообщением
    for part in parts:
        if part.strip():  # Проверяем, что часть не пустая
            # Ограничиваем длину каждого сообщения
            if len(part) > 4000:
                # Если часть слишком длинная, разбиваем её на меньшие части
                for i in range(0, len(part), 4000):
                    sub_part = part[i:i + 4000]
                    await callback_query.message.answer(sub_part)
                    await asyncio.sleep(0.5)  # Небольшая задержка между сообщениями
            else:
                await callback_query.message.answer(part)
                await asyncio.sleep(0.5)  # Небольшая задержка между сообщениями


@router.callback_query(lambda c: c.data in ["simple_response", "detailed_report"])
async def process_callback(callback_query: CallbackQuery, state: FSMContext):
    try:
        # Сначала отвечаем на callback, чтобы убрать "часики"
        await callback_query.answer()
        
        data = await state.get_data()
        await state.clear()
        
        if not "name" in data:
            await callback_query.message.answer(i18n.format_value("no_name_text"))
            return
            
        msg = data['name']
        user_choice = callback_query.data
        
        # Отправляем сообщение о начале обработки
        processing_msg = await callback_query.message.answer(i18n.format_value("processing_message"))
        
        try:
            searcher = _get_searcher()
            if user_choice == "simple_response":
                response = searcher.ask(msg, callback_query.from_user.id, True)
                await print_parts(response, callback_query)
            elif user_choice == "detailed_report":
                response = searcher.report(msg, callback_query.from_user.id, True)
                await print_parts(response, callback_query)
            
            # Удаляем сообщение о подготовке ответа
            await processing_msg.delete()
            
        except Exception as e:
            logger.exception(f"Error processing request: {str(e)}")
            # Экранируем специальные символы в тексте ошибки
            error_text = html.escape(str(e))
            await processing_msg.edit_text(
                i18n.format_value("error_processing_request", {"error": error_text})
            )
            
    except Exception as e:
        logger.exception(f"Error in callback handler: {str(e)}")
        # Экранируем специальные символы в тексте ошибки
        error_text = html.escape(str(e))
        await callback_query.message.answer(
            i18n.format_value("error_in_callback", {"error": error_text})
        )