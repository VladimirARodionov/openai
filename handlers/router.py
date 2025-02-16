import logging

from aiogram import Router, F
from aiogram.filters import CommandStart, Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
from aiogram.types import Message

from filters.filter import IsAllowed, IsSuperUser
from keyboards.kbs import back_kb, main_kb
from locale_config import i18n
from services.common import add_user, delete_user, list_users, get_profile, toggle_inet, add_superusers
from services.embedding import EmbeddingsSearch

router = Router()
logger = logging.getLogger(__name__)

add_superusers()
searcher = EmbeddingsSearch()

class FSMAddUser(StatesGroup):
    wait_user = State()


class FSMDeleteUser(StatesGroup):
    wait_user = State()


# хендлер команды старт
@router.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(text=i18n.format_value("start_menu_text"), reply_markup=main_kb(message.from_user.id))


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
    res = list_users()
    await message.answer(res, reply_markup=main_kb(message.from_user.id))


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
    response = searcher.load_documents_from_directory('load')
    await message.answer(response, reply_markup=main_kb(message.from_user.id))


@router.message(Command('clear_database'), IsSuperUser())
async def clear_database(message: Message):
    response = searcher.clear_database()
    await message.answer(response, reply_markup=main_kb(message.from_user.id))


@router.message(F.text, IsAllowed())
async def chat_with_gpt(message):
    #response = ask_gpt(message.text)
    response = searcher.ask(message.text, True)
    await message.answer(response, reply_markup=main_kb(message.from_user.id))
