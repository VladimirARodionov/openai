import logging

from aiogram import Router, F
from aiogram.filters import CommandStart, Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
from aiogram.types import Message
from openai import OpenAI

from create_bot import env_config
from filters.filter import IsAllowed, IsSuperUser
from keyboards.kbs import back_kb, main_kb
from locale_config import i18n
from services.common import add_user, delete_user, list_users

router = Router()
logger = logging.getLogger(__name__)

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
    try:
        await message.answer(text=i18n.format_value("get_profile_text", {"username": message.from_user.username or 'Не указан', "id": str(message.from_user.id)}), reply_markup=main_kb(message.from_user.id))
    except Exception as e:
        logger.exception(str(e))


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


@router.message(F.text.contains(i18n.format_value("back_menu_text")), IsSuperUser())
async def cmd_back(message: Message, state: FSMContext):
    await state.clear()
    await message.answer(i18n.format_value("back_text"), reply_markup=main_kb(message.from_user.id))



client = OpenAI(
    api_key=env_config.get('OPEN_AI_TOKEN'),  # This is the default and can be omitted
)
model=env_config.get('MODEL')

def ask_gpt(prompt):
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model,
        )

        return response.choices[0].message.content
    except Exception as e:
        logger.exception(str(e))
        return "OpenAI вернул ошибку: " + str(e)

@router.message(F.text, IsAllowed())
async def chat_with_gpt(message):
    response = ask_gpt(message.text)
    await message.answer(response, reply_markup=main_kb(message.from_user.id))
