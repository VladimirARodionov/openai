import logging

from aiogram import Router, F
from aiogram.filters import CommandStart, Command
from aiogram.types import Message
from openai import OpenAI

from create_bot import env_config

router = Router()
logger = logging.getLogger(__name__)

# хендлер команды старт
@router.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(text="Привет! Я бот, основанный на Агни Йоге. Задайте мне вопрос.")


@router.message(Command('stop'))
async def cmd_stop(message: Message):
    pass

client = OpenAI(
    api_key=env_config.get('OPEN_AI_TOKEN'),  # This is the default and can be omitted
)

def ask_gpt(prompt):
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-4",
        )

        return response["choices"][0]["message"]["content"]
    except Exception as e:
        logger.exception(str(e))
        return "OpenAI вернул ошибку: " + str(e)

@router.message(F.text)
async def chat_with_gpt(message):
    response = ask_gpt(message.text)
    await message.answer(response)

