import asyncio

from aiogram import Bot, Dispatcher, types
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from logging import config as logging_config

from aiogram.types import BotCommand, BotCommandScopeDefault

from create_bot import get_logger, env_config
from handlers.router import router


logger = get_logger(__name__)


bot = Bot(token=env_config.get('TOKEN'), default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

# Функция, которая настроит командное меню (дефолтное для всех пользователей)
async def set_commands():
    commands = [BotCommand(command='start', description='Старт'),
                BotCommand(command='stop', description='Стоп')]
    await bot.set_my_commands(commands, BotCommandScopeDefault())


# Функция, которая выполнится когда бот запустится
async def start_bot():
    await set_commands()
    logger.info('Бот стартован')


# Функция, которая выполнится когда бот завершит свою работу
async def stop_bot():
    logger.info('Бот остановлен')


async def main():
    dp.include_router(router)

    # регистрация функций
    dp.startup.register(start_bot)
    dp.shutdown.register(stop_bot)

    logger.info('Бот запущен.')
    try:
        await bot.delete_webhook(drop_pending_updates=True)
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    except KeyboardInterrupt:
        pass
    finally:
        await bot.session.close()
        logger.info('Бот остановлен.')


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info('Клавиатурное прерывание')
    except asyncio.CancelledError:
        logger.info('Прерывание')
