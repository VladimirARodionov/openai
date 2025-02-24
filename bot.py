import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from alembic.config import Config
from alembic import command

from locale_config import i18n

from aiogram.types import BotCommand, BotCommandScopeDefault

from create_bot import env_config

alembic_cfg = Config("alembic.ini")
alembic_cfg.attributes['configure_logger'] = False
command.upgrade(alembic_cfg, "head")

logger = logging.getLogger(__name__)


bot = Bot(token=env_config.get('TOKEN'), default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

# Функция, которая настроит командное меню (дефолтное для всех пользователей)
async def set_commands():
    commands = [BotCommand(command='start', description=i18n.format_value("start_menu")),
                BotCommand(command='profile', description=i18n.format_value("my_profile_text")),
                BotCommand(command='stop', description=i18n.format_value("stop_menu"))]
    await bot.set_my_commands(commands, BotCommandScopeDefault())


# Функция, которая выполнится когда бот запустится
async def start_bot():
    await set_commands()
    logger.info('Бот стартован')


# Функция, которая выполнится когда бот завершит свою работу
async def stop_bot():
    logger.info('Бот остановлен')


async def main():
    from handlers.router import router

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
    except Exception:
        logger.exception('Неизвестная ошибка')
