from aiogram.filters import BaseFilter
from aiogram.types import Message

from services.common import is_superuser, is_allowed


class IsSuperUser(BaseFilter):
    async def __call__(self, message: Message) -> bool:
        return is_superuser(message.from_user.id)

class IsAllowed(BaseFilter):
    async def __call__(self, message: Message) -> bool:
        return is_allowed(message.from_user)
