import asyncio
import logging 
from app.db.init__db import init__db
from app.db.session import async_session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



async def creat_init_db() -> None:
    async with async_session as sesion:
        await init__db(sesion)


async def main() -> None:
    logger.info("Creating initial data")
    await creat_init_db()
    logger.info("Initial data created")


if __name__ == "__main__":
    asyncio.run(main())