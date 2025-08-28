import logging 
import heapq
import numpy as np
import math 
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio
from app.core.config import settings
from app.app import schemas, crud



logger = logging.getLogger(__name__)


async def create_super_admin(db: AsyncSession)-> None:
    user = await crud.user.get_by_email(db=db, email=settings.FIRST_SUPERUSER)
    if not user:
        user = schemas.U