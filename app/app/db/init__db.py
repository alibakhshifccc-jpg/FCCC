import logging 
import heapq
import numpy as np
import math 
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio
from app.core.config import settings
from app.app import schemas, crud



logger = logging.getLogger(__name__)


async def create_super_admin(db: AsyncSession) -> None:
    user = await crud.user.get_by_email(db=db, email=settings.FIRST_SUPERUSER)
    if not user:
        user = schemas.UserCreate(
            email=settings.FIRST_SUPERUSER,
            password=settings.FIRST_SUPERUSER_PASSWORD,
            is_superuser=True,
        )
        await crud.user.create(db=db, obj_in=user)


def dijkstra(graph: dict[int, dict[int, float]], start_vertex: int):
    queue: list[tuple[int, int]] = [(0, start_vertex)]
    distances: dict[int, float] = {vertex: float("inf") for vertex in graph}
    distances[start_vertex] = 0
    previous_vertices: dict[int, int | None] = {vertex: None for vertex in graph}

    while queue:
        current_distance, current_vertex = heapq.heappop(queue)

        if current_distance > distances[current_vertex]:
            continue

        for neighbor, weight in graph[current_vertex].items():
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_vertices[neighbor] = current_vertex
                heapq.heappush(queue, (distance, neighbor))
    return distances, previous_vertices