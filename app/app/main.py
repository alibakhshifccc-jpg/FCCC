import os 
import logging, sys , os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response 
from fastapi.middleware import Middleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from starlette.middleware.cors import CORSMiddleware
from cache import Cache

current_dir = os.path.dirname(__file__)
static_dir = os.path.join(current_dir, "static")

def init_logger()->None:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(levelname)s:%(asctime)s %(name)s:%(funcName)s:%(lineno)s %(message)s"
        )
    )
    logger.addHandler(handler)

init_logger()


def make_middleware()-> list[Middleware]:
    pass 


@asynccontextmanager
async def lifespan(app: FastAPI):
    redis_cache = Cache()
    url = str 
    await redis_cache.init(
        host_url=url,
        prefix="api-cache",
        response_header="X-API-Cache",
        ignore_arg_types=[Request, Response, Session, AsyncSession]
    )
    yield


app = FastAPI(

)