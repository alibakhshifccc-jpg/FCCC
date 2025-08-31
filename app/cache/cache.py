from ast import pattern
import asyncio
from datetime import timedelta
from functools import partial, update_wrapper, wraps
from http import HTTPMethod
from typing import Union
from fastapi import Response





def cache(
    *, namespace: str | None = None,
    expire: int | timedelta = None,
):
    def outer_wrapper(func):
        @wraps(func)
        async def inner_warper(*args, **kwargs):
            func_kwarg = kwargs.copy()
            request = func_kwarg.pop("request", None)
            response = func_kwarg.pop("response", None)
            creat_response_directory = not response
            
            if  creat_response_directory:
                response = Response()
            redic_cache = Cache()

            if redic_cache.not_connected  or redic_cache.request_is_not_cacheable(
                request
            )
            return 
        


async def get_api_response_async(func,*args, **kwargs):
    return (
        await func(*args, **kwargs)
        if asyncio.iscoroutinefunction(func)
        else func(*args, **kwargs)
    )

def calculate_ttl(expire: Union[int, timedelta])-> int:
    if isinstance(expire, timedelta):
        expire = int(expire.total_seconds)
    return min(expire) 