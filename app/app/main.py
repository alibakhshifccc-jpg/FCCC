from contextlib import asynccontextmanager
import logging, sys , os
from fastapi import FastAPI, requests, Response 
from fastapi.middleware import Middleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session1
from starlette.middleware.cors import CORSMiddleware


import os 