from pydantic import BaseModel, ConfigDict, EmailStr
from typing import Optional




class UserBase(BaseModel):
    email: EmailStr | None = None
    is_active: bool | None = True
    is_superuser: bool = False
    full_name: str | None = None


class UserCreate(UserBase):
    password: str | None = None