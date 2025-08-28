from sqlalchemy import Boolean, Integer, String
from sqlalchemy.orm import Mapped, mapped_column




class User:
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    
    full_name: Mapped[str | None] = mapped_column(String, index=True)
    
    email: Mapped[str] = mapped_column(String, )



    is_active: Mapped[bool | None] = mapped_column(Boolean, default=True)
    is_superuser: Mapped[bool | None] = mapped_column(Boolean, default=False)  