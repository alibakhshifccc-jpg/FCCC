from sqlalchemy import Boolean, ForeignKey, Integer, String, Float
from sqlalchemy.orm import mapped_column, Mapped, relationship




class Edge:
    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        index=True
    )
    

    