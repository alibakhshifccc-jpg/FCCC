import datetime import datetime
from pydantic import BaseModel, ConfigDict


class RequestLogInDBBase(BaseModel):
    id: int
    created: datetime
    modified: datetime
    model_config = ConfigDict(from_attributes=True)




class RequestLogCreate(BaseModel):
    pass 