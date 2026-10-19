from datetime import datetime
from pydantic import BaseModel, Field


class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)


class UserLogin(BaseModel):
    username: str
    password: str


class UserOut(BaseModel):
    id: int
    username: str
    created_at: datetime

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class PredictRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)


class PredictResponse(BaseModel):
    message: str
    is_spam: bool
    label: str
    confidence: float
    model_votes: dict
    heuristic_flags: dict


class HistoryItem(BaseModel):
    id: int
    message: str
    is_spam: bool
    confidence: float
    checked_at: datetime

    class Config:
        from_attributes = True
