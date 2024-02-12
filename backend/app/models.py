from typing import List, Union, Any
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, FunctionMessage, AIMessage


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str | None = None


class User(BaseModel):
    """
    "user_id": 1,
    "age": 24,
    "gender": "M",
    "occupation": "technician",
    "zip_code": "85711",
    "username": "Margy",
    "full_name": "Catherine Oliver",
    "email": "njimenez@example.com",
    "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",
    "disabled": false
    """
    username: str
    user_id : int
    age: int
    gender: str
    occupation: str
    zip_code: str
    email: Union[str, None] = None
    full_name: Union[str, None] = None
    disabled: Union[bool, None] = None


class UserInDB(User):
    hashed_password: str


class RecSysInput(BaseModel):
    # uid: int = Field(description="User id")
    k: int = Field(description="Number of movies to be recommended")

class KGRetrieverInput(BaseModel):
    question: str = Field(description="User questions")


class Input(BaseModel):
    input: str
    role: str = Field(default="Pakar Film")
    instructions: str = Field(default="")
    chat_history: List[Union[HumanMessage, AIMessage]]


class Output(BaseModel):
    output: Any
