from typing import List, Union, Any
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, FunctionMessage, AIMessage


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str | None = None


class User(BaseModel):
    username: str
    user_id : int
    email: Union[str, None] = None
    full_name: Union[str, None] = None
    disabled: Union[bool, None] = None


class UserInDB(User):
    hashed_password: str


class RecSysInput(BaseModel):
    # uid: int = Field(description="User id")
    k: int = Field(description="Number of movies to be recommended")


class Input(BaseModel):
    input: str
    role: str = Field(default="Pakar Film")
    chat_history: List[Union[HumanMessage, AIMessage, FunctionMessage]]


class Output(BaseModel):
    output: Any
