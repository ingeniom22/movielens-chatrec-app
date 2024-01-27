import json
from typing import Annotated, Any, Dict, Optional

import tensorflow as tf
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# from .router.auth import per_req_config_modifier
from langchain.schema.runnable import RunnablePassthrough
from langchain.tools import StructuredTool
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.runnables import (
    ConfigurableField,
    Runnable,
    RunnableConfig,
    RunnableSerializable,
)
from langserve import add_routes
from libreco.algorithms import DeepFM
from libreco.data import DataInfo
from sqlmodel import Session

from .database import engine
from .models import Input, Output, RecSysInput, User
from .router import auth
from .router.auth import get_current_active_user, get_current_active_user_from_request

# Load konfigurasi dari file .env
load_dotenv()

MODEL_PATH = "recsys_models/movielens_model"
MODEL_NAME = "movielens_deepfm_model"

# Load mapping untuk ID penyedia dari file JSON
with open("data/movie_id_mappings.json", "r") as json_file:
    movie_id_mappings = json.load(json_file)

# Reset graph TensorFlow yang ada
tf.compat.v1.reset_default_graph()

# Load informasi data dan model sistem rekomendasi
data_info = DataInfo.load(MODEL_PATH, model_name=MODEL_NAME)
model = DeepFM.load(
    path=MODEL_PATH, model_name=MODEL_NAME, data_info=data_info, manual=True
)


session = Session(engine)


# Konversi fungsi rekomendasi menjadi tool terstruktur
# recsys = StructuredTool.from_function(
#     func=recommend_top_k,
#     name="RecSys",
#     description="Retrieve top k recommended company for a User",
#     args_schema=RecSysInput,
#     return_direct=False,
# )

# List tools yang digunakan
# tools = [
#     recsys,
#     # TODO: Add more tools
# ]

# Key untuk menyimpan riwayat obrolan dalam memory
MEMORY_KEY = "chat_history"

# Template prompt untuk percakapan
# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "Anda adalah seorang {role} yang sedang berbicara dengan user uid={user_id}.",
#         ),
#         MessagesPlaceholder(variable_name=MEMORY_KEY),
#         ("user", "{input}"),
#         MessagesPlaceholder(variable_name="agent_scratchpad"),
#     ]
# )

# Inisialisasi LLM ChatGPT
# llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

# Bind tools ke model bahasa
# llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])

# Konfigurasi agen percakapan
# agent = (
#     {
#         "input": lambda x: x["input"],
#         "agent_scratchpad": lambda x: format_to_openai_function_messages(
#             x["intermediate_steps"]
#         ),
#         "chat_history": lambda x: x["chat_history"],
#         "user_id": lambda x: x["user_id"],
#         "role": lambda x: x["role"],
#     }
#     | prompt
#     | llm_with_tools
#     | OpenAIFunctionsAgentOutputParser()
# )

# Inisialisasi executor untuk agen
# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, user_id=None)


class CustomAgentExecutor(RunnableSerializable):
    user_id: Optional[str]
    agent: Any
    recsys: Any
    prompt: Any
    llm: Any
    tools: Any
    llm_with_tools: Any


    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.recsys = StructuredTool.from_function(
            func=self._recommend_top_k,
            name="RecSys",
            description="Retrieve top k recommended company for a User",
            args_schema=RecSysInput,
            return_direct=False,
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Anda adalah seorang {role}",
                ),
                MessagesPlaceholder(variable_name=MEMORY_KEY),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        self.tools = [self.recsys]

        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
        self.llm_with_tools = self.llm.bind(
            functions=[format_tool_to_openai_function(t) for t in self.tools]
        )

        self.agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_function_messages(
                    x["intermediate_steps"]
                ),
                "chat_history": lambda x: x["chat_history"],
                "role": lambda x: x["role"],
            }
            | self.prompt
            | self.llm_with_tools
            | OpenAIFunctionsAgentOutputParser()
        )

    # Fungsi untuk merekomendasikan perusahaan
    def _recommend_top_k(self, k: int):
        """Retrieve top k recommended movies for a User"""
        prediction = model.recommend_user(
            user=self.user_id,
            n_rec=k,
        )  # TODO: ambil user feature berdasarkan uid
        movie_ids = prediction[self.user_id]
        movies = [f"{str(mid)}:{movie_id_mappings[str(mid)]}" for mid in movie_ids]

        info = f"Rekomendasi film untuk user adalah {', '.join(movies)}"
        return info

    def invoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Output:
        agent_executor = AgentExecutor(
            agent=self.agent, tools=self.tools, verbose=True
        ).with_config({"run_name": "executor"})

        # input["user_id"] = self.user_id
        return agent_executor.invoke(input, config=config, **kwargs)


async def per_req_config_modifier(config: Dict, request: Request) -> Dict:
    """Modify the config for each request."""
    user = await get_current_active_user_from_request(request)
    config["configurable"] = {}
    # Attention: Make sure that the user ID is over-ridden for each request.
    # We should not be accepting a user ID from the user in this case!
    config["configurable"]["user_id"] = user.user_id
    return config


app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# add_routes(
#     app,
#     agent_executor.with_types(input_type=Input, output_type=Output),
#     dependencies=[Depends(get_current_active_user)]
#     # per_req_config_modifier=per_req_config_modifier,
# )


runnable = CustomAgentExecutor(user_id=None).configurable_fields(
    user_id=ConfigurableField(
        id="user_id",
        name="User ID",
        description="The user ID to use for the retriever.",
    )
)
add_routes(
    app,
    runnable.with_types(input_type=Input, output_type=Output),
    per_req_config_modifier=per_req_config_modifier,
    enabled_endpoints=["invoke"],
)

# add_routes(
#     app,
#     per_user_chatrec_agent,
#     per_req_config_modifier=per_req_config_modifier,
#     enabled_endpoints=["invoke"],
# )

app.include_router(auth.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
