import json
import os
from typing import Any, Optional

import tensorflow as tf
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.graphs.neo4j_graph import Neo4jGraph
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import StructuredTool
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.runnables import (
    ConfigurableField,
    RunnableConfig,
    RunnableSerializable,
)
from langserve import add_routes
from libreco.algorithms import DeepFM, PinSage
from libreco.data import DataInfo
from sqlmodel import Session

from .database import engine
from .models import Input, KGRetrieverInput, LKPPRecSysInput, MovieRecSysInput, Output
from .router import auth
from .router.auth import per_req_config_modifier

from langchain.chains import GraphCypherQAChain
from langchain.chains.graph_qa.cypher_utils import CypherQueryCorrector, Schema


# Load konfigurasi dari file .env
load_dotenv()

# Load mapping untuk ID penyedia dari file JSON
with open("data/movie_id_mappings.json", "r") as json_file:
    movie_id_mappings = json.load(json_file)

with open("data/penyedia_id_mappings.json", "r") as json_file:
    penyedia_id_mappings = json.load(json_file)

session = Session(engine)


class CustomAgentExecutor(RunnableSerializable):
    MOVIE_MODEL_PATH: str = "recsys_models/movielens_model"
    LKPP_MODEL_PATH: str = "recsys_models/lkpp_model"
    MOVIE_MODEL_NAME: str = "movielens_deepfm_model"
    LKPP_MODEL_NAME: str = "pinsage_model_lkpp"
    MEMORY_KEY: str = "chat_history"
    NEO4J_URI: str = os.getenv("NEO4J_URI")
    NEO4J_USERNAME: str = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD")
    user_id: Optional[int]

    # recsys
    movie_data_info: Any
    movie_recsys_model: Any
    lkpp_data_info: Any
    lkpp_recsys_model: Any
    movie_recsys: Any
    lkpp_recsys: Any

    # knowledge graph
    neo4j_graph_store: Any
    kg_retriever: Any
    corrector_schema: Any
    cypher_validation: Any

    # agent
    agent: Any
    prompt: Any
    llm: Any
    tools: Any
    llm_with_tools: Any

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        tf.compat.v1.reset_default_graph()

        self.movie_data_info = DataInfo.load(
            self.MOVIE_MODEL_PATH, model_name=self.MOVIE_MODEL_NAME
        )
        self.lkpp_data_info = DataInfo.load(
            self.LKPP_MODEL_PATH, model_name=self.LKPP_MODEL_NAME
        )

        self.movie_recsys_model = DeepFM.load(
            path=self.MOVIE_MODEL_PATH,
            model_name=self.MOVIE_MODEL_NAME,
            data_info=self.movie_data_info,
            manual=True,
        )

        self.lkpp_recsys_model = PinSage.load(
            path=self.LKPP_MODEL_PATH,
            model_name=self.LKPP_MODEL_NAME,
            data_info=self.lkpp_data_info,
            manual=True,
        )

        self.movie_recsys = StructuredTool.from_function(
            func=self._recommend_top_k_movies,
            name="MovieRecSys",
            description="""
            Retrieve top k recommended movies for a user based on historical data, 
            do not use for any other purpose. Only use when user asks for a reccomendation of films.
            """,
            args_schema=MovieRecSysInput,
            return_direct=False,
        )

        self.lkpp_recsys = StructuredTool.from_function(
            func=self._recommend_top_k_companies,
            name="LKPPRecSys",
            description="""
            Retrieve top k recommended company for a user based on historical data, 
            do not use for any other purpose. Only use when user asks for a reccomendation of companies.
            """,
            args_schema=LKPPRecSysInput,
            return_direct=False,
        )

        self.neo4j_graph_store = Neo4jGraph(
            url=self.NEO4J_URI,
            username=self.NEO4J_USERNAME,
            password=self.NEO4J_PASSWORD,
        )

        self.kg_retriever = StructuredTool.from_function(
            func=self._retrieve_kg,
            name="KGRetriever",
            description="""
            Retrieve knowledge graph from existing database to help answer user questions about movies, or company
            Examples: Retrieve movies with similar genre and ratings. Retrieve company with hghest average ratings.
            Retrieve movies with most ratings.
            do not user for any other purpose.
            """,
            args_schema=KGRetrieverInput,
            return_direct=False,
        )

        self.corrector_schema = [
            Schema(el["start"], el["type"], el["end"])
            for el in self.neo4j_graph_store.structured_schema.get("relationships")
        ]

        self.cypher_validation = CypherQueryCorrector(self.corrector_schema)

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Anda adalah seorang {role}",
                ),
                (
                    "system",
                    "{instructions}",
                ),
                MessagesPlaceholder(variable_name=self.MEMORY_KEY),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
        self.tools = [self.movie_recsys, self.lkpp_recsys, self.kg_retriever]
        # self.tools = [self.recsys_lkpp, self.kg_retriever_lkpp, self.kg_retriever_movielens]

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
                "instructions": lambda x: x["instructions"],
            }
            | self.prompt
            | self.llm_with_tools
            | OpenAIFunctionsAgentOutputParser()
        )

    def _recommend_top_k_movies(self, k: int):
        """Retrieve top k recommended movies for a User"""
        prediction = self.movie_recsys_model.recommend_user(
            user=self.user_id,
            n_rec=k,
        )
        movie_ids = prediction[self.user_id]
        movies = [f"{str(mid)}:{movie_id_mappings[str(mid)]}" for mid in movie_ids]

        info = f"Rekomendasi film untuk user {self.user_id} berdasarkan data historis adalah {', '.join(movies)}"
        return info

    def _recommend_top_k_companies(self, k: int):
        """Retrieve top k recommended companies for a User"""
        prediction = self.lkpp_recsys_model.recommend_user(
            user=self.user_id,
            n_rec=k,
        )
        company_ids = prediction[self.user_id]
        companies = [
            f"{str(cid)}:{penyedia_id_mappings[str(cid)]}" for cid in company_ids
        ]

        info = f"Rekomendasi penyedia untuk user {self.user_id} berdasarkan data historis adalah {', '.join(companies)}"
        return info

    def _retrieve_kg(self, question: str):
        """Retrieve data from knowledge graph to answer user question"""
        chain = GraphCypherQAChain(cypher_query_corrector=self.cypher_validation).from_llm(
            self.llm, graph=self.neo4j_graph_store, verbose=True, return_direct=True, validate_cypher=True
        )

        result = chain.run(question)
        return result

    def invoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Output:
        agent_executor = AgentExecutor(
            agent=self.agent, tools=self.tools, verbose=True
        ).with_config({"run_name": "executor"})

        return agent_executor.invoke(input, config=config, **kwargs)


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

app.include_router(auth.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
