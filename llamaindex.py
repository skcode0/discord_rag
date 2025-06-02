from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo
from llama_index.vector_stores.postgres import PGVectorStore
import os
from dotenv import load_dotenv
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from sqlalchemy import make_url
from llama_index.llms.ollama import Ollama

vector_store_info = VectorStoreInfo(
    content_info="Transcribed texts",
    metadata_info=[
        MetadataInfo(
            name="trans_id",
            description="Transcription id of text",
            type="integer"
        ),
        MetadataInfo(
            name="timestamp",
            description="time-zone aware timestamp of when the text was created",
            type="datetime"
        ),
        MetadataInfo(
            name="speaker",
            description="Speaker of the text",
            type="string"
        ),
        MetadataInfo(
            name="text",
            description="text content",
            type="string"
        )
    ]
)

# --------------------------
# env variables
# --------------------------
load_dotenv(override=True)
pg_username = os.environ.get('POSTGRESS_USER')
pg_password = os.environ.get('POSTGRES_PASSWORD')
short_db_name = os.environ.get('SHORT_TERM_DB')
short_port = os.environ.get('SHORT_TERM_HOST_PORT')
long_db_name = os.environ.get('LONG_TERM_DB')
long_port = os.environ.get('LONG_TERM_HOST_PORT')

emb_dim = os.environ.get('EMBEDDING_DIM')
emb_model = os.environ.get('EMBEDDING_MODEL')
llm_model = os.environ.get('LLM_MODEL')

# settings
Settings.embed_model = HuggingFaceEmbedding(
    model_name=emb_model
)
Settings.llm = Ollama(model=llm_model, request_timeout=120.0)

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'
task = 'Given a user query, retrieve relevant texts that answer the query: '

# q = get_detailed_instruct(task, "Who am I?")
# vector = HuggingFaceEmbedding(
#     model_name=emb_model
# ).get_text_embedding(q)
# print("EMBEDDING shape/type:", type(vector), "Length: ", len(vector) ,"first few values:", vector[:5])


class CustomPGVectorStore(PGVectorStore):
    def _get_table_name(self):
        # Use the exact table name provided without adding 'data_'
        return self.table_name

# --------------------------
# short-term
# --------------------------
short_uri = make_url(f'postgresql+asyncpg://{pg_username}:{pg_password}@{"localhost"}:{short_port}/{short_db_name}')

short_store = CustomPGVectorStore.from_params(
    database=short_db_name,
    host=short_uri.host,
    password=short_uri.password,
    port=short_uri.port,
    user=short_uri.username,
    table_name="transcriptionsvectors",
    embed_dim=int(emb_dim),
    schema_name="public",
    perform_setup=False, # do not create tables/indexes
)

short_index = VectorStoreIndex.from_vector_store(vector_store=short_store)

# https://docs.llamaindex.ai/en/stable/api_reference/retrievers/vector/#llama_index.core.retrievers.VectorIndexAutoRetriever
short_retriever = VectorIndexAutoRetriever(
    short_index, vector_store_info=vector_store_info
)

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'
task = 'Given a user query, retrieve relevant texts that answer the query: '

results = short_retriever.retrieve(get_detailed_instruct(task, 'Who am I?'))
for node_with_score in results:
    print(node_with_score.node.metadata, node_with_score.node.text[:100])

# --------------------------
# long-term
# --------------------------
