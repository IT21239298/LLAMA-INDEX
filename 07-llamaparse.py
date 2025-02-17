# lamaparse ca used more complex file 
import os
from dotenv import load_dotenv
import nest_asyncio

nest_asyncio.apply()

from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

load_dotenv()
LLAMA_CLOUD = os.getenv("LLAMA_CLOUD")

parser = LlamaParse(
    api_key= LLAMA_CLOUD,
    result_type="markdown",
    verbose=True
)

file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader(
    "./pdf", file_extractor=file_extractor
).load_data()

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()

response = query_engine.query("who helped create this project")

print(response)