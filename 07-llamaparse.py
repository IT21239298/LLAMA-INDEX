# lamaparse ca used more complex file 
import os
from dotenv import load_dotenv
import nest_asyncio

nest_asyncio.apply()

from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

load_dotenv()
LLAMA_CLOUD = os.getenv("LLAMA_CLOUD")

# "text": Outputs the parsed content as plain text.
# "markdown": Outputs the parsed content in Markdown format, preserving structural elements like headings, lists, and tables.
# "json": Outputs the parsed content as a JSON object, providing a structured representation of the document, including elements like text, headings, tables (in CSV and JSON formats), and metadata about each node. This format is particularly useful for programmatic access and manipulation of the document's content.
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

response = query_engine.query("i need information about visa requiremments to travel to the US")

print(response)

