from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI

load_dotenv()

OpenAI(model="gpt-4o-mini")

documents = SimpleDirectoryReader("pdf/").load_data()

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()

response = query_engine.query("What are the design goals and please give details about them!")

print(response)