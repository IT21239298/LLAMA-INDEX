import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader

load_dotenv()

# 1. load data
documents =SimpleDirectoryReader("./data/").load_data()

# 2. Create Index
# pip3 install chromadb llama-index-vector-stores-chroma

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

db = chromadb.PersistentClient(path="./chromadb")

chroma_collection = db.get_or_create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)