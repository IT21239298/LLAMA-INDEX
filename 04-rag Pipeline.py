import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader

load_dotenv()

# 1. load data
documents =SimpleDirectoryReader("./data/").load_data()

# 2. Create Index
# pip3 install chromadb llama-index-vector-stores-chroma

