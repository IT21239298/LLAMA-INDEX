from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI

load_dotenv()

llm = OpenAI(model="gpt-4o-mini")
data = SimpleDirectoryReader("./data/").load_data()
index = VectorStoreIndex.from_documents(data)

# to chat_mode we can use,

#  Chat modes:
#             - `ChatMode.BEST` (default): Chat engine that uses an agent (react or openai) with a query engine tool
#             - `ChatMode.CONTEXT`: Chat engine that uses a retriever to get context
#             - `ChatMode.CONDENSE_QUESTION`: Chat engine that condenses questions
#             - `ChatMode.CONDENSE_PLUS_CONTEXT`: Chat engine that condenses questions and uses a retriever to get context
#             - `ChatMode.SIMPLE`: Simple chat engine that uses the LLM directly
#             - `ChatMode.REACT`: Chat engine that uses a react agent with a query engine tool
#             - `ChatMode.OPENAI`: Chat engine that uses an openai agent with a query engine tool
#         """

chat_engine = index.as_chat_engine(chat_mode="best", llm=llm, verbose = True)

response = chat_engine.chat("what are the first programs Paul Graham tried writing?")

print(response)