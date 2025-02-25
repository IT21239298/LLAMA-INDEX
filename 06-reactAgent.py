import os 
from dotenv import load_dotenv
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool

load_dotenv()

llm = OpenAI(model="gpt-4o-mini")

def add(a: float, b: float) -> float:
    "Add two numbers and return the sum"
    return a + b

def subtract(a: float, b: float) -> float:
    "Subtract two numbers and returns the difference"
    return a - b

def multiplay(a: float, b: float) -> float:
    "Multiply two numbers and returns the product"
    return a * b

def divide(a: float, b: float) -> float:
    "Divide two numbers and returns the quotient"
    return a / b
#tools
add_tool = FunctionTool.from_defaults(fn=add)
sub_tool = FunctionTool.from_defaults(fn=subtract)
mul_tool = FunctionTool.from_defaults(fn=multiplay)
div_tool = FunctionTool.from_defaults(fn=divide)

#cll to reactAgent
# verbose parameter controls the level of output detail during the agent's execution.
# When verbose=True:
#     The agent will print detailed information about its thought process
# When verbose=False:
#     The agent will only show the final result

agent = ReActAgent.from_tools([add_tool, sub_tool, mul_tool, div_tool], llm=llm, verbose=True)
response = agent.chat("what is 20 + (2 * 4) / (6 - 2)? Use a tool to calculate every step.")
print(response)