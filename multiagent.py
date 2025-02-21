import os
import asyncio
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from typing import List, Optional, Dict, Any

# Import the existing agents (ensure these are in the same directory)
from websiteReader import WebsiteReader
from dbchatbot import TenantDatabaseChatbot

class MultiAgentSupervisor:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize OpenAI LLM
        self.llm = OpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
        Settings.llm = self.llm
        
        # Create individual agents
        self.website_reader_agent = self._create_website_reader_agent()
        self.database_agent = self._create_database_agent()

    def _create_website_reader_agent(self) -> ReActAgent:
        """Create a ReAct agent for the WebsiteReader."""
        website_reader = WebsiteReader()
        
        # Create a tool for answering website-related questions
        website_tool = FunctionTool.from_defaults(
            fn=website_reader.answer_question,
            name="website_search",
            description="Search and retrieve information from the eConsulate website. Provide detailed answers to queries about the website content."
        )
        
        return ReActAgent.from_tools(
            [website_tool],
            llm=self.llm,
            verbose=True,
            system_prompt="You are an agent specialized in searching and retrieving information from the eConsulate website. Provide precise and relevant information."
        )

    def _create_database_agent(self) -> ReActAgent:
        """Create a ReAct agent for the TenantDatabaseChatbot."""
        database_chatbot = TenantDatabaseChatbot()
        
        # Create function tools from the existing database tools
        database_tools = database_chatbot.tools
        
        return ReActAgent.from_tools(
            database_tools,
            llm=self.llm,
            verbose=True,
            system_prompt="You are an agent specialized in querying and retrieving information from the MongoDB database. Provide accurate and detailed database information."
        )

    def _determine_agent(self, query: str) -> ReActAgent:
        """Determine which agent to use based on the query."""
        # Use the LLM to classify the query
        classification_prompt = f"""
        Classify the following query to determine which agent should handle it:
        
        Query: {query}
        
        Possible agents:
        1. WEBSITE: For queries about eConsulate website content, services, or information
        2. DATABASE: For queries about specific records, contacts, messages, or database-related information
        
        Respond ONLY with 'WEBSITE' or 'DATABASE':
        """
        
        try:
            classification = self.llm.complete(classification_prompt).text.strip().upper()
            
            if 'WEBSITE' in classification:
                return self.website_reader_agent
            elif 'DATABASE' in classification:
                return self.database_agent
            else:
                # Default to database agent if unclear
                return self.database_agent
        except Exception as e:
            print(f"Error in agent classification: {e}")
            # Fallback to database agent
            return self.database_agent

    def chat(self):
        """Start an interactive multi-agent chat session."""
        print("\nMulti-Agent AI Assistant")
        print("This assistant can help you with website information and database queries.")
        print("Available agents: Website Reader, Database Query")
        print("Type 'exit' to end the conversation.")
        
        while True:
            user_input = input("\nUser: ").strip()
            
            if user_input.lower() == 'exit':
                break
            
            try:
                # Determine the appropriate agent
                selected_agent = self._determine_agent(user_input)
                
                # Process the query with the selected agent
                response = selected_agent.chat(user_input)
                print(f"\nAssistant: {response}")
            
            except Exception as e:
                print(f"\nError: {str(e)}")
                print("Please try rephrasing your question.")

def main():
    multi_agent_system = MultiAgentSupervisor()
    multi_agent_system.chat()

if __name__ == "__main__":
    main()