import os
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId
from llama_index.core import VectorStoreIndex, Document
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.core.settings import Settings
from typing import Dict, Any, List
import json
from datetime import datetime

class MongoJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class TenantDatabaseChatbot:
    def __init__(self):
        # Initialize MongoDB
        load_dotenv()
        mongo_uri = os.getenv("MONGODB_URI")
        print(f"Connecting to MongoDB...")
        self.mongo_client = MongoClient(mongo_uri)
        self.db = self.mongo_client.tn_txbglk
        
        # Initialize OpenAI
        self.llm = OpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
        
        # Configure global settings
        Settings.llm = self.llm
        
        # Get collections
        self.collections = self._get_collections()
        print(f"Available collections: {list(self.collections.keys())}")
        
        # Initialize tools and agent
        self.tools = self._create_tools()
        self.agent = self._create_agent()

    def _get_collections(self) -> Dict[str, Any]:
        """Get all available collections in the database."""
        collections = {}
        try:
            collection_names = self.db.list_collection_names()
            print(f"Found collections: {collection_names}")
            
            for name in collection_names:
                collection = self.db[name]
                sample_doc = collection.find_one()
                if sample_doc:
                    collections[name] = {
                        "collection": collection,
                        "sample_schema": {k: type(v).__name__ for k, v in sample_doc.items()}
                    }
            return collections
        except Exception as e:
            print(f"Error getting collections: {str(e)}")
            return {}

    def _format_document(self, doc: Dict) -> Dict:
        """Format a single document for output."""
        formatted = {}
        for key, value in doc.items():
            if isinstance(value, ObjectId):
                formatted[key] = str(value)
            elif isinstance(value, datetime):
                formatted[key] = value.isoformat()
            elif isinstance(value, (list, dict)):
                # Handle nested structures
                formatted[key] = json.loads(json.dumps(value, cls=MongoJSONEncoder))
            else:
                formatted[key] = value
        return formatted

    def _format_results(self, results: List[Dict]) -> str:
        """Format a list of documents for output."""
        formatted_results = [self._format_document(doc) for doc in results]
        return json.dumps(formatted_results, indent=2, ensure_ascii=False)

    def search_collection(self, collection: str, query: Dict = None, limit: int = 10) -> str:
        """Search documents in a specific collection."""
        try:
            if collection not in self.collections:
                available = list(self.collections.keys())
                return f"Collection '{collection}' not found. Available collections: {available}"
            
            mongo_collection = self.collections[collection]["collection"]
            results = list(mongo_collection.find(query or {}).limit(limit))
            
            if not results:
                return f"No documents found in collection '{collection}'"
                
            return self._format_results(results)
        except Exception as e:
            return f"Error searching collection: {str(e)}"

    def get_collection_stats(self, collection: str) -> str:
        """Get statistics about a specific collection."""
        try:
            if collection not in self.collections:
                available = list(self.collections.keys())
                return f"Collection '{collection}' not found. Available collections: {available}"
            
            mongo_collection = self.collections[collection]["collection"]
            stats = {
                "total_documents": mongo_collection.count_documents({}),
                "fields": list(self.collections[collection]["sample_schema"].keys())
            }
            return json.dumps(stats, indent=2)
        except Exception as e:
            return f"Error getting collection stats: {str(e)}"

    def get_recent_documents(self, collection: str, limit: int = 5) -> str:
        """Get most recent documents from a collection."""
        try:
            if collection not in self.collections:
                available = list(self.collections.keys())
                return f"Collection '{collection}' not found. Available collections: {available}"
            
            mongo_collection = self.collections[collection]["collection"]
            
            # Choose appropriate sort field based on collection
            if collection == "contacts":
                results = list(mongo_collection.find().sort("sequenceIdReadOnly", -1).limit(limit))
            else:
                results = list(mongo_collection.find().sort("createdAt", -1).limit(limit))
            
            if not results:
                return f"No documents found in collection '{collection}'"
                
            return self._format_results(results)
        except Exception as e:
            return f"Error getting recent documents: {str(e)}"

    def _create_tools(self) -> List[FunctionTool]:
        """Create tools for database interaction."""
        return [
            FunctionTool.from_defaults(
                fn=self.search_collection,
                name="search_collection",
                description="Search for documents in a specific collection. Args: collection (str), query (dict, optional), limit (int, optional)"
            ),
            FunctionTool.from_defaults(
                fn=self.get_collection_stats,
                name="get_collection_stats",
                description="Get statistics about a specific collection. Args: collection (str)"
            ),
            FunctionTool.from_defaults(
                fn=self.get_recent_documents,
                name="get_recent_documents",
                description="Get the most recent documents from a collection. Args: collection (str), limit (int, optional)"
            )
        ]

    def _create_agent(self) -> ReActAgent:
        """Create the ReAct agent with collection context."""
        available_collections = list(self.collections.keys())
        
        context = f"""You are an AI assistant managing a MongoDB database.
        Available collections: {available_collections}
        
        Collection schemas:
        {json.dumps({k: v['sample_schema'] for k, v in self.collections.items()}, indent=2)}
        
        When searching for contacts, use the 'contacts' collection.
        When searching for messages, use the 'messages' collection.
        When searching for conversations, use the 'conversations' collection.
        When working with contacts, remember they are sorted by sequenceIdReadOnly.
        """

        return ReActAgent.from_tools(
            self.tools,
            llm=self.llm,
            verbose=True,
            context=context
        )

    def chat(self):
        """Start an interactive chat session."""
        print("\nAI Database Assistant Initialized")
        print(f"Available collections: {list(self.collections.keys())}")
        print("\nYou can ask questions like:")
        print("- Show me recent contacts")
        print("- How many messages do we have?")
        print("- Find conversations from last week")
        print("- Show me customer details")
        
        while True:
            user_input = input("\nUser: ").strip()
            if user_input.lower() == 'exit':
                break
            
            try:
                response = self.agent.chat(user_input)
                print(f"\nAssistant: {response}")
            except Exception as e:
                print(f"\nError: {str(e)}")
                print("Please try rephrasing your question.")

if __name__ == "__main__":
    chatbot = TenantDatabaseChatbot()
    chatbot.chat()