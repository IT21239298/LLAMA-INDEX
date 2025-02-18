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
    """Custom JSON encoder for MongoDB specific types."""
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
        
        # Initialize cache
        self.data_cache = {}
        
        # Define collection-specific configurations
        self.collection_config = {
            'contacts': {
                'unique_fields': ['email', 'primaryContactNo', 'sequenceId'],
                'sort_field': 'sequenceIdReadOnly'
            },
            'messages': {
                'unique_fields': ['messageId'],
                'sort_field': 'createdAt'
            },
            'conversations': {
                'unique_fields': ['conversationId'],
                'sort_field': 'createdAt'
            },
            'customers': {
                'unique_fields': ['customerId', 'email'],
                'sort_field': 'createdAt'
            },
            'cases': {
                'unique_fields': ['caseId'],
                'sort_field': 'createdAt'
            }
        }
        
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

    def verify_document(self, collection: str, field: str, value: str) -> str:
        """Verify document existence in any collection by field value."""
        try:
            if collection not in self.collections:
                return json.dumps({"error": f"Collection '{collection}' not found"})

            query = {field: value}
            mongo_collection = self.collections[collection]["collection"]
            results = list(mongo_collection.find(query))

            if not results:
                return json.dumps({"error": f"No document found in {collection} with {field}: {value}"})

            return self._format_results(results)
        except Exception as e:
            return json.dumps({"error": f"Error verifying document: {str(e)}"})

    def get_document_details(self, collection: str, query_params: Dict) -> str:
        """Get detailed document information from any collection."""
        try:
            if collection not in self.collections:
                return json.dumps({"error": f"Collection '{collection}' not found"})

            # Build query based on provided parameters
            query = {}
            config = self.collection_config.get(collection, {})
            unique_fields = config.get('unique_fields', [])

            for field in unique_fields:
                if field in query_params:
                    query[field] = query_params[field]

            if not query:
                return json.dumps({"error": "No valid search criteria provided"})

            results = json.loads(self.search_collection(collection, query, 1))
            
            if isinstance(results, dict) and "error" in results:
                return json.dumps(results)

            return json.dumps({
                "found": True,
                "document": results[0] if results else None
            })
        except Exception as e:
            return json.dumps({"error": f"Error getting document details: {str(e)}"})

    def get_related_documents(self, collection: str, ref_field: str, ref_value: str) -> str:
        """Get related documents across collections."""
        try:
            if collection not in self.collections:
                return json.dumps({"error": f"Collection '{collection}' not found"})

            query = {ref_field: ref_value}
            results = json.loads(self.search_collection(collection, query))

            if isinstance(results, dict) and "error" in results:
                return json.dumps(results)

            return json.dumps({
                "found": True,
                "documents": results
            })
        except Exception as e:
            return json.dumps({"error": f"Error getting related documents: {str(e)}"})

    def search_collection(self, collection: str, query: Dict = None, limit: int = 10) -> str:
        """Search documents in a specific collection."""
        try:
            if collection not in self.collections:
                available = list(self.collections.keys())
                return json.dumps({"error": f"Collection '{collection}' not found", "available": available})
            
            mongo_collection = self.collections[collection]["collection"]
            
            # Add validation for query
            if query and not isinstance(query, dict):
                return json.dumps({"error": "Invalid query format"})
            
            results = list(mongo_collection.find(query or {}).limit(limit))
            
            if not results:
                return json.dumps({"error": f"No documents found in collection '{collection}' matching query"})
                
            return self._format_results(results)
        except Exception as e:
            return json.dumps({"error": f"Error searching collection: {str(e)}"})

    def get_recent_documents(self, collection: str, limit: int = 5) -> str:
        """Get most recent documents from a collection."""
        try:
            if collection not in self.collections:
                available = list(self.collections.keys())
                return json.dumps({"error": f"Collection '{collection}' not found", "available": available})
            
            mongo_collection = self.collections[collection]["collection"]
            config = self.collection_config.get(collection, {})
            sort_field = config.get('sort_field', 'createdAt')
            
            results = list(mongo_collection.find().sort(sort_field, -1).limit(limit))
            
            if not results:
                return json.dumps({"error": f"No documents found in collection '{collection}'"})
                
            return self._format_results(results)
        except Exception as e:
            return json.dumps({"error": f"Error getting recent documents: {str(e)}"})

    def _create_tools(self) -> List[FunctionTool]:
        """Create tools for database interaction."""
        return [
            FunctionTool.from_defaults(
                fn=self.search_collection,
                name="search_collection",
                description="Search any collection with custom query. Args: collection (str), query (dict, optional), limit (int, optional)"
            ),
            FunctionTool.from_defaults(
                fn=self.verify_document,
                name="verify_document",
                description="Verify document existence in any collection. Args: collection (str), field (str), value (str)"
            ),
            FunctionTool.from_defaults(
                fn=self.get_document_details,
                name="get_document_details",
                description="Get detailed document information from any collection. Args: collection (str), query_params (dict)"
            ),
            FunctionTool.from_defaults(
                fn=self.get_related_documents,
                name="get_related_documents",
                description="Get related documents across collections. Args: collection (str), ref_field (str), ref_value (str)"
            ),
            FunctionTool.from_defaults(
                fn=self.get_recent_documents,
                name="get_recent_documents",
                description="Get recent documents from any collection. Args: collection (str), limit (int, optional)"
            )
        ]

    def _create_agent(self) -> ReActAgent:
        """Create the ReAct agent with multi-collection context."""
        context = f"""You are an AI assistant managing a MongoDB database.
        Available collections: {list(self.collections.keys())}
        
        IMPORTANT RULES:
        1. ALWAYS verify data exists before making statements
        2. For each collection, use appropriate search criteria:
           - contacts: email, primaryContactNo, sequenceId
           - messages: messageId
           - conversations: conversationId
           - customers: customerId, email
           - cases: caseId
        3. When searching across collections:
           - Use get_related_documents for linked data
           - Verify relationships exist
        4. Never make assumptions about data
        5. If multiple results exist, mention that
        6. For any uncertain responses, check the database again
        
        Collection schemas:
        {json.dumps({k: v['sample_schema'] for k, v in self.collections.items()}, indent=2)}
        
        Collection configurations:
        {json.dumps(self.collection_config, indent=2)}
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
        print("- Find messages from [email]")
        print("- Show me conversations with [customer]")
        print("- Get case details for [caseId]")
        print("- Find all messages in conversation [conversationId]")
        
        while True:
            user_input = input("\nUser: ").strip()
            if user_input.lower() == 'exit':
                break
            
            try:
                # Clear cache for new query
                self.data_cache = {}
                
                response = self.agent.chat(user_input)
                print(f"\nAssistant: {response}")
            except Exception as e:
                print(f"\nError: {str(e)}")
                print("Please try rephrasing your question.")

if __name__ == "__main__":
    chatbot = TenantDatabaseChatbot()
    chatbot.chat()