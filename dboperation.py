import os
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from bson import ObjectId
from llama_index.core import VectorStoreIndex, Document
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.core.settings import Settings
from typing import Dict, Any, List, Union
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
        # [Previous initialization code remains the same]
        load_dotenv()
        mongo_uri = os.getenv("MONGODB_URI")
        print(f"Connecting to MongoDB...")
        self.mongo_client = MongoClient(mongo_uri)
        self.db = self.mongo_client.chatbotSithum
        
        self.llm = OpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
        Settings.llm = self.llm
        self.data_cache = {}
        
        # Enhanced collection configuration with required fields
        self.collection_config = {
            'contacts': {
                'unique_fields': ['email', 'primaryContactNo', 'sequenceId'],
                'sort_field': 'sequenceIdReadOnly',
                'required_fields': ['email', 'firstName', 'lastName']
            },
            'messages': {
                'unique_fields': ['messageId'],
                'sort_field': 'createdAt',
                'required_fields': ['messageId', 'content', 'sender']
            },
            'conversations': {
                'unique_fields': ['conversationId'],
                'sort_field': 'createdAt',
                'required_fields': ['conversationId', 'participants']
            },
            'customers': {
                'unique_fields': ['customerId', 'email'],
                'sort_field': 'createdAt',
                'required_fields': ['customerId', 'email']
            },
            'cases': {
                'unique_fields': ['caseId'],
                'sort_field': 'createdAt',
                'required_fields': ['caseId', 'status', 'priority']
            },
            'hotels': {
                'unique_fields': ['_id'],
                'sort_field': 'name',
                'required_fields': ['name', 'description', 'location']
            },
            'attractions': {
                'unique_fields': ['_id'],
                'sort_field': 'name',
                'required_fields': ['name', 'category', 'description']
            },
            'cruises': {
                'unique_fields': ['_id'],
                'sort_field': 'name',
                'required_fields': ['name', 'cruise_line', 'departure_port', 'duration_nights', 'price_per_person_AED']
            },
        }
        
        self.collections = self._get_collections()
        print(f"Available collections: {list(self.collections.keys())}")
        
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

    def create_document(self, collection: str, document: Dict) -> str:
        """Create a new document in the specified collection."""
        try:
            if collection not in self.collections:
                return json.dumps({"error": f"Collection '{collection}' not found"})

            # Validate required fields
            required_fields = self.collection_config[collection]['required_fields']
            missing_fields = [field for field in required_fields if field not in document]
            if missing_fields:
                return json.dumps({
                    "error": f"Missing required fields: {missing_fields}",
                    "required_fields": required_fields
                })

            # Add timestamps
            document['createdAt'] = datetime.utcnow()
            document['updatedAt'] = document['createdAt']

            # Insert document
            mongo_collection = self.collections[collection]["collection"]
            result = mongo_collection.insert_one(document)

            # Fetch and return the created document
            created_doc = mongo_collection.find_one({"_id": result.inserted_id})
            return self._format_results([created_doc])

        except Exception as e:
            return json.dumps({"error": f"Error creating document: {str(e)}"})

    def update_document(self, collection: str, query: Dict, updates: Dict) -> str:
        """Update an existing document in the specified collection."""
        try:
            if collection not in self.collections:
                return json.dumps({"error": f"Collection '{collection}' not found"})

            mongo_collection = self.collections[collection]["collection"]

            # Validate document exists
            existing_doc = mongo_collection.find_one(query)
            if not existing_doc:
                return json.dumps({"error": f"No document found matching query: {query}"})

            # Add updated timestamp
            updates['$set'] = updates.get('$set', {})
            updates['$set']['updatedAt'] = datetime.utcnow()

            # Perform update
            result = mongo_collection.update_one(query, updates)

            if result.modified_count > 0:
                # Fetch and return the updated document
                updated_doc = mongo_collection.find_one(query)
                return self._format_results([updated_doc])
            else:
                return json.dumps({"error": "No changes made to document"})

        except Exception as e:
            return json.dumps({"error": f"Error updating document: {str(e)}"})

    def delete_document(self, collection: str, query: Dict) -> str:
        """Delete a document from the specified collection."""
        try:
            if collection not in self.collections:
                return json.dumps({"error": f"Collection '{collection}' not found"})

            mongo_collection = self.collections[collection]["collection"]

            # Validate document exists
            existing_doc = mongo_collection.find_one(query)
            if not existing_doc:
                return json.dumps({"error": f"No document found matching query: {query}"})

            # Perform deletion
            result = mongo_collection.delete_one(query)

            if result.deleted_count > 0:
                return json.dumps({
                    "success": True,
                    "message": f"Document successfully deleted from {collection}",
                    "deleted_document": self._format_document(existing_doc)
                })
            else:
                return json.dumps({"error": "No document was deleted"})

        except Exception as e:
            return json.dumps({"error": f"Error deleting document: {str(e)}"})

    def bulk_update_documents(self, collection: str, updates: List[Dict[str, Any]]) -> str:
        """Perform bulk updates on multiple documents."""
        try:
            if collection not in self.collections:
                return json.dumps({"error": f"Collection '{collection}' not found"})

            mongo_collection = self.collections[collection]["collection"]
            
            # Prepare bulk operations
            operations = []
            for update in updates:
                if 'query' not in update or 'update' not in update:
                    return json.dumps({"error": "Each update must contain 'query' and 'update' fields"})
                
                update['update']['$set'] = update['update'].get('$set', {})
                update['update']['$set']['updatedAt'] = datetime.utcnow()
                
                operations.append(
                    UpdateOne(update['query'], update['update'])
                )

            # Execute bulk update
            if operations:
                result = mongo_collection.bulk_write(operations)
                return json.dumps({
                    "success": True,
                    "matched_count": result.matched_count,
                    "modified_count": result.modified_count
                })
            else:
                return json.dumps({"error": "No update operations provided"})

        except Exception as e:
            return json.dumps({"error": f"Error performing bulk update: {str(e)}"})

    def _create_tools(self) -> List[FunctionTool]:
        """Create tools for database interaction including CRUD operations."""
        # Include previous tools
        tools = [
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

        # Add CRUD operation tools
        crud_tools = [
            FunctionTool.from_defaults(
                fn=self.create_document,
                name="create_document",
                description="Create a new document in a collection. Args: collection (str), document (dict)"
            ),
            FunctionTool.from_defaults(
                fn=self.update_document,
                name="update_document",
                description="Update an existing document. Args: collection (str), query (dict), updates (dict)"
            ),
            FunctionTool.from_defaults(
                fn=self.delete_document,
                name="delete_document",
                description="Delete a document from a collection. Args: collection (str), query (dict)"
            ),
            FunctionTool.from_defaults(
                fn=self.bulk_update_documents,
                name="bulk_update_documents",
                description="Perform bulk updates on multiple documents. Args: collection (str), updates (list of dicts)"
            )
        ]

        return tools + crud_tools

    def _create_agent(self) -> ReActAgent:
        """Create the ReAct agent with enhanced context for CRUD operations."""
        context = f"""You are an AI assistant managing a MongoDB database with full CRUD capabilities.
        Available collections: {list(self.collections.keys())}
        
        IMPORTANT RULES:
        1. ALWAYS verify data exists before making statements or modifications
        2. For each collection, use appropriate search criteria:
           - attractions: search by '_id', 'name', or 'category', or filter by 'description', or 'location'
           - cruises: search by '_id', 'name', 'cruise_line', or 'departure_port', or 'duration_nights',or 'price_per_person_AED'
           - hotels: search by '_id', 'name', or filter by facilities or 'hotel_type' or 'hotel_class', or 'location' or 'hotel'
           - contacts: email, primaryContactNo, sequenceId
           - messages: messageId
           - conversations: conversationId
           - customers: customerId, email
           - cases: caseId
        3. When searching across collections:
           - Use get_related_documents for linked data
           - Verify relationships exist
        4. Never make assumptions about data
        5. For any uncertain responses, check the database again
        6. When creating documents, ensure all required fields are provided
        7. When updating documents:
           - Verify the document exists first
           - Only update specified fields
           - Maintain data integrity
        8. Before deleting:
           - Verify the document exists
           - Check for dependent documents in other collections
        
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
        """Start an interactive chat session with enhanced CRUD capabilities."""
        print("\nAI Database Assistant Initialized with CRUD Support")
        print(f"Available collections: {list(self.collections.keys())}")
        print("\nYou can perform operations like:")
        print("- Create: Create a new contact with email=[email]")
        print("- Read: Show me recent contacts")
        print("- Update: Update status for case [caseId]")
        print("- Delete: Delete message [messageId]")
        print("- Bulk Update: Update status for all open cases")
        
        while True:
            user_input = input("\nUser: ").strip()
            if user_input.lower() == 'exit':
                break
            
            try:
                self.data_cache = {}
                response = self.agent.chat(user_input)
                print(f"\nAssistant: {response}")
            except Exception as e:
                print(f"\nError: {str(e)}")
                print("Please try rephrasing your question.")

if __name__ == "__main__":
    chatbot = TenantDatabaseChatbot()
    chatbot.chat()