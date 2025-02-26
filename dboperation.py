import os
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from bson import ObjectId
from llama_index.core import VectorStoreIndex, Document, ServiceContext
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.core.settings import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.schema import TextNode
from typing import Dict, Any, List, Union, Tuple
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
        
        # Initialize OpenAI components
        self.llm = OpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.2)
        self.embed_model = OpenAIEmbedding(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Set up global settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        
        self.data_cache = {}
        
        # Enhanced collection configuration with required fields and descriptions
        self.collection_config = {
            'contacts': {
                'unique_fields': ['email', 'primaryContactNo', 'sequenceId'],
                'sort_field': 'sequenceIdReadOnly',
                'required_fields': ['email', 'firstName', 'lastName'],
                'description': 'Contact information for individuals including email, name, phone numbers'
            },
            'messages': {
                'unique_fields': ['messageId'],
                'sort_field': 'createdAt',
                'required_fields': ['messageId', 'content', 'sender'],
                'description': 'Communication messages with content, sender, and timestamp information'
            },
            'conversations': {
                'unique_fields': ['conversationId'],
                'sort_field': 'createdAt',
                'required_fields': ['conversationId', 'participants'],
                'description': 'Message threads between participants with conversation history'
            },
            'customers': {
                'unique_fields': ['customerId', 'email'],
                'sort_field': 'createdAt',
                'required_fields': ['customerId', 'email'],
                'description': 'Customer profiles with personal information and preferences'
            },
            'cases': {
                'unique_fields': ['caseId'],
                'sort_field': 'createdAt',
                'required_fields': ['caseId', 'status', 'priority'],
                'description': 'Support or service cases with status, priority, and related information'
            },
            'hotels': {
                'unique_fields': ['_id'],
                'sort_field': 'name',
                'required_fields': ['name', 'description', 'location'],
                'description': 'Hotel listings with amenities, location, pricing, and availability information'
            },
            'attractions': {
                'unique_fields': ['_id'],
                'sort_field': 'name',
                'required_fields': ['name', 'category', 'description'],
                'description': 'Tourist attractions with category, description, and location details'
            },
            'cruises': {
                'unique_fields': ['_id'],
                'sort_field': 'name',
                'required_fields': ['name', 'cruise_line', 'departure_port', 'duration_nights', 'price_per_person_AED'],
                'description': 'Cruise packages with pricing, duration, ports, and amenities information'
            },
            'rewards': {
                'unique_fields': ['_id', 'reward_id'],
                'sort_field': 'name',
                'required_fields': ['name', 'description', 'points_required'],
                'description': 'Loyalty rewards and redemption options for customers'
            },
            'users': {
                'unique_fields': ['_id', 'user_id', 'email'],
                'sort_field': 'created_at',
                'required_fields': ['email', 'username'],
                'description': 'User accounts with authentication and profile information'
            },
            'attachments': {
                'unique_fields': ['_id', 'file_id'],
                'sort_field': 'uploaded_at',
                'required_fields': ['file_name', 'file_type', 'file_size'],
                'description': 'File attachments related to cases, messages, or other records'
            }
        }
        
        self.collections = self._get_collections()
        print(f"Available collections: {list(self.collections.keys())}")
        
        # Create vector index for collection matching
        self._create_collection_index()
        
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
                    # Extract field names with types and add example values
                    field_info = {}
                    for k, v in sample_doc.items():
                        field_info[k] = {
                            'type': type(v).__name__,
                            'example': str(v)[:50] if v is not None else None
                        }
                    
                    collections[name] = {
                        "collection": collection,
                        "sample_schema": field_info,
                        "field_names": list(sample_doc.keys())
                    }
            return collections
        except Exception as e:
            print(f"Error getting collections: {str(e)}")
            return {}    

    def _create_collection_index(self):
        """Create a vector index for collection matching."""
        # Create documents for each collection with rich context
        documents = []
        
        for coll_name, config in self.collection_config.items():
            # Skip if collection doesn't exist
            if coll_name not in self.collections:
                continue
                
            # Get field names and sample values
            field_names = self.collections[coll_name].get("field_names", [])
            
            # Create document with descriptive text
            doc_text = f"Collection: {coll_name}\n"
            doc_text += f"Description: {config.get('description', 'No description available')}\n"
            doc_text += f"Contains fields: {', '.join(field_names)}\n"
            doc_text += f"Unique identifiers: {', '.join(config.get('unique_fields', []))}\n"
            doc_text += f"Required fields: {', '.join(config.get('required_fields', []))}\n"
            
            # Add some example queries
            doc_text += "Example queries for this collection:\n"
            
            # Generate query examples based on collection type
            if coll_name == 'contacts':
                doc_text += "- Find contact information for a person\n"
                doc_text += "- Get contact details by email\n"
                doc_text += "- Search for a contact by phone number\n"
            elif coll_name == 'hotels':
                doc_text += "- Find hotels in a specific location\n"
                doc_text += "- Get information about a hotel by name\n"
                doc_text += "- Search for hotels with specific amenities\n"
            elif coll_name == 'attractions':
                doc_text += "- Find tourist attractions in a location\n"
                doc_text += "- Get information about a specific attraction\n"
                doc_text += "- Search for attractions by category\n"
            elif coll_name == 'cruises':
                doc_text += "- Find cruise packages with specific duration\n"
                doc_text += "- Get information about a cruise by name\n"
                doc_text += "- Search for cruises by departure port\n"
            
            # Create document
            node = TextNode(text=doc_text, metadata={"collection": coll_name})
            documents.append(node)
        
        # Create vector index
        self.collection_index = VectorStoreIndex(documents)

    def determine_collection(self, query_text: str) -> List[str]:
        """
        Determine the most likely collection(s) for a given query.
        Returns a list of collection names sorted by relevance.
        """
        # Use vector search to find most relevant collections
        query_engine = self.collection_index.as_query_engine(similarity_top_k=3)
        response = query_engine.query(query_text)
        
        # Extract collection names from metadata
        collections = []
        if hasattr(response, 'source_nodes') and response.source_nodes:
            for node in response.source_nodes:
                coll_name = node.metadata.get("collection")
                if coll_name and coll_name not in collections:
                    collections.append(coll_name)
        
        # If no collections found, return all available
        if not collections:
            collections = list(self.collections.keys())
            
        return collections

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

    def smart_search(self, query_text: str, limit: int = 10) -> str:
        """
        Intelligently search across collections based on the query text.
        This function determines the most relevant collections and searches them.
        """
        try:
            # Determine most likely collections for this query
            likely_collections = self.determine_collection(query_text)
            
            if not likely_collections:
                return json.dumps({"error": "Could not determine appropriate collection for this query"})
            
            # Extract potential search terms using LLM
            prompt = f"""
            Extract search terms from this query: "{query_text}"
            Format the response as a JSON with field names and values to search for.
            Only include fields that are explicitly mentioned or strongly implied.
            Keep it simple and use basic field names that would appear in a database.
            """
            
            search_terms_response = self.llm.complete(prompt)
            search_terms_text = search_terms_response.text
            
            # Extract JSON from the response
            try:
                # Find JSON in the response
                start_idx = search_terms_text.find('{')
                end_idx = search_terms_text.rfind('}')
                
                if start_idx != -1 and end_idx != -1:
                    json_str = search_terms_text[start_idx:end_idx+1]
                    search_terms = json.loads(json_str)
                else:
                    # Fallback: use a simple text-based search
                    search_terms = {}
                    words = [w for w in query_text.split() if len(w) > 3]
                    if words:
                        # Create simple search term for name or description
                        term = " ".join(words)
                        search_terms = {"name": term}
            except json.JSONDecodeError:
                # Fallback for parsing issues
                search_terms = {"name": query_text}
                
            print(f"Search terms extracted: {search_terms}")
                
            # Special case handling for attractions
            if "attractions" in likely_collections and ("attractions" in query_text.lower() or len(likely_collections) == 1):
                # Prioritize the attractions collection for attraction searches
                specific_collection = "attractions"
                
                # Direct lookup in attractions collection
                try:
                    # Get all attractions if no specific search terms
                    attractions_results = list(self.collections["attractions"]["collection"].find({}).limit(limit))
                    
                    if attractions_results:
                        # Format in a simple, structured way designed to avoid parsing errors
                        formatted_attractions = []
                        
                        for attraction in attractions_results:
                            formatted_attraction = self._format_document(attraction)
                            formatted_attractions.append(formatted_attraction)
                        
                        return json.dumps({
                            "collection": "attractions",
                            "count": len(formatted_attractions),
                            "data": formatted_attractions
                        })
                except Exception as e:
                    print(f"Error in attractions search: {str(e)}")
            
            # Perform searches
            all_results = {}
            for collection in likely_collections:
                # Skip if collection is not available
                if collection not in self.collections:
                    continue
                    
                # Try to match search terms to collection fields
                collection_fields = self.collections[collection].get("field_names", [])
                query = {}
                
                # Build query based on field matching
                for term_key, term_value in search_terms.items():
                    if term_key in collection_fields:
                        # Direct field match
                        if isinstance(term_value, str):
                            # Use regex for partial text matching
                            query[term_key] = {"$regex": term_value, "$options": "i"}
                        else:
                            query[term_key] = term_value
                
                # If no direct field matches, try name and description search
                if not query and search_terms:
                    # Generic search across common fields
                    if "name" in collection_fields:
                        query["name"] = {"$regex": query_text, "$options": "i"}
                    elif "description" in collection_fields:
                        query["description"] = {"$regex": query_text, "$options": "i"}
                    else:
                        # Use the first available field as fallback
                        if collection_fields:
                            query[collection_fields[0]] = {"$regex": query_text, "$options": "i"}
                
                # If still no results, get recent documents as a fallback
                if not query:
                    # Get all documents in this collection as a fallback
                    try:
                        results = list(self.collections[collection]["collection"].find({}).limit(limit))
                        if results:
                            all_results[collection] = results
                    except Exception as e:
                        print(f"Error getting fallback for {collection}: {str(e)}")
                    continue
                
                # Execute the query
                try:
                    results = list(self.collections[collection]["collection"].find(query).limit(limit))
                    if results:
                        all_results[collection] = results
                except Exception as e:
                    print(f"Error searching {collection}: {str(e)}")
            
            # Format and return results
            if not all_results:
                # Return empty but well-structured result to avoid parsing errors
                return json.dumps({
                    "status": "no_results",
                    "message": "No matching documents found in any collection",
                    "searched_collections": likely_collections
                })
            
            formatted_results = {}
            for coll, results in all_results.items():
                formatted_results[coll] = json.loads(self._format_results(results))
            
            # Return a simplified structure that's less prone to parsing errors
            return json.dumps({
                "status": "success",
                "matched_collections": list(formatted_results.keys()),
                "results": formatted_results
            })
            
        except Exception as e:
            print(f"Smart search error: {str(e)}")
            # Return a simplified error response
            return json.dumps({
                "status": "error",
                "message": f"Search encountered an error: {str(e)}",
                "error_details": str(e)
            })

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
                fn=self.smart_search,
                name="smart_search",
                description="Intelligently search across collections based on the query text. Args: query_text (str), limit (int, optional)"
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
        1. ALWAYS use smart_search for natural language queries about database contents
        2. For specific collection searches, use search_collection
        3. For each collection, use appropriate search criteria:
           - attractions: search by '_id', 'name', or 'category', or filter by 'description', or 'location'
           - cruises: search by '_id', 'name', 'cruise_line', or 'departure_port', or 'duration_nights',or 'price_per_person_AED'
           - hotels: search by '_id', 'name', or filter by facilities or 'hotel_type' or 'hotel_class', or 'location' or 'hotel'
           - contacts: email, primaryContactNo, sequenceId
           - messages: messageId
           - conversations: conversationId
           - customers: customerId, email
           - cases: caseId
        4. When searching across collections:
           - Use smart_search for natural language queries
           - Use get_related_documents for linking specific data points
        5. Never make assumptions about data - always verify
        6. For any uncertain responses, check the database again
        7. When creating documents, ensure all required fields are provided
        8. When updating documents:
           - Verify the document exists first
           - Only update specified fields
           - Maintain data integrity
        9. Before deleting:
           - Verify the document exists
           - Check for dependent documents in other collections
        
        Collection schemas:
        {json.dumps({k: {"fields": list(v['sample_schema'].keys()), 
                         "description": self.collection_config.get(k, {}).get('description', '')} 
                    for k, v in self.collections.items()}, indent=2)}
        
        Collection configurations:
        {json.dumps(self.collection_config, indent=2)}
        
        CRITICAL: When answering user queries, follow this process:
        
        1. REASONING STEP:
           - First, analyze what the user is asking for
           - Determine which collection is most relevant
           - Identify key search parameters or filters
        
        2. SELECTION STEP:
           - Choose the best tool for the task:
             * For natural language queries, use smart_search
             * For specific collection lookups, use search_collection
             * For relationship data, use get_related_documents
        
        3. EXECUTION STEP:
           - Execute the chosen tool with appropriate parameters
           - If no results are found, try alternative search terms or collections
           - Always verify data is returned before making conclusions
        
        4. FORMATTING STEP:
           - Transform raw results into user-friendly structured information
           - Include relevant fields like name, description, location, etc.
           - Present information in a clear, organized manner
           - For list results, number items and include 3-5 key attributes for each
           - For detailed results, organize by logical sections
        
        For attractions specifically:
        - Always include name, category, description, and location
        - Format with clear section headers and good readability
        - Avoid just presenting raw JSON data to the user
        
        NEVER skip any of these steps. Your output should always follow logical reasoning, provide accurate data, and be well-organized.
        """

        # Configure the Settings with the LLM and embed_model
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        
        # Set the temperature in the llm
        self.llm.temperature = 0.2  # Set temperature for better reasoning
        
        # Create the ReAct agent with the updated settings
        agent = ReActAgent.from_tools(
            self.tools,
            verbose=True,
            context=context
        )
        
        # Add custom error handling for output parsing issues
        agent.reset()
        
        return agent

    def chat(self):
        """Start an interactive chat session with enhanced CRUD capabilities."""
        print("\nAI Database Assistant Initialized with Semantic Search Support")
        print(f"Available collections: {list(self.collections.keys())}")
        print("\nYou can perform operations like:")
        print("- Natural queries: 'Find hotels in Dubai with a pool'")
        print("- Create: 'Create a new contact with email=[email]'")
        print("- Read: 'Show me recent contacts'")
        print("- Update: 'Update status for case [caseId]'")
        print("- Delete: 'Delete message [messageId]'")
        print("- Bulk Update: 'Update status for all open cases'")
        
        while True:
            user_input = input("\nUser: ").strip()
            if user_input.lower() == 'exit':
                break
            
            try:
                self.data_cache = {}
                
                # First, analyze the user query using LLM to determine intent and appropriate collection
                intent_analysis_prompt = f"""
                Analyze this database query: "{user_input}"
                
                Available collections: {list(self.collections.keys())}
                
                1. What is the primary intent? (search, create, update, delete, etc.)
                2. Which collection(s) are most relevant?
                3. What are the key search terms or entities?
                4. Are there any specific fields that should be used for filtering?
                
                Format your response as JSON:
                {{
                  "intent": "search|create|update|delete",
                  "primary_collection": "most_relevant_collection_name",
                  "secondary_collections": ["other_collection1", "other_collection2"],
                  "search_terms": {{"field1": "value1", "field2": "value2"}},
                  "response_format": "preferred response format (list, details, summary)",
                  "reasoning": "brief explanation of your analysis"
                }}
                """
                
                # Get the LLM's analysis of the query
                analysis_response = self.llm.complete(intent_analysis_prompt)
                analysis_text = analysis_response.text
                
                # Extract the JSON response
                try:
                    # Find JSON in the response
                    start_idx = analysis_text.find('{')
                    end_idx = analysis_text.rfind('}')
                    
                    if start_idx != -1 and end_idx != -1:
                        intent_json = analysis_text[start_idx:end_idx+1]
                        intent_data = json.loads(intent_json)
                        
                        # Log the analysis for debugging
                        print(f"\nQuery analysis: {json.dumps(intent_data, indent=2)}")
                        
                        # Use the analysis to guide the search
                        primary_collection = intent_data.get("primary_collection")
                        search_terms = intent_data.get("search_terms", {})
                        response_format = intent_data.get("response_format", "list")
                        
                        # Enhanced search strategy based on LLM's thinking
                        if primary_collection in self.collections:
                            if intent_data.get("intent") == "search":
                                # Direct collection access guided by LLM's thinking
                                if search_terms:
                                    # Build a MongoDB query based on the search terms
                                    mongo_query = {}
                                    for field, value in search_terms.items():
                                        if isinstance(value, str):
                                            # Use regex for text fields for partial matching
                                            mongo_query[field] = {"$regex": value, "$options": "i"}
                                        else:
                                            mongo_query[field] = value
                                    
                                    print(f"\nExecuting guided query: {mongo_query} on collection {primary_collection}")
                                    results = list(self.collections[primary_collection]["collection"].find(mongo_query).limit(10))
                                else:
                                    # No specific search terms, get recent documents
                                    results = list(self.collections[primary_collection]["collection"].find().limit(10))
                                
                                # Format and present results based on the identified format preference
                                if results:
                                    # Create a prompt for the LLM to format the results based on collection type
                                    result_json = json.dumps([self._format_document(doc) for doc in results], cls=MongoJSONEncoder)
                                    
                                    format_prompt = f"""
                                    Format these {primary_collection} search results for a user:
                                    {result_json}
                                    
                                    Present the information in a {response_format} format.
                                    Include all important details organized in a clear, readable structure.
                                    For each item, include the name, and any other key information like location, description, etc.
                                    """
                                    
                                    # Get formatted results from LLM
                                    formatted_response = self.llm.complete(format_prompt)
                                    print(f"\nAssistant: {formatted_response.text}")
                                    continue
                except Exception as analysis_error:
                    print(f"\nError in query analysis: {str(analysis_error)}")
                    # Continue with standard processing if analysis fails
                
                # Standard processing using the agent
                response = self.agent.chat(user_input)
                print(f"\nAssistant: {response}")
                
            except Exception as e:
                print(f"\nError in processing: {str(e)}")
                # Fallback with simpler pattern when everything else fails
                try:
                    # Attempt to determine the most relevant collection
                    likely_collections = self.determine_collection(user_input)
                    
                    if likely_collections:
                        primary_collection = likely_collections[0]
                        print(f"\nI'll try to search the {primary_collection} collection for you.")
                        
                        try:
                            # Simple text search across name and description fields
                            results = list(self.collections[primary_collection]["collection"].find().limit(5))
                            
                            if results:
                                print(f"\nAssistant: Here are some results from {primary_collection}:")
                                for idx, item in enumerate(results, 1):
                                    print(f"\n{idx}. {item.get('name', 'Unnamed')}")
                                    if "description" in item:
                                        print(f"   Description: {item['description']}")
                            else:
                                print(f"\nAssistant: I couldn't find any results in the {primary_collection} collection.")
                        except Exception as search_error:
                            print(f"\nSearch error: {str(search_error)}")
                            print("Please try a different query.")
                    else:
                        print("\nAssistant: I couldn't determine which collection to search. Please be more specific.")
                
                except Exception as fallback_error:
                    print(f"\nFallback error: {str(fallback_error)}")
                    print("Please try rephrasing your question.")

if __name__ == "__main__":
    chatbot = TenantDatabaseChatbot()
    chatbot.chat()