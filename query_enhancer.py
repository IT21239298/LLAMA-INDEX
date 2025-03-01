import os
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from bson import ObjectId
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from typing import Dict, Any, List, Optional, Tuple, Union
import json
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import numpy as np
from functools import lru_cache

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
        load_dotenv()
        mongo_uri = os.getenv("MONGODB_URI")
        self.mongo_client = MongoClient(mongo_uri)
        self.db = self.mongo_client.chatbotSithum
        
        # Initialize LLM and embeddings
        self.llm = OpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
        Settings.llm = self.llm
        self.embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize collections and schema
        self.collections = {}
        self.schema_cache = {}
        self.field_embeddings = {}  # Store field embeddings for better semantic matching
        self._initialize_collections()
        
        # Create tools and agent
        self.tools = self._create_tools()
        self.agent = self._create_agent()
        
        # Query history for context
        self.query_history = []
        self.max_history = 5

    def _initialize_collections(self):
        """Analyze collections and their schemas efficiently."""
        collection_names = self.db.list_collection_names()
        
        for name in collection_names:
            collection = self.db[name]
            # Increase sample size for better schema inference
            sample_docs = list(collection.find().limit(20))
            
            if not sample_docs:
                continue
                
            # Create schema from samples
            schema = self._analyze_schema(sample_docs)
            
            # Extract key metadata
            unique_fields = self._identify_unique_fields(collection, sample_docs)
            required_fields = self._identify_required_fields(sample_docs)
            relationships = self._identify_relationships(name, schema)
            
            # Improved sort field detection with fallbacks
            sort_field = self._identify_sort_field(sample_docs)
            
            # Store collection metadata
            self.collections[name] = {
                "collection": collection,
                "schema": schema,
                "unique_fields": unique_fields,
                "required_fields": required_fields,
                "relationships": relationships,
                "sort_field": sort_field,
                "sample_count": len(sample_docs),
                "total_count": collection.count_documents({})
            }
            
            # Update schema cache for query guidance
            self.schema_cache[name] = {
                "fields": list(schema.keys()),
                "unique_fields": unique_fields,
                "required_fields": required_fields,
                "relationships": relationships,
                "count": collection.count_documents({})
            }
            
            # Generate embeddings for field names to improve semantic matching
            self._generate_field_embeddings(name, schema)
    
    def _generate_field_embeddings(self, collection_name: str, schema: Dict):
        """Generate embeddings for field names to improve semantic matching."""
        fields = list(schema.keys())
        if not fields:
            return
            
        field_descriptions = [f"{field} ({schema[field]['type']})" for field in fields]
        try:
            embeddings = self.embeddings.embed_documents(field_descriptions)
            self.field_embeddings[collection_name] = {
                'fields': fields,
                'embeddings': embeddings
            }
        except Exception as e:
            return json.dumps({"error": f"Error deleting document: {str(e)}"})

    def _create_tools(self) -> List[FunctionTool]:
        """Create tools for database interaction."""
        return [
            FunctionTool.from_defaults(fn=self.get_collection_schema, name="get_collection_schema",
                description="Get schema information about a collection. Args: collection (str)"),
            FunctionTool.from_defaults(fn=self.search_collection, name="search_collection",
                description="Search a collection with custom query. Args: collection (str), query (dict, optional), limit (int, optional), sort (dict, optional)"),
            FunctionTool.from_defaults(fn=self.semantic_search, name="semantic_search",
                description="Perform semantic search on a collection. Args: collection (str), search_text (str), limit (int, optional)"),
            FunctionTool.from_defaults(fn=self.natural_language_query, name="natural_language_query",
                description="Process natural language queries. Args: query_text (str), collection (str, optional)"),
            FunctionTool.from_defaults(fn=self.get_recent_documents, name="get_recent_documents",
                description="Get recent documents from a collection. Args: collection (str), limit (int, optional)"),
            FunctionTool.from_defaults(fn=self.create_document, name="create_document",
                description="Create a new document. Args: collection (str), document (dict)"),
            FunctionTool.from_defaults(fn=self.update_document, name="update_document",
                description="Update an existing document. Args: collection (str), query (dict), updates (dict)"),
            FunctionTool.from_defaults(fn=self.delete_document, name="delete_document",
                description="Delete a document. Args: collection (str), query (dict)"),
            FunctionTool.from_defaults(fn=self.run_aggregation, name="run_aggregation",
                description="Run MongoDB aggregation pipeline. Args: collection (str), pipeline (list of dict)"),
            FunctionTool.from_defaults(fn=self.search_related_documents, name="search_related_documents",
                description="Find documents related to a specific document. Args: collection (str), doc_id (str), limit (int, optional)")
        ]
    
    def _create_agent(self) -> ReActAgent:
        """Create the ReAct agent with database context."""
        schema_info = json.dumps(self.schema_cache, indent=2, cls=MongoJSONEncoder)
        
        context = f"""You are an AI assistant managing a MongoDB database with full CRUD capabilities and semantic search.
        Available collections: {list(self.collections.keys())}
        
        IMPORTANT GUIDELINES:
        1. ALWAYS verify data exists before making statements or modifications
        2. For each collection, use the following information to guide your queries:
        {schema_info}
        
        3. When searching data:
           - Use search_collection for exact matches and filtering with specific criteria
           - Use semantic_search for natural language queries about a specific collection
           - Use natural_language_query for open-ended questions across collections
           - Use search_related_documents to find connected data between collections
           
        4. For advanced analysis:
           - Use run_aggregation to perform complex aggregations, grouping, and calculations
        
        5. Always interpret MongoDB query syntax for the user
        6. Focus on providing actionable insights from the data
        7. When displaying results, format them in a user-friendly way
        8. If you're not certain about a query's intent, ask clarifying questions
        """

        return ReActAgent.from_tools(
            self.tools,
            llm=self.llm,
            verbose=True,
            context=context
        )
        
    def run_aggregation(self, collection: str, pipeline: List[Dict]) -> str:
        """Run MongoDB aggregation pipeline on a collection."""
        try:
            if collection not in self.collections:
                return json.dumps({"error": f"Collection '{collection}' not found"})

            # Run aggregation
            results = list(self.collections[collection]["collection"].aggregate(pipeline))
            
            if not results:
                return json.dumps({"error": f"No results returned from aggregation pipeline"})
                
            return self._format_results(results)
        except Exception as e:
            return json.dumps({"error": f"Error running aggregation: {str(e)}"})

    def search_related_documents(self, collection: str, doc_id: str, limit: int = 5) -> str:
        """Find documents related to a specific document."""
        try:
            if collection not in self.collections:
                return json.dumps({"error": f"Collection '{collection}' not found"})
                
            # Find the source document
            source_doc = None
            try:
                if ObjectId.is_valid(doc_id):
                    source_doc = self.collections[collection]["collection"].find_one({"_id": ObjectId(doc_id)})
            except:
                pass
                
            if not source_doc:
                # Try as string ID
                source_doc = self.collections[collection]["collection"].find_one({"_id": doc_id})
                
            if not source_doc:
                return json.dumps({"error": f"Document with ID '{doc_id}' not found in collection '{collection}'"})
                
            # Find related collections through relationships
            related_results = {}
            relationships = self.collections[collection].get("relationships", {})
            
            # Check for direct references in this document
            for field, target_collection in relationships.items():
                if field in source_doc and source_doc[field]:
                    field_value = source_doc[field]
                    
                    if target_collection in self.collections:
                        # Handle different reference types (single ID, array of IDs)
                        if isinstance(field_value, list):
                            ref_ids = field_value
                            query = {"_id": {"$in": ref_ids}}
                        else:
                            ref_ids = [field_value]
                            query = {"_id": field_value}
                            
                        # Try to find the referenced documents
                        refs = list(self.collections[target_collection]["collection"].find(query).limit(limit))
                        if refs:
                            related_results[target_collection] = refs
            
            # Check for documents in other collections referencing this document
            for other_coll, other_data in self.collections.items():
                if other_coll == collection:
                    continue
                    
                other_relationships = other_data.get("relationships", {})
                
                # Look for reverse relationships
                for field, target_collection in other_relationships.items():
                    if target_collection == collection:
                        query = {field: source_doc["_id"]}
                        refs = list(self.collections[other_coll]["collection"].find(query).limit(limit))
                        if refs:
                            related_results[other_coll] = refs
            
            # Format results
            formatted_results = {}
            for coll, docs in related_results.items():
                formatted_results[coll] = json.loads(self._format_results(docs))
                
            if not formatted_results:
                return json.dumps({"message": f"No related documents found for document '{doc_id}'"})
                
            return json.dumps(formatted_results, indent=2)
        except Exception as e:
            return json.dumps({"error": f"Error finding related documents: {str(e)}"})
            
    def chat(self):
        """Start an interactive chat session with improved query processing."""
        print(f"\nAI Database Assistant - Available collections: {list(self.collections.keys())}")
        print(f"Total documents: {sum(self.collections[coll]['total_count'] for coll in self.collections)}")
        print("Type 'exit' to quit, 'help' for guidance")
        
        while True:
            user_input = input("\nUser: ").strip()
            if user_input.lower() == 'exit':
                break
                
            if user_input.lower() == 'help':
                self._show_help()
                continue
            
            try:
                # Special command handling
                if user_input.lower().startswith(('schema ', 'show schema ')):
                    # Direct schema request
                    collection = user_input.lower().replace('schema ', '').replace('show schema ', '').strip()
                    if collection in self.collections:
                        result = self.get_collection_schema(collection)
                        parsed = json.loads(result)
                        print(f"\nSchema for collection '{collection}':")
                        print(f"Fields: {', '.join(parsed['schema'].keys())}")
                        print(f"Unique fields: {parsed['unique_fields']}")
                        print(f"Required fields: {parsed['required_fields']}")
                        if parsed.get('relationships'):
                            print(f"Relationships: {parsed['relationships']}")
                        continue
                    
                # Detect query type for optimized handling
                query_type = self._classify_query(user_input)
                
                if query_type == 'direct':
                    # Process as natural language query
                    result = self.natural_language_query(user_input)
                    try:
                        parsed_result = json.loads(result)
                        if isinstance(parsed_result, dict) and "error" in parsed_result:
                            # Fall back to agent for error cases
                            response = self.agent.chat(user_input)
                            print(f"\nAssistant: {response}")
                        else:
                            # Format and display results in a user-friendly way
                            self._display_formatted_results(parsed_result)
                    except Exception as parsing_error:
                        # Fall back to agent if result parsing fails
                        print(f"\nError parsing results: {str(parsing_error)}")
                        response = self.agent.chat(user_input)
                        print(f"\nAssistant: {response}")
                else:
                    # Use the agent for complex requests
                    response = self.agent.chat(user_input)
                    print(f"\nAssistant: {response}")
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
                print("Please try rephrasing your question.")
                
    def _show_help(self):
        """Show help information."""
        print("\n=== AI Database Assistant Help ===")
        print("Available collections:", ", ".join(self.collections.keys()))
        print("\nQuery examples:")
        print("- 'schema <collection>' - View schema for a collection")
        print("- 'find all users' - Get all documents from a collection")
        print("- 'find users where email contains gmail' - Search with criteria")
        print("- 'show me bookings for user john@example.com' - Natural language query")
        print("- 'what's the average booking value per month?' - Run aggregation")
        print("- 'create a new user with name John and email john@example.com' - Create document")
        print("- 'update user with email john@example.com to set status to active' - Update document")
        print("\nType 'exit' to quit")
    
    def _classify_query(self, query: str) -> str:
        """Classify the type of query for optimized handling."""
        query_lower = query.lower()
        
        # Direct query patterns
        direct_patterns = [
            'find', 'search', 'get', 'show', 'list', 'what', 'which', 
            'where', 'who', 'how many', 'tell me about', 'display'
        ]
        
        # CRUD operation patterns
        crud_patterns = [
            'create', 'add', 'insert', 'new', 
            'update', 'modify', 'change', 'edit',
            'delete', 'remove', 'drop'
        ]
        
        # Aggregation patterns
        agg_patterns = [
            'average', 'avg', 'sum', 'total', 'count', 'mean', 
            'group by', 'stats', 'statistics', 'report', 
            'calculate', 'aggregate', 'summarize'
        ]
        
        # Classify query type
        if any(query_lower.startswith(pattern) for pattern in direct_patterns):
            return 'direct'
        elif any(pattern in query_lower for pattern in crud_patterns):
            return 'crud'
        elif any(pattern in query_lower for pattern in agg_patterns):
            return 'aggregation'
        else:
            return 'complex'
    
    def _display_formatted_results(self, results):
        """Display results in a user-friendly format."""
        if isinstance(results, list):
            print(f"\nFound {len(results)} relevant results:")
            for i, item in enumerate(results[:5], 1):
                collection = item.pop("_collection", "unknown") if "_collection" in item else "unknown"
                score = item.pop("_similarity_score", 0) if "_similarity_score" in item else 0
                matching_context = item.pop("_matching_context", []) if "_matching_context" in item else []
                
                print(f"\n--- Result {i} (Collection: {collection}, Relevance: {score:.2f}) ---")
                
                # Display important fields first
                important_fields = ["_id", "name", "title", "email", "status"]
                for field in important_fields:
                    if field in item:
                        print(f"{field}: {item[field]}")
                        
                # Display other fields
                for key, value in item.items():
                    if key not in important_fields and not key.startswith("_"):
                        if isinstance(value, (dict, list)):
                            print(f"{key}: [Complex data]")
                        else:
                            print(f"{key}: {value}")
                
                # Display matching context if available
                if matching_context and score > 0.7:  # Only show for highly relevant results
                    print("\nMatching context:")
                    for ctx in matching_context:
                        print(f"- {ctx[:150]}..." if len(ctx) > 150 else f"- {ctx}")
        else:
            # Format dictionary results
            if isinstance(results, dict):
                if "error" in results:
                    print(f"\nError: {results['error']}")
                elif "success" in results:
                    print(f"\nSuccess: {results['message']}")
                    if "deleted_document" in results:
                        print(f"Deleted: {results['deleted_document']}")
                else:
                    print(f"\nResults: {json.dumps(results, indent=2)}")
            print(f"Warning: Could not generate field embeddings: {str(e)}")
    
    def _identify_sort_field(self, documents: List[Dict]) -> str:
        """Identify the most appropriate sort field with fallbacks."""
        # Priority order for timestamp fields
        timestamp_fields = ['createdAt', 'updatedAt', 'date', 'timestamp', 'created', 'updated']
        
        # Check if any of these fields exist in the documents
        for field in timestamp_fields:
            if any(field in doc for doc in documents):
                return field
                
        # Next, look for date fields
        date_fields = [field for field, value in documents[0].items() 
                       if isinstance(value, datetime)]
        if date_fields:
            return date_fields[0]
            
        # Default to _id if no timestamp fields found
        return '_id'
    
    def _identify_relationships(self, collection_name: str, schema: Dict) -> Dict[str, str]:
        """Identify potential relationships between collections."""
        relationships = {}
        
        # Check for fields that might reference other collections
        for field, info in schema.items():
            # Foreign key patterns
            if field.endswith('Id') or field.endswith('ID'):
                # Extract potential referenced collection name
                ref_collection = field[:-2].lower()
                # Check if plural form exists
                if ref_collection + 's' in self.db.list_collection_names():
                    relationships[field] = ref_collection + 's'
                # Check if singular form exists
                elif ref_collection in self.db.list_collection_names():
                    relationships[field] = ref_collection
            
            # Check array fields that might contain references
            if info['type'] == 'list' and field.endswith('s'):
                singular = field[:-1]
                if singular in self.db.list_collection_names() or singular + 's' in self.db.list_collection_names():
                    relationships[field] = singular if singular in self.db.list_collection_names() else singular + 's'
        
        return relationships
    
    def _analyze_schema(self, documents: List[Dict]) -> Dict:
        """Extract schema from documents efficiently with improved type detection."""
        if not documents:
            return {}
            
        schema = {}
        doc_count = len(documents)
        
        # Combine field analysis from all documents
        for doc in documents:
            for key, value in doc.items():
                if key not in schema:
                    schema[key] = {
                        "type": self._determine_type(value),
                        "sample": self._format_sample(value),
                        "null_percentage": 0,
                        "count": 0,
                        "unique_values": set()
                    }
                
                schema[key]["count"] += 1
                if value is None:
                    schema[key]["null_percentage"] += 1
                elif len(schema[key]["unique_values"]) < 10:  # Limit unique values for efficiency
                    if isinstance(value, (list, dict)):
                        schema[key]["unique_values"].add(str(value)[:100])  # Truncate long complex values
                    else:
                        schema[key]["unique_values"].add(value)
        
        # Calculate null percentages and finalize schema
        for field in schema:
            schema[field]["null_percentage"] = round((schema[field]["count"] - schema[field]["null_percentage"]) / doc_count * 100, 2)
            schema[field]["unique_values"] = list(schema[field]["unique_values"])
            schema[field]["cardinality"] = len(schema[field]["unique_values"])
                
        return schema
    
    def _determine_type(self, value) -> str:
        """Determine the type of a value with more precision."""
        if value is None:
            return "null"
        elif isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "integer"
        elif isinstance(value, float):
            return "float"
        elif isinstance(value, str):
            # Try to detect if it's a date string
            if len(value) > 8 and any(sep in value for sep in ['/', '-', ':']):
                try:
                    datetime.fromisoformat(value.replace('Z', '+00:00'))
                    return "date_string"
                except:
                    pass
            return "string"
        elif isinstance(value, datetime):
            return "date"
        elif isinstance(value, list):
            return "list"
        elif isinstance(value, dict):
            return "object"
        elif isinstance(value, ObjectId):
            return "objectid"
        else:
            return type(value).__name__
    
    def _format_sample(self, value):
        """Format sample value for schema with improved handling for complex types."""
        if isinstance(value, (ObjectId, datetime)):
            return str(value)
        elif isinstance(value, dict):
            return {k: self._format_sample(v) for k, v in list(value.items())[:5]}  # Limit nested fields
        elif isinstance(value, list) and value:
            if len(value) > 3:
                return [self._format_sample(value[0]), self._format_sample(value[1]), "..."]
            return [self._format_sample(v) for v in value]
        return value
    
    def _identify_unique_fields(self, collection, documents: List[Dict]) -> List[str]:
        """Identify potential unique fields with improved heuristics."""
        unique_fields = ['_id']  # _id is always unique
        
        # Check MongoDB indexes for unique constraints
        indexes = collection.index_information()
        for idx_name, idx_info in indexes.items():
            if idx_info.get('unique', False):
                for key in idx_info['key']:
                    if key[0] != '_id' and key[0] not in unique_fields:
                        unique_fields.append(key[0])
        
        # Check for fields with unique values and logical unique fields
        if documents:
            field_values = {}
            for doc in documents:
                for field, value in doc.items():
                    if field not in field_values:
                        field_values[field] = set()
                    
                    # Handle complex types
                    if isinstance(value, (dict, list)):
                        value = str(value)
                    field_values[field].add(value)
            
            # Check for uniqueness
            for field, values in field_values.items():
                if field != '_id' and len(values) == len(documents) and field not in unique_fields:
                    # Fields that typically contain unique values
                    if (field.endswith('Id') or field.endswith('ID') or 
                        field in ['email', 'username', 'slug', 'handle', 'code', 'sku']):
                        unique_fields.append(field)
            
        return unique_fields
    
    def _identify_required_fields(self, documents: List[Dict]) -> List[str]:
        """Identify likely required fields with improved heuristics."""
        if not documents:
            return []
            
        # Find fields present in all documents
        common_fields = set(documents[0].keys())
        for doc in documents[1:]:
            common_fields &= set(doc.keys())
        
        # Remove _id from required fields
        if '_id' in common_fields:
            common_fields.remove('_id')
            
        # Add fields that are typically required
        required_candidates = []
        all_fields = set().union(*[doc.keys() for doc in documents])
        
        # Common required field patterns
        required_patterns = [
            'id', 'name', 'email', 'title', 'key', 'code', 'status', 
            'type', 'category', 'priority', 'date'
        ]
        
        for field in all_fields:
            field_lower = field.lower()
            # Check for common required field patterns
            if any(pattern in field_lower for pattern in required_patterns):
                required_candidates.append(field)
            
            # Check if field is non-null in most documents (>90%)
            non_null_count = sum(1 for doc in documents if field in doc and doc[field] is not None)
            if non_null_count / len(documents) > 0.9:
                required_candidates.append(field)
        
        return list(common_fields.union(set(required_candidates)))
    
    def _format_document(self, doc: Dict) -> Dict:
        """Format a document for output."""
        return json.loads(json.dumps(doc, cls=MongoJSONEncoder))
    
    def _format_results(self, results: List[Dict]) -> str:
        """Format results for output."""
        return json.dumps([self._format_document(doc) for doc in results], 
                         indent=2, ensure_ascii=False)
    
    def get_collection_schema(self, collection: str) -> str:
        """Get schema information about a collection."""
        try:
            if collection not in self.collections:
                return json.dumps({"error": f"Collection '{collection}' not found", 
                                  "available": list(self.collections.keys())})
            
            return json.dumps({
                "collection": collection,
                "schema": self.collections[collection]["schema"],
                "unique_fields": self.collections[collection]["unique_fields"],
                "required_fields": self.collections[collection]["required_fields"],
                "relationships": self.collections[collection]["relationships"],
                "sort_field": self.collections[collection]["sort_field"],
                "sample_count": self.collections[collection]["sample_count"],
                "total_count": self.collections[collection]["total_count"]
            }, cls=MongoJSONEncoder)
        except Exception as e:
            return json.dumps({"error": f"Error getting schema: {str(e)}"})
    
    def search_collection(self, collection: str, query: Dict = None, limit: int = 10, sort: Dict = None) -> str:
        """Search documents in a collection with improved sorting and filtering."""
        try:
            if collection not in self.collections:
                return json.dumps({"error": f"Collection '{collection}' not found", 
                                  "available": list(self.collections.keys())})
            
            if query and not isinstance(query, dict):
                return json.dumps({"error": "Invalid query format"})
            
            # Determine sort order with smart defaults
            if not sort:
                sort_field = self.collections[collection]["sort_field"]
                sort = [(sort_field, -1)]  # Default to descending on the primary sort field
            
            cursor = self.collections[collection]["collection"].find(query or {})
            
            # Apply sort
            if sort:
                cursor = cursor.sort(sort)
                
            # Apply limit
            if limit:
                cursor = cursor.limit(limit)
                
            results = list(cursor)
            
            if not results:
                return json.dumps({"error": f"No documents found in collection '{collection}' matching query"})
                
            return self._format_results(results)
        except Exception as e:
            return json.dumps({"error": f"Error searching collection: {str(e)}"})

    def _cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0
        return dot_product / (norm_a * norm_b)

    @lru_cache(maxsize=50)
    def _get_embedding(self, text: str):
        """Get embedding for text with caching."""
        return self.embeddings.embed_query(text)

    def semantic_search(self, collection: str, search_text: str, limit: int = 5) -> str:
        """Perform semantic search across documents with improved chunking and scoring."""
        try:
            if collection not in self.collections:
                return json.dumps({"error": f"Collection '{collection}' not found", 
                                  "available": list(self.collections.keys())})
            
            # Find relevant fields based on semantic similarity to search query
            relevant_fields = self._find_relevant_fields(collection, search_text)
            
            # Get documents from collection
            documents = list(self.collections[collection]["collection"].find())
            
            if not documents:
                return json.dumps({"error": f"No documents found in collection '{collection}'"})
            
            # Initialize text splitter with adaptive chunk size
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", ", ", " ", ""]
            )
            
            # Process documents into chunks
            all_chunks = []
            chunk_metadata = []
            
            for doc in documents:
                # Flatten document to text with field weighting
                flattened_text = self._document_to_text(doc, relevant_fields)
                
                # Split document into chunks
                chunks = text_splitter.split_text(flattened_text)
                
                # Store chunks with metadata
                for chunk in chunks:
                    all_chunks.append(chunk)
                    chunk_metadata.append({"doc_id": doc["_id"]})
            
            if not all_chunks:
                return json.dumps({"error": f"No text content found for semantic search"})
            
            # Create embeddings and calculate similarity
            query_embedding = self._get_embedding(search_text)
            
            # Use batched embedding to avoid token limits
            batch_size = 100
            chunk_embeddings = []
            
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i:i+batch_size]
                batch_embeddings = self.embeddings.embed_documents(batch)
                chunk_embeddings.extend(batch_embeddings)
            
            # Calculate cosine similarity
            similarity_scores = [
                (i, self._cosine_similarity(query_embedding, embedding))
                for i, embedding in enumerate(chunk_embeddings)
            ]
            
            # Sort by similarity
            similarity_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Get unique documents from top chunks with improved selection
            top_matches = similarity_scores[:min(limit*3, len(similarity_scores))]
            unique_doc_ids = {chunk_metadata[idx]["doc_id"] for idx, _ in top_matches}
            
            # Calculate document scores and retrieve full documents
            results = []
            for doc_id in unique_doc_ids:
                doc = self.collections[collection]["collection"].find_one({"_id": doc_id})
                if doc:
                    # Calculate score as weighted average of its chunks' scores
                    doc_chunks = [i for i, meta in enumerate(chunk_metadata) 
                                 if meta["doc_id"] == doc_id and i < len(similarity_scores)]
                    
                    # Use exponential weighting to prioritize higher scores
                    scores = [similarity_scores[i][1] for i in doc_chunks]
                    weights = [2**s for s in scores]  # Exponential weighting
                    score = sum(s*w for s, w in zip(scores, weights)) / sum(weights) if weights else 0
                    
                    # Add document with score
                    doc_with_score = self._format_document(doc)
                    doc_with_score["_similarity_score"] = score
                    
                    # Add matching context (best matching chunks)
                    doc_chunk_scores = [(i, similarity_scores[i][1]) for i in doc_chunks]
                    doc_chunk_scores.sort(key=lambda x: x[1], reverse=True)
                    top_doc_chunks = [all_chunks[idx] for idx, _ in doc_chunk_scores[:2]]
                    
                    if top_doc_chunks:
                        doc_with_score["_matching_context"] = top_doc_chunks
                        
                    results.append(doc_with_score)
            
            # Sort results by score and limit
            results.sort(key=lambda x: x["_similarity_score"], reverse=True)
            results = results[:limit]
            
            if not results:
                return json.dumps({"error": f"No relevant documents found for '{search_text}'"})
                
            return json.dumps(results, indent=2, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": f"Error performing semantic search: {str(e)}"})
    
    def _find_relevant_fields(self, collection: str, search_text: str, top_n: int = 5) -> List[str]:
        """Find fields most relevant to the search query for better weighting."""
        if collection not in self.field_embeddings:
            return []
            
        field_data = self.field_embeddings[collection]
        query_embedding = self._get_embedding(search_text)
        
        similarities = [
            (field, self._cosine_similarity(query_embedding, embedding))
            for field, embedding in zip(field_data['fields'], field_data['embeddings'])
        ]
        
        # Sort by similarity and get top fields
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [field for field, score in similarities[:top_n] if score > 0.2]
    
    def _document_to_text(self, doc: Dict, relevant_fields: List[str] = None) -> str:
        """Convert document to searchable text format with field weighting."""
        parts = []
        
        # Emphasize relevant fields by repeating them
        emphasis_factor = 2 if relevant_fields else 1
        
        for key, value in doc.items():
            if key.startswith('_') and key != '_id':
                continue
                
            # Apply emphasis for relevant fields
            repetitions = emphasis_factor if relevant_fields and key in relevant_fields else 1
            
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    content = f"{key}.{subkey}: {subvalue}"
                    parts.extend([content] * repetitions)
            elif isinstance(value, list):
                if value and isinstance(value[0], dict):
                    for i, item in enumerate(value):
                        for subkey, subvalue in item.items():
                            content = f"{key}[{i}].{subkey}: {subvalue}"
                            parts.extend([content] * repetitions)
                else:
                    content = f"{key}: {', '.join(str(x) for x in value)}"
                    parts.extend([content] * repetitions)
            else:
                content = f"{key}: {value}"
                parts.extend([content] * repetitions)
                
        return "\n".join(parts)

    def get_recent_documents(self, collection: str, limit: int = 5) -> str:
        """Get most recent documents from a collection."""
        try:
            if collection not in self.collections:
                return json.dumps({"error": f"Collection '{collection}' not found", 
                                  "available": list(self.collections.keys())})
            
            sort_field = self.collections[collection]["sort_field"]
            results = list(self.collections[collection]["collection"]
                          .find().sort([(sort_field, -1)]).limit(limit))
            
            if not results:
                return json.dumps({"error": f"No documents found in collection '{collection}'"})
                
            return self._format_results(results)
        except Exception as e:
            return json.dumps({"error": f"Error getting recent documents: {str(e)}"})

    def natural_language_query(self, query_text: str, collection: str = None) -> str:
        """Process natural language queries against database with context awareness."""
        try:
            # Store query in history for context
            self.query_history.append(query_text)
            if len(self.query_history) > self.max_history:
                self.query_history.pop(0)
                
            # Detect collection from query if not specified
            if not collection:
                collection = self._infer_collection(query_text)
            
            # If collection found, search in that collection
            if collection:
                return self.semantic_search(collection, query_text)
            
            # Otherwise search across all collections
            all_results = []
            for coll in self.collections:
                results_json = self.semantic_search(coll, query_text, 3)
                try:
                    results = json.loads(results_json)
                    if not isinstance(results, dict) or "error" not in results:
                        for doc in results:
                            doc["_collection"] = coll
                        all_results.extend(results)
                except:
                    pass
            
            # Sort by similarity score
            all_results.sort(key=lambda x: x.get("_similarity_score", 0), reverse=True)
            
            if all_results:
                return json.dumps(all_results[:5], indent=2)
            else:
                return json.dumps({"error": "No relevant results found across collections"})
                
        except Exception as e:
            return json.dumps({"error": f"Error processing natural language query: {str(e)}"})
            
    def _infer_collection(self, query_text: str) -> Optional[str]:
        """Infer the collection name from the query with improved heuristics."""
        query_lower = query_text.lower()
        
        # First, check for explicit collection mentions
        collection_keywords = {}
        for coll in self.collections:
            # Add collection name and singularized version
            collection_keywords[coll] = coll
            if coll.endswith('s'):
                collection_keywords[coll[:-1]] = coll
        
        # Add additional domain-specific mappings
        common_mappings = {
            'hotel': 'hotels', 'room': 'hotels', 'accommodation': 'hotels',
            'attraction': 'attractions', 'place': 'attractions', 'location': 'attractions',
            'cruise': 'cruises', 'ship': 'cruises', 'voyage': 'cruises',
            'booking': 'bookings', 'reservation': 'bookings', 'appointment': 'bookings',
            'case': 'cases', 'ticket': 'cases', 'issue': 'cases', 'incident': 'cases',
            'contact': 'contacts', 'person': 'contacts', 'customer': 'contacts', 'client': 'contacts',
            'user': 'users', 'account': 'users', 'member': 'users',
            'deal': 'deals', 'opportunity': 'deals', 'sale': 'deals',
            'product': 'products', 'item': 'products', 'merchandise': 'products'
        }
        
        # Update with common mappings if collections exist
        for keyword, coll in common_mappings.items():
            if coll in self.collections:
                collection_keywords[keyword] = coll
        
        # Check for explicit collection mentions
        for keyword, coll in collection_keywords.items():
            if keyword in query_lower:
                return coll
                
        # Try to match field names
        query_embedding = self._get_embedding(query_text)
        best_match = None
        best_score = 0
        
        for coll, field_data in self.field_embeddings.items():
            for field, embedding in zip(field_data['fields'], field_data['embeddings']):
                score = self._cosine_similarity(query_embedding, embedding)
                if score > best_score:
                    best_score = score
                    best_match = coll
        
        # Return best match if score is above threshold
        if best_score > 0.3:
            return best_match
            
        return None

    def create_document(self, collection: str, document: Dict) -> str:
        """Create a new document in the specified collection."""
        try:
            if collection not in self.collections:
                return json.dumps({"error": f"Collection '{collection}' not found"})

            # Validate required fields
            required_fields = self.collections[collection]["required_fields"]
            missing_fields = [field for field in required_fields if field not in document]
            if missing_fields:
                return json.dumps({
                    "error": f"Missing required fields: {missing_fields}",
                    "required_fields": required_fields
                })

            # Add timestamps
            current_time = datetime.utcnow()
            document['createdAt'] = document.get('createdAt', current_time)
            document['updatedAt'] = document.get('updatedAt', current_time)

            # Insert document
            result = self.collections[collection]["collection"].insert_one(document)
            created_doc = self.collections[collection]["collection"].find_one({"_id": result.inserted_id})
            
            return self._format_results([created_doc])
        except Exception as e:
            return json.dumps({"error": f"Error creating document: {str(e)}"})

    def update_document(self, collection: str, query: Dict, updates: Dict) -> str:
        """Update an existing document."""
        try:
            if collection not in self.collections:
                return json.dumps({"error": f"Collection '{collection}' not found"})

            # Verify document exists
            mongo_collection = self.collections[collection]["collection"]
            existing_doc = mongo_collection.find_one(query)
            if not existing_doc:
                return json.dumps({"error": f"No document found matching query: {query}"})

            # Add updated timestamp
            if '$set' not in updates:
                updates['$set'] = {}
            updates['$set']['updatedAt'] = datetime.utcnow()

            # Perform update
            result = mongo_collection.update_one(query, updates)
            
            if result.modified_count > 0:
                updated_doc = mongo_collection.find_one(query)
                return self._format_results([updated_doc])
            else:
                return json.dumps({"error": "No changes made to document"})
        except Exception as e:
            return json.dumps({"error": f"Error updating document: {str(e)}"})

    def delete_document(self, collection: str, query: Dict) -> str:
        """Delete a document from a collection."""
        try:
            if collection not in self.collections:
                return json.dumps({"error": f"Collection '{collection}' not found"})

            # Verify document exists
            mongo_collection = self.collections[collection]["collection"]
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
            return json.dumps({"error": f"Error updating document: {str(e)}"})    

    def _create_tools(self) -> List[FunctionTool]:
        """Create tools for database interaction."""
        return [
            FunctionTool.from_defaults(fn=self.get_collection_schema, name="get_collection_schema",
                description="Get schema information about a collection. Args: collection (str)"),
            FunctionTool.from_defaults(fn=self.search_collection, name="search_collection",
                description="Search a collection with custom query. Args: collection (str), query (dict, optional), limit (int, optional)"),
            FunctionTool.from_defaults(fn=self.semantic_search, name="semantic_search",
                description="Perform semantic search on a collection. Args: collection (str), search_text (str), limit (int, optional)"),
            FunctionTool.from_defaults(fn=self.natural_language_query, name="natural_language_query",
                description="Process natural language queries. Args: query_text (str), collection (str, optional)"),
            FunctionTool.from_defaults(fn=self.get_recent_documents, name="get_recent_documents",
                description="Get recent documents from a collection. Args: collection (str), limit (int, optional)"),
            FunctionTool.from_defaults(fn=self.create_document, name="create_document",
                description="Create a new document. Args: collection (str), document (dict)"),
            FunctionTool.from_defaults(fn=self.update_document, name="update_document",
                description="Update an existing document. Args: collection (str), query (dict), updates (dict)"),
            FunctionTool.from_defaults(fn=self.delete_document, name="delete_document",
                description="Delete a document. Args: collection (str), query (dict)")
        ]

    def _create_agent(self) -> ReActAgent:
        """Create the ReAct agent with database context."""
        schema_info = json.dumps(self.schema_cache, indent=2, cls=MongoJSONEncoder)
        
        context = f"""You are an AI assistant managing a MongoDB database with full CRUD capabilities and semantic search.
        Available collections: {list(self.collections.keys())}
        
        IMPORTANT GUIDELINES:
        1. ALWAYS verify data exists before making statements or modifications
        2. For each collection, use the following information to guide your queries:
        {schema_info}
        
        3. When searching data:
           - Use search_collection for exact matches
           - Use semantic_search for natural language queries about a specific collection
           - Use natural_language_query for open-ended questions
           
        4. Always provide clear, accurate responses based on the actual data in the database
        5. When displaying results, format them in a user-friendly way
        """

        return ReActAgent.from_tools(
            self.tools,
            llm=self.llm,
            verbose=True,
            context=context
        )
            
    def chat(self):
        """Start an interactive chat session."""
        print(f"\nAI Database Assistant - Available collections: {list(self.collections.keys())}")
        
        while True:
            user_input = input("\nUser: ").strip()
            if user_input.lower() == 'exit':
                break
            
            try:
                # Detect if it's a direct query
                is_direct_query = any(user_input.lower().startswith(prefix) for prefix in 
                                     ["find", "search", "get", "show", "list", "what", "which", 
                                      "where", "who", "how many", "tell me about"])
                
                if is_direct_query and "schema" not in user_input.lower():
                    # Process as natural language query
                    result = self.natural_language_query(user_input)
                    try:
                        parsed_result = json.loads(result)
                        if isinstance(parsed_result, dict) and "error" in parsed_result:
                            # Fall back to agent for error cases
                            response = self.agent.chat(user_input)
                            print(f"\nAssistant: {response}")
                        else:
                            # Format and display results
                            if isinstance(parsed_result, list):
                                print(f"\nFound {len(parsed_result)} relevant results:")
                                for i, item in enumerate(parsed_result[:5], 1):
                                    collection = item.pop("_collection", "unknown") if "_collection" in item else "unknown"
                                    score = item.pop("_similarity_score", 0) if "_similarity_score" in item else 0
                                    print(f"\n--- Result {i} (Collection: {collection}, Relevance: {score:.2f}) ---")
                                    for key, value in item.items():
                                        if not key.startswith("_") and not isinstance(value, (dict, list)):
                                            print(f"{key}: {value}")
                            else:
                                print(f"\nResults: {json.dumps(parsed_result, indent=2)}")
                    except:
                        # Fall back to agent if result parsing fails
                        response = self.agent.chat(user_input)
                        print(f"\nAssistant: {response}")
                else:
                    # Use the agent for complex requests
                    response = self.agent.chat(user_input)
                    print(f"\nAssistant: {response}")
            except Exception as e:
                print(f"\nError: {str(e)}")
                print("Please try rephrasing your question.")

if __name__ == "__main__":
    chatbot = TenantDatabaseChatbot()
    chatbot.chat()