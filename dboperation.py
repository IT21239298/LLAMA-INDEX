import os
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from bson import ObjectId
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from typing import Dict, Any, List
import json
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from user_managment import UserManager, mongo_to_json

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
        self._initialize_collections()
        
        # Create tools and agent
        self.tools = self._create_tools()
        self.agent = self._create_agent()

        # Initialize user management
        self.user_manager = UserManager(mongo_uri, "chatbotSithum")
    
        # Track current user in session
        self.current_user = None
        self.session_token = None

    def _initialize_collections(self):
        """Analyze collections and their schemas efficiently."""
        collection_names = self.db.list_collection_names()
        
        for name in collection_names:
            collection = self.db[name]
            sample_docs = list(collection.find().limit(10))
            
            if not sample_docs:
                continue
                
            # Create schema from samples
            schema = self._analyze_schema(sample_docs)
            
            # Extract key metadata
            unique_fields = self._identify_unique_fields(collection, sample_docs)
            required_fields = self._identify_required_fields(sample_docs)
            sort_field = next((field for field in ['createdAt', 'updatedAt', 'date', 'timestamp'] 
                              if any(field in doc for doc in sample_docs)), '_id')
            
            # Store collection metadata
            self.collections[name] = {
                "collection": collection,
                "schema": schema,
                "unique_fields": unique_fields,
                "required_fields": required_fields,
                "sort_field": sort_field,
                "sample_count": len(sample_docs)
            }
            
            # Update schema cache for query guidance
            self.schema_cache[name] = {
                "fields": list(schema.keys()),
                "unique_fields": unique_fields,
                "required_fields": required_fields
            }
    
    def _analyze_schema(self, documents: List[Dict]) -> Dict:
        """Extract schema from documents efficiently."""
        if not documents:
            return {}
            
        schema = {}
        doc_count = len(documents)
        
        # Combine field analysis from all documents
        for doc in documents:
            for key, value in doc.items():
                if key not in schema:
                    schema[key] = {
                        "type": type(value).__name__,
                        "sample": self._format_sample(value),
                        "null_percentage": 0,
                        "count": 0
                    }
                
                schema[key]["count"] += 1
                if value is None:
                    schema[key]["null_percentage"] += 1
        
        # Calculate null percentages
        for field in schema:
            if schema[field]["count"] > 0:
                schema[field]["null_percentage"] = 100 - (schema[field]["count"] / doc_count * 100)
                
        return schema
    
    def _format_sample(self, value):
        """Format sample value for schema."""
        if isinstance(value, (ObjectId, datetime)):
            return str(value)
        return value
    
    def _identify_unique_fields(self, collection, documents: List[Dict]) -> List[str]:
        """Identify potential unique fields."""
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
                        field == 'email' or field == 'username'):
                        unique_fields.append(field)
            
        return unique_fields
    
    def _identify_required_fields(self, documents: List[Dict]) -> List[str]:
        """Identify likely required fields."""
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
        for field in all_fields:
            if (field.endswith('Id') or field.endswith('ID') or 
                'name' in field.lower() or 'email' in field.lower() or
                field in ['firstName', 'lastName', 'status', 'priority']):
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
                "sort_field": self.collections[collection]["sort_field"],
                "sample_count": self.collections[collection]["sample_count"]
            }, cls=MongoJSONEncoder)
        except Exception as e:
            return json.dumps({"error": f"Error getting schema: {str(e)}"})
    
    def search_collection(self, collection: str, query: Dict = None, limit: int = 10) -> str:
        """Search documents in a collection."""
        try:
            # Check authentication
            self.require_authentication()

            if collection not in self.collections:
                return json.dumps({"error": f"Collection '{collection}' not found", 
                                  "available": list(self.collections.keys())})
            
            if query and not isinstance(query, dict):
                return json.dumps({"error": "Invalid query format"})
            
            results = list(self.collections[collection]["collection"].find(query or {}).limit(limit))
            
            if not results:
                return json.dumps({"error": f"No documents found in collection '{collection}' matching query"})
                
            return self._format_results(results)
        except Exception as e:
            return json.dumps({"error": f"Error searching collection: {str(e)}"})

    def semantic_search(self, collection: str, search_text: str, limit: int = 5) -> str:
        """
        Perform semantic search with LLM-powered result interpretation.
        
        Args:
            collection (str): Collection to search
            search_text (str): Search query
            limit (int, optional): Maximum number of results. Defaults to 5.
        
        Returns:
            str: JSON-formatted, LLM-processed search results
        """
        try:
            # Validate collection
            if collection not in self.collections:
                return json.dumps({
                    "error": f"Collection '{collection}' not found", 
                    "available": list(self.collections.keys())
                })
            
            # Get documents from collection
            documents = list(self.collections[collection]["collection"].find())
            
            if not documents:
                return json.dumps({
                    "error": f"No documents found in collection '{collection}'"
                })
            
            # Use advanced text splitter for semantic chunking
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            
            # Process documents into chunks
            all_chunks = []
            chunk_metadata = []
            
            for doc in documents:
                # Flatten document to text
                flattened_text = self._document_to_text(doc)
                
                # Split document into chunks
                chunks = text_splitter.split_text(flattened_text)
                
                # Store chunks with metadata
                for chunk in chunks:
                    all_chunks.append(chunk)
                    chunk_metadata.append({"doc_id": doc["_id"], "doc": doc})
            
            if not all_chunks:
                return json.dumps({
                    "error": f"No text content found for semantic search"
                })
            
            # Create embeddings and calculate similarity
            query_embedding = self.embeddings.embed_query(search_text)
            chunk_embeddings = self.embeddings.embed_documents(all_chunks)
            
            # Calculate cosine similarity
            similarity_scores = [
                (i, sum(a*b for a, b in zip(query_embedding, embedding)))
                for i, embedding in enumerate(chunk_embeddings)
            ]
            
            # Sort by similarity
            similarity_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Get unique documents from top chunks
            top_matches = similarity_scores[:min(limit*2, len(similarity_scores))]
            unique_docs = {}
            
            for idx, score in top_matches:
                doc_id = chunk_metadata[idx]["doc_id"]
                if doc_id not in unique_docs:
                    unique_docs[doc_id] = {
                        "doc": chunk_metadata[idx]["doc"],
                        "scores": [],
                        "chunks": []
                    }
                unique_docs[doc_id]["scores"].append(score)
                unique_docs[doc_id]["chunks"].append(all_chunks[idx])
            
            # Prepare results for LLM processing
            processed_results = []
            for doc_id, doc_info in unique_docs.items():
                # Calculate document score
                doc_score = max(doc_info["scores"])
                
                # Prepare document for processing
                formatted_doc = self._format_document(doc_info["doc"])
                formatted_doc["_similarity_score"] = doc_score
                formatted_doc["_matching_context"] = doc_info["chunks"][:2]
                
                processed_results.append(formatted_doc)
            
            # Sort results by score
            processed_results.sort(key=lambda x: x["_similarity_score"], reverse=True)
            processed_results = processed_results[:limit]
            
            # Prepare LLM context
            llm_context = {
                "query": search_text,
                "collection": collection,
                "results": processed_results
            }
            
            # Use LLM to generate user-friendly interpretation
            try:
                # Prepare LLM prompt
                llm_prompt = f"""You are an AI assistant interpreting search results for the query: "{search_text}"

                Context:
                - Collection searched: {collection}
                - Number of results: {len(processed_results)}

                Raw Results:
                {json.dumps(processed_results, indent=2)}

                Task:
                1. Analyze the search results carefully
                2. Generate a concise, informative, and engaging summary
                3. Highlight the most relevant information
                4. Use a friendly, conversational tone
                5. Provide context and insights
                6. If results are limited, suggest alternatives or provide helpful guidance

                Output Requirements:
                - Start with a brief introduction explaining the search context
                - Summarize key findings
                - Provide brief details about top results
                - End with suggestions or next steps
                """
                # Use the LLM to process results
                response_tool = FunctionTool.from_defaults(
                    fn=lambda: llm_prompt,
                    name="process_search_results",
                    description="Generate user-friendly interpretation of search results"
                )

                # Create a temporary agent for result processing
                result_agent = ReActAgent.from_tools(
                    [response_tool],
                    llm=self.llm,
                    verbose=False,
                    context=llm_prompt
                )

                # Generate response
                llm_response = result_agent.chat(f"Interpret search results for query: {search_text}")
                
                # Prepare final output
                final_output = {
                    "original_query": search_text,
                    "collection": collection,
                    "llm_interpretation": llm_response,
                    "raw_results": processed_results
                }
                
                return json.dumps(final_output, indent=2, ensure_ascii=False)
            
            except Exception as llm_error:
                # Fallback if LLM processing fails
                return json.dumps({
                    "original_query": search_text,
                    "collection": collection,
                    "raw_results": processed_results,
                    "error": f"LLM processing failed: {str(llm_error)}"
                }, indent=2)
        
        except Exception as e:
            return json.dumps({
                "error": f"Error performing semantic search: {str(e)}"
            }, indent=2)
    
    def _document_to_text(self, doc: Dict) -> str:
        """Convert document to searchable text format."""
        parts = []
        for key, value in doc.items():
            if key.startswith('_') and key != '_id':
                continue
                
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    parts.append(f"{key}.{subkey}: {subvalue}")
            elif isinstance(value, list):
                if value and isinstance(value[0], dict):
                    for i, item in enumerate(value):
                        for subkey, subvalue in item.items():
                            parts.append(f"{key}[{i}].{subkey}: {subvalue}")
                else:
                    parts.append(f"{key}: {', '.join(str(x) for x in value)}")
            else:
                parts.append(f"{key}: {value}")
                
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
        """Process natural language queries against database."""
        try:
            # Detect collection from query if not specified
            if not collection:
                collection_keywords = {
                    'hotel': 'hotels', 'room': 'hotels', 'attraction': 'attractions',
                    'cruise': 'cruises', 'booking': 'bookings', 'case': 'cases',
                    'contact': 'contacts', 'user': 'users', 'deal': 'deals'
                }
                
                query_lower = query_text.lower()
                for keyword, coll in collection_keywords.items():
                    if keyword in query_lower and coll in self.collections:
                        collection = coll
                        break
            
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
            document['createdAt'] = document['updatedAt'] = datetime.utcnow()

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
            updates['$set'] = updates.get('$set', {})
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
            return json.dumps({"error": f"Error deleting document: {str(e)}"})

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
           - Use semantic_search for natural language queries about a specific collection also need to update results using natural_language_querty
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
    
    # user authentication and related functionalities
    def authenticate_user(self, name, email, contact_number):
        """
        Authenticate a user by creating or retrieving their profile.
        
        Args:
            name (str): User's name
            email (str): User's email
            contact_number (str): User's contact number
            
        Returns:
            tuple: (success, message)
        """
        # Check if user exists
        user = self.user_manager.get_user(email, "email")
        
        # Create user if doesn't exist
        if not user:
            success, result = self.user_manager.create_user(
                name=name,
                email=email,
                contact_number=contact_number
            )
            
            if not success:
                return False, f"Failed to create user: {result}"
                
            user_id = result
        else:
            user_id = str(user["_id"])
            
            # Update user info if needed
            updates = {}
            if user.get("name") != name:
                updates["name"] = name
            if user.get("contact_number") != contact_number:
                updates["contact_number"] = contact_number
                
            if updates:
                self.user_manager.update_user(user_id, updates)
        
        # Create session
        success, token = self.user_manager.create_session(user_id)
        if not success:
            return False, f"Failed to create session: {token}"
            
        # Store current user
        is_valid, user = self.user_manager.validate_session(token)
        if not is_valid:
            return False, "Failed to validate new session"
            
        self.current_user = user
        self.session_token = token
        
        return True, "Authentication successful"

    def is_authenticated(self):
        """Check if a user is authenticated in the current session."""
        return self.current_user is not None

    def require_authentication(self):
        """Check authentication and raise exception if not authenticated."""
        if not self.is_authenticated():
            raise Exception("Authentication required. Please provide your name, email, and contact number.")

    def validate_session(self, token):
        """Validate a session token and set current user."""
        is_valid, user = self.user_manager.validate_session(token)
        if is_valid:
            self.current_user = user
            self.session_token = token
        return is_valid

    def record_interaction(self, query, response=None, metadata=None):
        """Record a user interaction with the system."""
        if self.current_user:
            return self.user_manager.record_interaction(
                str(self.current_user["_id"]),
                query,
                response,
                metadata
            )
        return False
            
    def chat(self):
        """Start an interactive chat session with authentication."""
        print(f"\nAI Database Assistant - Please authenticate to continue")
    
        # Authentication flow
        authenticated = False
        while not authenticated:
            print("\nPlease provide your information to continue:")
            name = input("Name: ").strip()
            email = input("Email: ").strip()
            contact = input("Contact Number: ").strip()
            
            if not name or not email or not contact:
                print("All fields are required. Please try again.")
                continue
            
            success, message = self.authenticate_user(name, email, contact)
            if success:
                authenticated = True
                print(f"\nAuthentication successful. Welcome, {name}!")
            else:
                print(f"\nAuthentication failed: {message}")
        
        print(f"\nAvailable collections: {list(self.collections.keys())}")
        
        while True:
            user_input = input("\nUser: ").strip()
            if user_input.lower() == 'exit':
                break
            
            try:
                 # Record the interaction 
                self.record_interaction(user_input)
            
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