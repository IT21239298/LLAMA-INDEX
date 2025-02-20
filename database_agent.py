import os
import json
from typing import Dict, Any, List, Optional

from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime

from dotenv import load_dotenv

from base_agent import BaseAgent

class MongoJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for MongoDB specific types."""
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class DatabaseAgent(BaseAgent):
    """
    Advanced DatabaseAgent for sophisticated MongoDB interactions.
    """
    def __init__(self, mongo_uri: str = None):
        """
        Initialize DatabaseAgent with MongoDB connection.
        
        Args:
            mongo_uri: MongoDB connection string
        """
        super().__init__(name="DatabaseAgent", priority=1)
        
        # Load environment variables if no URI provided
        if not mongo_uri:
            load_dotenv()
            mongo_uri = os.getenv("MONGODB_URI")
        
        # Initialize MongoDB connection
        self.mongo_client = MongoClient(mongo_uri)
        self.db = self.mongo_client.tn_txbglk
        
        # Define collection-specific configurations
        self.collection_config = {
            'contacts': {
                'unique_fields': ['email', 'primaryContactNo', 'sequenceId'],
                'sort_field': 'sequenceIdReadOnly',
                'search_fields': [
                    'firstName', 'lastName', 'email', 
                    'primaryContactNo', 'country'
                ]
            },
            'customers': {
                'unique_fields': ['customerId', 'email'],
                'sort_field': 'createdAt',
                'search_fields': [
                    'firstName', 'lastName', 'email', 
                    'customerType'
                ]
            },
            'messages': {
                'unique_fields': ['messageId'],
                'sort_field': 'createdAt',
                'search_fields': [
                    'content', 'senderId', 
                    'receiverId', 'conversationId'
                ]
            },
            'conversations': {
                'unique_fields': ['conversationId'],
                'sort_field': 'createdAt',
                'search_fields': [
                    'participants', 'lastMessage', 
                    'status'
                ]
            },
            'cases': {
                'unique_fields': ['caseId'],
                'sort_field': 'createdAt',
                'search_fields': [
                    'description', 'status', 
                    'priority', 'assignedTo'
                ]
            }
        }
        
        # Get available collections
        self.collections = self._get_collections()
    
    def _get_collections(self) -> Dict[str, Any]:
        """
        Retrieve available collections in the database.
        
        Returns:
            Dictionary of available collections
        """
        collections = {}
        try:
            collection_names = self.db.list_collection_names()
            
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
        """
        Format a single document for output.
        
        Args:
            doc: MongoDB document to format
        
        Returns:
            Formatted document
        """
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
        """
        Format a list of documents for output.
        
        Args:
            results: List of MongoDB documents
        
        Returns:
            JSON-formatted string of results
        """
        formatted_results = [self._format_document(doc) for doc in results]
        return json.dumps(formatted_results, indent=2, ensure_ascii=False)
    
    def search_collection(self, collection: str, query: Dict = None, limit: int = 10) -> str:
        """
        Search documents in a specific collection.
        
        Args:
            collection: Target collection name
            query: Search query dictionary
            limit: Maximum number of results
        
        Returns:
            JSON-formatted search results
        """
        try:
            if collection not in self.collections:
                available = list(self.collections.keys())
                return json.dumps({
                    "error": f"Collection '{collection}' not found", 
                    "available": available
                })
            
            mongo_collection = self.collections[collection]["collection"]
            
            # Validate query
            if query and not isinstance(query, dict):
                return json.dumps({"error": "Invalid query format"})
            
            # Perform search
            results = list(mongo_collection.find(query or {}).limit(limit))
            
            if not results:
                return json.dumps({
                    "error": f"No documents found in collection '{collection}' matching query"
                })
                
            return self._format_results(results)
        except Exception as e:
            return json.dumps({"error": f"Error searching collection: {str(e)}"})
    
    def get_recent_documents(self, collection: str, limit: int = 5) -> str:
        """
        Get most recent documents from a collection.
        
        Args:
            collection: Target collection name
            limit: Number of recent documents to retrieve
        
        Returns:
            JSON-formatted recent documents
        """
        try:
            if collection not in self.collections:
                available = list(self.collections.keys())
                return json.dumps({
                    "error": f"Collection '{collection}' not found", 
                    "available": available
                })
            
            mongo_collection = self.collections[collection]["collection"]
            config = self.collection_config.get(collection, {})
            sort_field = config.get('sort_field', 'createdAt')
            
            results = list(mongo_collection.find().sort(sort_field, -1).limit(limit))
            
            if not results:
                return json.dumps({
                    "error": f"No documents found in collection '{collection}'"
                })
                
            return self._format_results(results)
        except Exception as e:
            return json.dumps({"error": f"Error getting recent documents: {str(e)}"})
    
    def _build_search_query(self, query: str, collection: str) -> Dict:
        """
        Build an intelligent search query for a specific collection.
        
        Args:
            query: User's search query
            collection: Target collection name
        
        Returns:
            MongoDB query dictionary
        """
        # Normalize query
        query = query.lower().strip()
        
        # Check if collection is configured
        if collection not in self.collection_config:
            return {}
        
        # Get searchable fields
        search_fields = self.collection_config[collection].get('search_fields', [])
        
        # Special handling for recent documents
        if query in ['recent', 'latest', '10 recent contacts', 'recent contacts']:
            return {}
        
        # Build OR conditions
        or_conditions = []
        
        # Special handling for contacts and customers
        if collection in ['contacts', 'customers']:
            name_parts = query.split()
            
            # Single term search
            if len(name_parts) == 1:
                or_conditions.extend([
                    {'firstName': {'$regex': f'^{query}', '$options': 'i'}},
                    {'lastName': {'$regex': f'^{query}', '$options': 'i'}},
                    {'email': {'$regex': query, '$options': 'i'}},
                    {'primaryContactNo': {'$regex': query, '$options': 'i'}}
                ])
            
            # Multi-part name search
            elif len(name_parts) > 1:
                or_conditions.extend([
                    {'$and': [
                        {'firstName': {'$regex': name_parts[0], '$options': 'i'}},
                        {'lastName': {'$regex': name_parts[-1], '$options': 'i'}}
                    ]},
                    {'$and': [
                        {'firstName': {'$regex': name_parts[-1], '$options': 'i'}},
                        {'lastName': {'$regex': name_parts[0], '$options': 'i'}}
                    ]}
                ])
        
        # Generic field searching for other collections
        for field in search_fields:
            or_conditions.append({field: {'$regex': query, '$options': 'i'}})
        
        return {'$or': or_conditions} if or_conditions else {}
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process query across multiple collections.
        
        Args:
            query: User's input query
        
        Returns:
            Dictionary with search results
        """
        try:
            # Normalize query
            normalized_query = query.lower().strip()
            
            # Priority collections for searching
            priority_collections = ['contacts', 'customers', 'conversations', 'messages', 'cases']
            
            # Aggregated results
            aggregated_results = {}
            
            # Check for recent documents request
            if normalized_query in ['recent contacts', '10 recent contacts', 'recent']:
                # Directly get recent contacts
                contacts_result = json.loads(self.get_recent_documents('contacts', limit=10))
                
                if not isinstance(contacts_result, dict) or 'error' not in contacts_result:
                    return {
                        'source': 'database',
                        'results': {'contacts': contacts_result},
                        'confidence': 1.0,
                        'query': query
                    }
            
            # Search across priority collections
            for collection_name in priority_collections:
                # Validate collection exists
                if collection_name not in self.collections:
                    continue
                
                # Get collection
                collection = self.collections[collection_name]['collection']
                
                # Build advanced query
                search_query = self._build_search_query(query, collection_name)
                
                # Retrieve results
                if not search_query:
                    # If no specific query, get recent documents
                    config = self.collection_config.get(collection_name, {})
                    sort_field = config.get('sort_field', 'createdAt')
                    collection_results = list(collection.find().sort(sort_field, -1).limit(5))
                else:
                    # Perform search with query
                    collection_results = list(collection.find(search_query).limit(5))
                
                # Format results
                formatted_results = [self._format_document(doc) for doc in collection_results]
                
                # Store results if found
                if formatted_results:
                    aggregated_results[collection_name] = formatted_results
            
            # Calculate confidence
            confidence = 1.0 if aggregated_results else 0.0
            
            return {
                'source': 'database',
                'results': aggregated_results,
                'confidence': confidence,
                'query': query,
                'error': None
            }
        
        except Exception as e:
            return {
                'source': 'database',
                'results': {},
                'confidence': 0.0,
                'query': query,
                'error': str(e)
            }