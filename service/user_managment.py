import os
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId
import json
import re
from datetime import datetime
import hashlib
import uuid

class UserManager:
    """Class to handle user management operations with MongoDB."""
    
    def __init__(self, mongo_uri=None, db_name=None):
        """Initialize the UserManager with MongoDB connection."""
        load_dotenv()
        
        # Use provided credentials or fetch from environment
        self.mongo_uri = mongo_uri or os.getenv("MONGODB_URI")
        self.db_name = db_name or os.getenv("DB_NAME", "chatbotSithum")
        
        # Connect to MongoDB
        self.client = MongoClient(self.mongo_uri)
        self.db = self.client[self.db_name]
        
        # User collection
        self.users_collection = self.db.users
        self.sessions_collection = self.db.user_sessions
        
        # Ensure indexes for performance and constraints
        self._ensure_indexes()
    
    def _ensure_indexes(self):
        """Create necessary indexes for the users collection."""
        # Create unique index for email
        self.users_collection.create_index("email", unique=True)
        
        # Create index for user lookup by contact number
        self.users_collection.create_index("contact_number")
        
        # Index for sessions by user_id and token
        self.sessions_collection.create_index("user_id")
        self.sessions_collection.create_index("token", unique=True)
    
    def validate_email(self, email):
        """Validate email format."""
        email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        return bool(re.match(email_pattern, email))
    
    def validate_contact_number(self, contact_number):
        """Basic validation for contact number."""
        # Remove all non-digit characters for validation
        digits_only = re.sub(r'\D', '', contact_number)
        # Check if we have at least 10 digits
        return len(digits_only) >= 10
    
    def create_user(self, name, email, contact_number, additional_info=None):
        """
        Create a new user in the database.
        
        Args:
            name (str): User's full name
            email (str): User's email address
            contact_number (str): User's contact number
            additional_info (dict, optional): Any additional user information
            
        Returns:
            tuple: (success, result) where success is a boolean and result is user_id or error message
        """
        # Validate inputs
        if not name or not email or not contact_number:
            return False, "Name, email, and contact number are required"
        
        if not self.validate_email(email):
            return False, "Invalid email format"
            
        if not self.validate_contact_number(contact_number):
            return False, "Invalid contact number format"
        
        # Check if user already exists
        existing_user = self.users_collection.find_one({"email": email})
        if existing_user:
            return False, "User with this email already exists"
        
        # Prepare user document
        user_doc = {
            "name": name,
            "email": email,
            "contact_number": contact_number,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "last_active": datetime.utcnow()
        }
        
        # Add any additional info
        if additional_info and isinstance(additional_info, dict):
            for key, value in additional_info.items():
                if key not in user_doc:
                    user_doc[key] = value
        
        # Insert into database
        try:
            result = self.users_collection.insert_one(user_doc)
            return True, str(result.inserted_id)
        except Exception as e:
            return False, f"Error creating user: {str(e)}"
    
    def get_user(self, identifier, id_type="email"):
        """
        Get user information from the database.
        
        Args:
            identifier (str): User identifier (email, _id, or contact_number)
            id_type (str): Type of identifier ('email', 'id', or 'contact')
            
        Returns:
            dict or None: User document if found, None otherwise
        """
        query = {}
        
        if id_type == "email":
            query = {"email": identifier}
        elif id_type == "id":
            try:
                query = {"_id": ObjectId(identifier)}
            except:
                return None
        elif id_type == "contact":
            query = {"contact_number": identifier}
        else:
            return None
        
        # Get user and update last active timestamp
        user = self.users_collection.find_one_and_update(
            query,
            {"$set": {"last_active": datetime.utcnow()}},
            return_document=True
        )
        
        return user
    
    def update_user(self, user_id, updates):
        """
        Update user information.
        
        Args:
            user_id (str): User's ObjectId as string
            updates (dict): Fields to update
            
        Returns:
            tuple: (success, result) where success is a boolean and result is updated user or error message
        """
        try:
            # Don't allow updating restricted fields
            restricted_fields = ["_id", "email", "created_at"]
            for field in restricted_fields:
                if field in updates:
                    del updates[field]
            
            # Add updated_at timestamp
            updates["updated_at"] = datetime.utcnow()
            
            # Convert string ID to ObjectId
            object_id = ObjectId(user_id)
            
            # Update the user
            result = self.users_collection.find_one_and_update(
                {"_id": object_id},
                {"$set": updates},
                return_document=True
            )
            
            if not result:
                return False, "User not found"
                
            return True, result
        except Exception as e:
            return False, f"Error updating user: {str(e)}"
    
    def create_session(self, user_id):
        """
        Create a new session for a user.
        
        Args:
            user_id (str): User's ObjectId as string
            
        Returns:
            tuple: (success, session_token or error message)
        """
        try:
            # Generate a unique session token
            token = str(uuid.uuid4())
            
            # Create session document
            session = {
                "user_id": ObjectId(user_id),
                "token": token,
                "created_at": datetime.utcnow(),
                "expires_at": datetime.utcnow().replace(hour=23, minute=59, second=59),  # End of day
                "is_active": True
            }
            
            # Insert session
            self.sessions_collection.insert_one(session)
            return True, token
        except Exception as e:
            return False, f"Error creating session: {str(e)}"
    
    def validate_session(self, token):
        """
        Validate a session token.
        
        Args:
            token (str): Session token
            
        Returns:
            tuple: (is_valid, user_info or error message)
        """
        try:
            # Find the session
            session = self.sessions_collection.find_one({
                "token": token,
                "is_active": True,
                "expires_at": {"$gt": datetime.utcnow()}
            })
            
            if not session:
                return False, "Invalid or expired session"
            
            # Get the user
            user = self.users_collection.find_one({"_id": session["user_id"]})
            if not user:
                return False, "User not found"
            
            # Update last active time
            self.users_collection.update_one(
                {"_id": user["_id"]},
                {"$set": {"last_active": datetime.utcnow()}}
            )
            
            return True, user
        except Exception as e:
            return False, f"Error validating session: {str(e)}"
    
    def end_session(self, token):
        """
        End a user session.
        
        Args:
            token (str): Session token
            
        Returns:
            bool: True if session was ended, False otherwise
        """
        try:
            result = self.sessions_collection.update_one(
                {"token": token},
                {"$set": {"is_active": False}}
            )
            return result.modified_count > 0
        except Exception:
            return False
    
    def record_interaction(self, user_id, query, response=None, metadata=None):
        """
        Record a user interaction with the chatbot.
        
        Args:
            user_id (str): User's ObjectId as string
            query (str): User's query
            response (str, optional): System's response
            metadata (dict, optional): Additional metadata
            
        Returns:
            bool: Success status
        """
        try:
            interaction = {
                "user_id": ObjectId(user_id),
                "query": query,
                "timestamp": datetime.utcnow(),
                "response": response
            }
            
            # Add metadata if provided
            if metadata and isinstance(metadata, dict):
                interaction["metadata"] = metadata
            
            # Insert interaction
            self.db.user_interactions.insert_one(interaction)
            return True
        except Exception:
            return False
    
    def get_user_interactions(self, user_id, limit=10):
        """
        Get recent interactions for a user.
        
        Args:
            user_id (str): User's ObjectId as string
            limit (int): Maximum number of interactions to return
            
        Returns:
            list: List of interaction documents
        """
        try:
            interactions = list(self.db.user_interactions.find(
                {"user_id": ObjectId(user_id)}
            ).sort("timestamp", -1).limit(limit))
            
            return interactions
        except Exception:
            return []

# Helper function to convert MongoDB documents to JSON
def mongo_to_json(doc):
    """Convert MongoDB document to JSON-serializable dict."""
    if doc is None:
        return None
        
    if isinstance(doc, list):
        return [mongo_to_json(item) for item in doc]
        
    if isinstance(doc, dict):
        result = {}
        for key, value in doc.items():
            if key == '_id' and isinstance(value, ObjectId):
                result[key] = str(value)
            elif isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, ObjectId):
                result[key] = str(value)
            elif isinstance(value, (dict, list)):
                result[key] = mongo_to_json(value)
            else:
                result[key] = value
        return result
        
    return doc

# Example usage
if __name__ == "__main__":
    # Initialize user manager
    user_manager = UserManager()
    
    # Example: Create a user
    success, result = user_manager.create_user(
        name="John Doe",
        email="john@example.com",
        contact_number="+1234567890",
        additional_info={"organization": "Example Corp"}
    )
    
    if success:
        print(f"User created with ID: {result}")
        
        # Create a session
        success, token = user_manager.create_session(result)
        if success:
            print(f"Session created with token: {token}")
            
            # Validate session
            is_valid, user = user_manager.validate_session(token)
            if is_valid:
                print(f"Session valid for user: {user['name']}")
                print(json.dumps(mongo_to_json(user), indent=2))
            else:
                print(f"Session validation failed: {user}")
    else:
        print(f"Failed to create user: {result}")