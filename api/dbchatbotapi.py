import sys
import os
import re
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dboperation import TenantDatabaseChatbot
from user_managment import mongo_to_json

# Flask API setup
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "default-secret-key-change-in-production")
CORS(app, supports_credentials=True)  # Enable CORS with credentials support

# Initialize the chatbot
chatbot = TenantDatabaseChatbot()

# In-memory user session cache
active_sessions = {}

# Helper function to extract user details from message text
def extract_user_details(message):
    """
    Try to extract name, email and contact from a message.
    Returns a dict with any found fields.
    """
    details = {}
    
    # Simple regex patterns for extraction
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    phone_pattern = r'(\+\d{1,3})?[\s.-]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}'
    
    # Extract email
    email_match = re.search(email_pattern, message)
    if email_match:
        details['email'] = email_match.group(0)
    
    # Extract phone number
    phone_match = re.search(phone_pattern, message)
    if phone_match:
        details['contact_number'] = phone_match.group(0)
    
    # Extract name (more complex, looking for potential name patterns)
    name_indicators = ['name is', 'I am', 'I\'m', 'call me', 'My name is']
    for indicator in name_indicators:
        if indicator.lower() in message.lower():
            parts = message.lower().split(indicator.lower(), 1)
            if len(parts) > 1:
                # Take what appears to be a name (first 30 chars max, stopping at punctuation)
                potential_name = parts[1].strip()
                name_end = min(30, len(potential_name))
                for i, char in enumerate(potential_name):
                    if i >= name_end:
                        break
                    if char in '.,:;!?' or (char == ',' and i > 2):
                        name_end = i
                        break
                
                details['name'] = potential_name[:name_end].strip()
                break
    
    return details

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    """API endpoint for chat interactions with improved session handling."""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Missing request body"}), 400
            
        # Get request IP and any identifying information to track the user
        client_ip = request.remote_addr
        user_agent = request.headers.get('User-Agent', '')
        client_id = f"{client_ip}_{user_agent}"
        
        # Extract message and any provided session token
        user_input = data.get('message', '')
        session_token = data.get('session_token')
        
        # Check if user has a session from previous interactions
        is_authenticated = False
        
        # First check if session token was provided
        if session_token:
            is_authenticated = chatbot.validate_session(session_token)
            if is_authenticated:
                active_sessions[client_id] = session_token
        
        # If no token provided or token invalid, check if we have one stored for this client
        if not is_authenticated and client_id in active_sessions:
            stored_token = active_sessions[client_id]
            is_authenticated = chatbot.validate_session(stored_token)
            if is_authenticated:
                session_token = stored_token
            else:
                # If stored token is invalid, remove it
                del active_sessions[client_id]
        
        # Extract user details from request body
        name = data.get('name')
        email = data.get('email')
        contact_number = data.get('contact_number')
        
        # If explicit user details aren't provided, try to extract from message
        if not is_authenticated and not all([name, email, contact_number]):
            extracted_details = extract_user_details(user_input)
            
            # Use extracted details to fill in missing fields
            if 'name' in extracted_details and not name:
                name = extracted_details['name']
            if 'email' in extracted_details and not email:
                email = extracted_details['email']
            if 'contact_number' in extracted_details and not contact_number:
                contact_number = extracted_details['contact_number']
        
        # If authentication information is still incomplete
        if not is_authenticated and not all([name, email, contact_number]):
            # Keep track of what we already have
            missing_fields = []
            if not name:
                missing_fields.append("name")
            if not email:
                missing_fields.append("email")
            if not contact_number:
                missing_fields.append("contact number")
            
            missing_str = ", ".join(missing_fields)
            
            # Respond with a request for the missing information
            welcome_message = "Welcome to our chatbot! "
            if user_input and not user_input.strip().lower() in ["hi", "hello", "hey"]:
                # If they sent a different message first, acknowledge it
                welcome_message += f"I'd be happy to help with '{user_input}', but first I need some information. "
            
            auth_message = (
                f"{welcome_message}Before we proceed, please provide your {missing_str}. "
                f"This helps us personalize your experience. "
                f"For example, you can say: 'My name is John Doe, my email is john@example.com, "
                f"and my contact number is +1234567890'."
            )
            
            return jsonify({
                "response": auth_message,
                "status": "success"
            })
        
        # Attempt authentication if we have all the details but aren't authenticated yet
        if not is_authenticated and all([name, email, contact_number]):
            success, result = chatbot.authenticate_user(name, email, contact_number)
            
            if not success:
                # Authentication failed, ask user to try again
                return jsonify({
                    "response": f"There was a problem with your details: {result}. Please try again with valid information.",
                    "status": "success"
                })
            
            # Authentication successful - IMPORTANT: Store the session token
            session_token = chatbot.session_token
            active_sessions[client_id] = session_token
            
            # If this was just a "hi" message with auth details, respond with a welcome
            if user_input.lower() in ['hi', 'hello', 'hey', 'hi there', 'hello there'] or not user_input.strip():
                return jsonify({
                    "response": f"Thank you {name}! Your information has been verified. How can I help you today?",
                    "status": "success",
                    "session_token": session_token
                })
        
        # At this point, user is authenticated. Process their actual message.
        if not user_input or user_input.strip() == "":
            return jsonify({
                "response": "Thank you for your information! How can I assist you today?",
                "status": "success",
                "session_token": session_token
            })
        
        # Process the message using your existing agent
        chatbot.data_cache = {}
        
        # Record this interaction
        if hasattr(chatbot, 'record_interaction'):
            chatbot.record_interaction(user_input)
        
        # Process the message
        response = chatbot.agent.chat(user_input)
        
        # Record the response
        if hasattr(chatbot, 'record_interaction'):
            chatbot.record_interaction(user_input, str(response))
        
        # Include session token in every successful response
        return jsonify({
            "response": str(response),
            "status": "success",
            "session_token": session_token
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "response": f"I encountered an error processing your request: {str(e)}. Please try again.",
            "status": "error"
        }), 500

if __name__ == "__main__":
    # Run the Flask app
    port = int(os.environ.get("PORT", 3004))
    app.run(host="0.0.0.0", port=port, debug=True)