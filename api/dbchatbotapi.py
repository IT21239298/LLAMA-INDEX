import sys
import os
import re
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from dotenv import load_dotenv
import logging
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dboperation import TenantDatabaseChatbot
from service.user_managment import mongo_to_json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('chatbot_api')

# Load environment variables
load_dotenv()

# Flask API setup
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "default-secret-key-change-in-production")
CORS(app, supports_credentials=True)  # Enable CORS with credentials support

# Initialize the chatbot
chatbot = TenantDatabaseChatbot()

# In-memory user session cache
active_sessions = {}

def extract_user_details_with_llm(message):
    """
    Use the LLM to extract name, email and contact information from a message.
    Returns a dict with the extracted fields.
    """
    try:
        # Skip extraction for simple greetings
        simple_greetings = ["hi", "hello", "hey", "hi there", "hello there"]
        if message.lower().strip() in simple_greetings:
            return {}

        # Create a prompt for the LLM to extract user information
        extraction_prompt = f"""
        Extract the user's name, email, and contact number from the following message if present.
        If any piece of information is missing, leave its field empty.
        Format your response as a valid JSON object with keys: "name", "email", "contact_number"

        Message: {message}

        Response format example:
        {{
            "name": "John Doe",
            "email": "john@example.com",
            "contact_number": "+1-234-567-8910"
        }}
        """

        # Use the chatbot's LLM to extract information
        extraction_response = chatbot.llm.complete(extraction_prompt)
        extraction_text = extraction_response.text.strip()

        # Try to parse the JSON response
        try:
            # Find and extract just the JSON part from the response
            json_match = re.search(r'\{[\s\S]*\}', extraction_text)
            if json_match:
                extraction_text = json_match.group(0)
            
            extracted_data = json.loads(extraction_text)
            
            # Validate and clean the extracted fields
            cleaned_data = {}
            if 'name' in extracted_data and extracted_data['name']:
                cleaned_data['name'] = extracted_data['name'].strip()
                
            if 'email' in extracted_data and extracted_data['email']:
                # Validate email format
                email = extracted_data['email'].strip()
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                if re.match(email_pattern, email):
                    cleaned_data['email'] = email
                    
            if 'contact_number' in extracted_data and extracted_data['contact_number']:
                # Clean and validate phone number
                contact = extracted_data['contact_number'].strip()
                # Keep only digits, plus sign, hyphens and parentheses
                contact = re.sub(r'[^\d\+\-\(\)]', '', contact)
                if len(re.sub(r'\D', '', contact)) >= 10:  # At least 10 digits
                    cleaned_data['contact_number'] = contact
                    
            logger.info(f"LLM extracted user details: {cleaned_data}")
            return cleaned_data
            
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM extraction response as JSON: {extraction_text}")
            # Fall back to regex-based extraction
            return extract_user_details_with_regex(message)
            
    except Exception as e:
        logger.error(f"Error in LLM extraction: {str(e)}", exc_info=True)
        # Fall back to regex-based extraction
        return extract_user_details_with_regex(message)

def extract_user_details_with_regex(message):
    """
    Use regex patterns to extract name, email and contact from a message.
    This serves as a fallback method when the LLM extraction fails.
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
    name_indicators = ['name is', 'I am', 'I\'m', 'call me', 'My name is', 'name']
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
    
    logger.info(f"Regex extracted user details: {details}")
    return details

def generate_authentication_success_message(name):
    """
    Generate an attractive authentication success message with emojis
    """
    try:
        prompt = f"""
        Generate an enthusiastic authentication success message for a user named {name}.
        The message should:
        1. Confirm their account has been successfully created
        2. Make them feel welcome to the service
        3. Ask how you can help them today
        4. Use emojis for an engaging feel
        5. Keep it concise (2-3 sentences)
        
        Just provide the message with no additional formatting or explanation.
        """
        
        response = chatbot.llm.complete(prompt)
        success_message = response.text.strip()
        
        # Remove any quotation marks around the message
        success_message = success_message.strip('"\'')
        
        return success_message
    except Exception as e:
        logger.error(f"Error generating auth success message: {str(e)}", exc_info=True)
        # Fallback message
        return f"Authentication successful, {name}! 🎉 Your account has been created successfully. How can I assist you today? Feel free to ask me anything!"

def generate_missing_info_message(missing_fields, user_input):
    """
    Use the LLM to generate a friendly message requesting missing user information.
    """
    try:
        missing_str = ", ".join(missing_fields)
        
        # Create a prompt for the LLM
        prompt = f"""
        Generate a friendly message asking the user to provide their {missing_str}.
        
        The user's original message was: "{user_input}"
        
        Your message should:
        1. Acknowledge their message if applicable
        2. Explain why we need this information in a friendly way (for personalization and account creation)
        3. Ask for specifically: {missing_str}
        4. Provide a clear example of how they could format their response
        5. Be warm and conversational
        6. Include appropriate emojis to make it engaging
        7. Be concise (2-3 sentences)
        
        Just provide the message to the user, with no additional formatting or explanation.
        """
        
        response = chatbot.llm.complete(prompt)
        message = response.text.strip()
        
        # Remove any quotation marks around the message
        message = message.strip('"\'')
        
        return message
    except Exception as e:
        logger.error(f"Error generating missing info message: {str(e)}", exc_info=True)
        # Fallback message
        welcome_message = "Thanks for reaching out! 👋 "
        return (
            f"{welcome_message}To create your account and provide personalized service, I just need your {missing_str}. "
            f"Please share it like this: 'My name is John, email: john@example.com, "
            f"number: +1234567890'. Then we can get started! ✨"
        )

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "ok", "service": "chatbot-api"})

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
        
        logger.info(f"Received chat request: client_id={client_id}, message_length={len(user_input)}")
        
        # Check if user has a session from previous interactions
        is_authenticated = False
        authenticated_user = None
        
        # First check if session token was provided
        if session_token:
            logger.debug(f"Validating provided session token")
            is_authenticated, authenticated_user = chatbot.validate_session(session_token)
            
            if is_authenticated:
                logger.info(f"Successfully authenticated with token")
                active_sessions[client_id] = session_token
        
        # If no token provided or token invalid, check if we have one stored for this client
        if not is_authenticated and client_id in active_sessions:
            stored_token = active_sessions[client_id]
            logger.debug(f"Trying stored session token")
            is_authenticated, authenticated_user = chatbot.validate_session(stored_token)
            
            if is_authenticated:
                logger.info(f"Successfully authenticated with stored token")
                session_token = stored_token
            else:
                # If stored token is invalid, remove it
                logger.debug(f"Stored token invalid, removing from active sessions")
                del active_sessions[client_id]
        
        # Extract user details from request body
        name = data.get('name')
        email = data.get('email')
        contact_number = data.get('contact_number')
        
        # If explicit user details aren't provided, try to extract from message
        if not is_authenticated and not all([name, email, contact_number]):
            # Use LLM to extract user details from message
            extracted_details = extract_user_details_with_llm(user_input)
            logger.debug(f"LLM-extracted user details: {extracted_details}")
            
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
            logger.info(f"Missing authentication details: {missing_str}")
            
            # Generate personalized message requesting the missing information
            auth_message = generate_missing_info_message(missing_fields, user_input)
            
            return jsonify({
                "response": auth_message,
                "status": "success"
            })
        
        # Attempt authentication if we have all the details but aren't authenticated yet
        if not is_authenticated and all([name, email, contact_number]):
            logger.info(f"Attempting to authenticate user: {name}, {email}")
            success, result = chatbot.authenticate_user(name, email, contact_number)
            
            if not success:
                # Authentication failed, ask user to try again
                logger.warning(f"Authentication failed: {result}")
                return jsonify({
                    "response": f"There was a problem with your details: {result}. Please try again with valid information.",
                    "status": "error"
                })
            
            # Authentication successful - Store the session token
            session_token = chatbot.session_token
            active_sessions[client_id] = session_token
            logger.info(f"New user authenticated and session created: {name}")
            
            # Generate an enthusiastic success message
            success_message = generate_authentication_success_message(name)
            
            return jsonify({
                "response": success_message,
                "status": "success",
                "session_token": session_token
            })
        
        # At this point, user is authenticated. Process their actual message.
        # Just pass the message to TenantDatabaseChatbot agent
        logger.info(f"Processing message for authenticated user: {user_input[:30]}...")
        
        # Record this interaction
        if hasattr(chatbot, 'record_interaction'):
            chatbot.record_interaction(user_input)
        
        # Process the message using the agent
        try:
            # Clear any cached data
            if hasattr(chatbot, 'data_cache'):
                chatbot.data_cache = {}
                
            # Process the message with the agent
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
        except Exception as chat_error:
            logger.error(f"Error processing chat with agent: {str(chat_error)}", exc_info=True)
            # Fallback to direct LLM response if agent fails
            try:
                fallback_prompt = f"""
                The user asked: "{user_input}"
                
                Provide a helpful and friendly response. If this seems like a data-related question,
                politely explain that you're currently having trouble accessing the database
                and ask if there's something else you can help with.
                
                Include appropriate emojis and keep it concise and friendly.
                """
                fallback_response = chatbot.llm.complete(fallback_prompt).text.strip()
                
                return jsonify({
                    "response": fallback_response,
                    "status": "partial_success",
                    "session_token": session_token
                })
            except:
                raise  # Re-raise if even the fallback fails
        
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        return jsonify({
            "error": str(e),
            "response": "I'm sorry, I'm having trouble processing your request right now. 😊 Please try again in a moment or rephrase your question.",
            "status": "error"
        }), 500

if __name__ == "__main__":
    # Run the Flask app
    port = int(os.environ.get("PORT", 3004))
    logger.info(f"Starting chatbot API on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)