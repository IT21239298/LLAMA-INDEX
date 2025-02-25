import sys
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dboperation import TenantDatabaseChatbot  
# Flask API setup
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the chatbot
chatbot = TenantDatabaseChatbot()

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    """API endpoint for chat interactions."""
    try:
        data = request.json
        if not data or 'message' not in data:
            return jsonify({"error": "Missing 'message' in request body"}), 400
        
        user_input = data['message']
        
        # Process the message using your existing agent
        chatbot.data_cache = {}
        response = chatbot.agent.chat(user_input)
        
        return jsonify({
            "response": str(response),
            "status": "success"
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

if __name__ == "__main__":
    # Run the Flask app
    port = int(os.environ.get("PORT", 3004))
    app.run(host="0.0.0.0", port=port, debug=True)