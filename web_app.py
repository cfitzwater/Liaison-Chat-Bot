import os
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from dotenv import load_dotenv

# Import the new search function from your app.py
from app import search_documents_web

load_dotenv()
app = Flask(__name__)

# --- Configuration ---
QA_FILE = "qa_history.json"
LIBRARY_DIR = "library"

# Ensure the clinical library folder exists
if not os.path.exists(LIBRARY_DIR):
    os.makedirs(LIBRARY_DIR)

def save_qa_history(question, answer):
    """Stores every interaction in a structured JSON file for your submission."""
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "answer": answer
    }
    
    history = []
    if os.path.exists(QA_FILE):
        try:
            with open(QA_FILE, "r") as f:
                history = json.load(f)
        except:
            history = []
            
    history.append(entry)
    
    with open(QA_FILE, "w") as f:
        json.dump(history, f, indent=4)

# --- Routes ---

@app.route('/')
def index():
    """Requirement: Serves the main Intermountain-themed dashboard."""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handles the user prompt and returns the AI's 'Thinking' result."""
    user_message = request.json.get("message")
    
    if not user_message:
        return jsonify({"answer": "Please enter a question."}), 400

    # Call the RAG logic in app.py
    try:
        ai_response = search_documents_web(user_message)
        
        # Log the Q&A to your history file
        save_qa_history(user_message, ai_response)
        
        return jsonify({"answer": ai_response})
    except Exception as e:
        return jsonify({"answer": f"System Error: {str(e)}"}), 500

@app.route('/api/files')
def list_files():
    """Requirement: Scans the library folder to populate the sidebar."""
    try:
        files = [f for f in os.listdir(LIBRARY_DIR) if f.endswith('.pdf')]
        return jsonify({"files": files})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/library/<filename>')
def get_pdf(filename):
    """Requirement: Allows you to open clinical PDFs directly from the web app."""
    return send_from_directory(LIBRARY_DIR, filename)

if __name__ == '__main__':
    """Requirement: Runs the Flask development server."""
    print("Liaison Bot Web Server Starting...")
    print(f"Access the bot at: http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
