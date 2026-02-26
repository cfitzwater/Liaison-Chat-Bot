from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
import os
import json
import uuid

# The bridge to your RAG logic and database in app.py
from app import search_documents_web, collection, model 

app = Flask(__name__)

# File paths and directories
LIBRARY_DIR = "library"
DATA_FILE = "library_data.json"

@app.route('/')
def index():
    """Main Chatbot Interface (The primary tab)"""
    return render_template('index.html')

@app.route('/add', methods=['GET'])
def add_form():
    """Displays the HTML form in a new tab for manual entry"""
    return render_template('add_item.html')

@app.route('/save_item', methods=['POST'])
def save_item():
    """
    1. Extracts Name, Email, and Notes from the form.
    2. Structures it as a dictionary and saves to JSON.
    3. Indexes the data in ChromaDB.
    4. Closes the tab automatically to return to the chat.
    """
    # 1. Extract data from the POST request
    user_name = request.form.get('user_name')
    user_email = request.form.get('user_email')
    notes_content = request.form.get('notes')

    if not all([user_name, user_email, notes_content]):
        return "Error: All fields are required.", 400

    # 2. Structure the data as a dictionary (Project Requirement)
    new_entry = {
        "id": str(uuid.uuid4()),
        "inputted_by": user_name,
        "email": user_email,
        "notes": notes_content,
        "timestamp": str(uuid.uuid1())
    }

    # 3. Save to the JSON data file (Persistence Requirement)
    data = []
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    
    data.append(new_entry)
    
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=4)

    # 4. Immediately index in ChromaDB so it is searchable
    embedding = model.encode([notes_content]).tolist()
    collection.add(
        ids=[new_entry["id"]],
        embeddings=embedding,
        documents=[notes_content],
        metadatas=[{"source": f"Manual Note by {user_name}", "page": "N/A"}]
    )

    # 5. The "Self-Destruct" response: 
    # This closes the new tab and leaves you looking at your original chat.
    return '<script type="text/javascript">window.close();</script>'

@app.route('/chat', methods=['POST'])
def chat():
    """Handles the chatbot messaging logic"""
    user_message = request.json.get('message')
    response = search_documents_web(user_message)
    return jsonify({"answer": response})

@app.route('/api/files')
def list_files():
    """Populates the sidebar with clinical PDFs"""
    if not os.path.exists(LIBRARY_DIR):
        return jsonify({"files": []})
    files = [f for f in os.listdir(LIBRARY_DIR) if f.endswith('.pdf')]
    return jsonify({"files": files})

@app.route('/library/<filename>')
def get_pdf(filename):
    """Serves the PDF files for viewing"""
    return send_from_directory(LIBRARY_DIR, filename)

if __name__ == '__main__':
    # Running on local development server
    app.run(debug=True, port=5000)