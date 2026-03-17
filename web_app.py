from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory, session
import os
import json
import uuid
from datetime import datetime

# Bridge to app.py
try:
    from app import search_documents_web, collection, model 
except ImportError:
    print("Warning: app.py not found or ChromaDB not initialized.")

app = Flask(__name__)
app.secret_key = 'liaison_secret_key_123'

# File paths
USER_FILE = "users.json"
CHAT_FILE = "chat_history.json"
DATA_FILE = "library_data.json"
LIBRARY_DIR = "library"

def load_json(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            try: return json.load(f)
            except: return []
    return []

def save_json(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

# --- AUTH ROUTES ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        users = load_json(USER_FILE)
        user = next((u for u in users if u['email'] == email and u['password'] == password), None)
        
        if user:
            session['user_email'] = user.get('email')
            session['first_name'] = user.get('first_name', 'User')
            session['last_name'] = user.get('last_name', '')
            session['is_admin'] = user.get('is_admin', False) or user.get('email') == 'fitz3663@gmail.com'
            return redirect(url_for('index'))
        return "Invalid credentials", 401
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        users = load_json(USER_FILE)
        email = request.form.get('email')
        new_user = {
            "first_name": request.form.get('first_name'),
            "last_name": request.form.get('last_name'),
            "email": email,
            "password": request.form.get('password'),
            "is_admin": True if email == 'fitz3663@gmail.com' else False
        }
        users.append(new_user)
        save_json(USER_FILE, users)
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# --- CHAT & HISTORY ---

@app.route('/')
def index():
    if 'user_email' not in session: return redirect(url_for('login'))
    all_chats = load_json(CHAT_FILE)
    user_chats = [c for c in all_chats if c.get('email') == session['user_email']]
    user_chats.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    return render_template('index.html', first_name=session.get('first_name'), history=user_chats)

@app.route('/chat', methods=['POST'])
def chat():
    if 'user_email' not in session: return jsonify({"error": "Unauthorized"}), 401
    user_message = request.json.get('message')
    response = search_documents_web(user_message)
    history = load_json(CHAT_FILE)
    history.append({
        "email": session['user_email'],
        "user": user_message,
        "bot": response,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    save_json(CHAT_FILE, history)
    return jsonify({"answer": response})

@app.route('/delete_chat', methods=['POST'])
def delete_chat():
    if 'user_email' not in session: return jsonify({"status": "fail"}), 401
    ts = request.json.get('timestamp')
    history = load_json(CHAT_FILE)
    history = [c for c in history if not (c.get('email') == session['user_email'] and c.get('timestamp') == ts)]
    save_json(CHAT_FILE, history)
    return jsonify({"status": "success"})

@app.route('/rename_chat', methods=['POST'])
def rename_chat():
    if 'user_email' not in session: return jsonify({"status": "fail"}), 401
    ts = request.json.get('timestamp')
    new_name = request.json.get('new_name')
    history = load_json(CHAT_FILE)
    for chat in history:
        if chat.get('email') == session['user_email'] and chat.get('timestamp') == ts:
            chat['user'] = new_name
    save_json(CHAT_FILE, history)
    return jsonify({"status": "success"})

# --- ADMIN FUNCTIONS ---

@app.route('/admin')
def admin_dashboard():
    if not session.get('is_admin'): return "Access Denied", 403
    users = load_json(USER_FILE)
    return render_template('admin.html', users=users)

@app.route('/toggle_admin/<email>', methods=['POST'])
def toggle_admin(email):
    if not session.get('is_admin'): return "Unauthorized", 403
    users = load_json(USER_FILE)
    for u in users:
        if u['email'] == email:
            u['is_admin'] = not u.get('is_admin', False)
    save_json(USER_FILE, users)
    return redirect(url_for('admin_dashboard'))

@app.route('/delete_user/<email>', methods=['POST'])
def delete_user(email):
    if not session.get('is_admin'): return "Unauthorized", 403
    users = load_json(USER_FILE)
    users = [u for u in users if u['email'] != email]
    save_json(USER_FILE, users)
    return redirect(url_for('admin_dashboard'))

# --- DATA ENTRY ---

@app.route('/add')
def add_form():
    if 'user_email' not in session: return redirect(url_for('login'))
    
    # FIXED: Safe retrieval to avoid KeyError
    fname = session.get('first_name', '')
    lname = session.get('last_name', '')
    email = session.get('user_email', '')
    
    return render_template('add_item.html', 
                           full_name=f"{fname} {lname}".strip(), 
                           email=email)

@app.route('/save_item', methods=['POST'])
def save_item():
    user_name = request.form.get('user_name')
    notes = request.form.get('notes')
    new_entry = {"id": str(uuid.uuid4()), "notes": notes, "timestamp": str(datetime.now())}
    data = load_json(DATA_FILE)
    data.append(new_entry)
    save_json(DATA_FILE, data)
    
    try:
        embedding = model.encode([notes]).tolist()
        collection.add(ids=[new_entry["id"]], embeddings=embedding, documents=[notes], metadatas=[{"source": "Manual"}])
    except Exception as e:
        print(f"Vector storage failed: {e}")
        
    return '<script>window.close();</script>'

@app.route('/api/files')
def list_files():
    files = [f for f in os.listdir(LIBRARY_DIR) if f.endswith('.pdf')] if os.path.exists(LIBRARY_DIR) else []
    return jsonify({"files": files})

@app.route('/library/<filename>')
def get_pdf(filename):
    return send_from_directory(LIBRARY_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)