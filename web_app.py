import os
import uuid
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory, session
from flask_admin import Admin
from flask_admin.contrib.sqla import ModelView
from models import db, User, ChatHistory, LibraryEntry

# Bridge to app.py logic
try:
    from app import search_documents_web, collection, model 
except ImportError:
    print("Warning: app.py not found or ChromaDB not initialized.")

app = Flask(__name__)
app.secret_key = 'liaison_secret_key_123'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///project.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
LIBRARY_DIR = "library"

db.init_app(app)

# --- FLASK-ADMIN (Raw Database Tool) ---
# Accessible at /admin_db/ 
class AdminModelView(ModelView):
    def is_accessible(self):
        return session.get('is_admin') == True

admin_gui = Admin(app, name='Liaison DB Admin', url='/admin_db')
admin_gui.add_view(AdminModelView(User, db.session))
admin_gui.add_view(AdminModelView(ChatHistory, db.session))

with app.app_context():
    db.create_all()

# --- AUTH ROUTES ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email, password=password).first()
        if user:
            session['user_id'] = user.id
            session['user_email'] = user.email
            session['first_name'] = user.first_name
            session['last_name'] = user.last_name
            session['is_admin'] = user.is_admin
            return redirect(url_for('index'))
        return "Invalid credentials", 401
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        if User.query.filter_by(email=email).first(): return "User exists", 400
        is_admin = (email == 'fitz3663@gmail.com' or User.query.first() is None)
        new_user = User(
            first_name=request.form.get('first_name'), 
            last_name=request.form.get('last_name'), 
            email=email, 
            password=request.form.get('password'), 
            is_admin=is_admin
        )
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# --- MAIN CHAT INTERFACE ---

@app.route('/')
def index():
    if 'user_id' not in session: return redirect(url_for('login'))
    
    # Emergency Admin Override
    if session.get('user_email') == 'fitz3663@gmail.com':
        session['is_admin'] = True

    all_chats = ChatHistory.query.filter_by(user_id=session['user_id']).order_by(ChatHistory.timestamp.desc()).all()
    seen_sessions, sidebar_history = set(), []
    for c in all_chats:
        if c.chat_session_id not in seen_sessions:
            sidebar_history.append(c)
            seen_sessions.add(c.chat_session_id)
            
    return render_template('index.html', first_name=session.get('first_name'), history=sidebar_history)

@app.route('/chat', methods=['POST'])
def chat():
    if 'user_id' not in session: return jsonify({"error": "Unauthorized"}), 401
    if 'current_chat_id' not in session: session['current_chat_id'] = str(uuid.uuid4())
    user_message = request.json.get('message')
    response = search_documents_web(user_message)
    new_chat = ChatHistory(user_id=session['user_id'], chat_session_id=session['current_chat_id'], user_message=user_message, bot_response=response)
    db.session.add(new_chat)
    db.session.commit()
    return jsonify({"answer": response})

# --- CUSTOM ADMIN DASHBOARD (Fixes the 404) ---

@app.route('/admin', strict_slashes=False)
def admin_dashboard():
    if not session.get('is_admin'): return "Access Denied", 403
    users = User.query.all()
    return render_template('admin.html', users=users)

@app.route('/toggle_admin/<int:user_id>', methods=['POST'])
def toggle_admin(user_id):
    if not session.get('is_admin'): return "Unauthorized", 403
    user = User.query.get(user_id)
    if user:
        user.is_admin = not user.is_admin
        db.session.commit()
    return redirect(url_for('admin_dashboard'))

@app.route('/delete_user/<int:user_id>', methods=['POST'])
def delete_user(user_id):
    if not session.get('is_admin'): return "Unauthorized", 403
    user = User.query.get(user_id)
    if user:
        db.session.delete(user)
        db.session.commit()
    return redirect(url_for('admin_dashboard'))

# --- KNOWLEDGE SUBMISSION (Full Name Attribution) ---

@app.route('/add')
def add_form():
    if 'user_id' not in session: return redirect(url_for('login'))
    # Pulling both names for the Contributor field
    full_name = f"{session.get('first_name')} {session.get('last_name')}"
    return render_template('add_item.html', full_name=full_name)

@app.route('/save_item', methods=['POST'])
def save_item():
    label = request.form.get('label_tag')
    notes = request.form.get('notes')
    user_full_name = request.form.get('user_name')
    entry_id = str(uuid.uuid4())
    
    formatted_content = f"LABEL: {label}\nCONTRIBUTOR: {user_full_name}\n\n{notes}"
    db.session.add(LibraryEntry(id=entry_id, notes=formatted_content))
    db.session.commit()
    
    try:
        embedding = model.encode([formatted_content]).tolist()
        collection.add(
            ids=[entry_id], 
            embeddings=embedding, 
            documents=[formatted_content], 
            metadatas=[{"source": f"Contributor: {user_full_name}"}]
        )
    except: pass
    return '<script>window.close();</script>'

# --- HELPERS ---

@app.route('/get_session/<session_id>')
def get_session(session_id):
    chats = ChatHistory.query.filter_by(chat_session_id=session_id).order_by(ChatHistory.timestamp.asc()).all()
    session['current_chat_id'] = session_id
    return jsonify([{"user": c.user_message, "bot": c.bot_response} for c in chats])

@app.route('/rename_chat', methods=['POST'])
def rename_chat():
    session_id = request.json.get('session_id')
    new_name = request.json.get('new_name')
    first_msg = ChatHistory.query.filter_by(chat_session_id=session_id).first()
    if first_msg:
        first_msg.user_message = new_name
        db.session.commit()
        return jsonify({"status": "success"})
    return jsonify({"status": "fail"}), 404

@app.route('/delete_chat', methods=['POST'])
def delete_chat():
    session_id = request.json.get('session_id')
    ChatHistory.query.filter_by(chat_session_id=session_id).delete()
    db.session.commit()
    return jsonify({"status": "success"})

@app.route('/api/files')
def list_files():
    files = [f for f in os.listdir(LIBRARY_DIR) if f.endswith('.pdf')] if os.path.exists(LIBRARY_DIR) else []
    return jsonify({"files": files})

@app.route('/library/<filename>')
def get_pdf(filename):
    return send_from_directory(LIBRARY_DIR, filename)

@app.route('/new_chat')
def new_chat_redirect():
    session.pop('current_chat_id', None)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)