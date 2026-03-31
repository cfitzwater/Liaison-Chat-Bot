import os
import uuid
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory, session
from werkzeug.utils import secure_filename
from pypdf import PdfReader
from models import db, User, ChatHistory, LibraryEntry, LibraryFile

# Link to AI Engine
try:
    from app import search_documents_web, collection, model, CHROMA_PATH
    print(f"--- Web App Linked to AI Memory at: {CHROMA_PATH} ---")
except ImportError:
    print("CRITICAL: app.py not found.")

app = Flask(__name__)
app.secret_key = 'liaison_secret_key_123'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///project.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'library')

db.init_app(app)

with app.app_context():
    db.create_all()

# --- AUTH & SIGNUP ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(email=request.form.get('email'), password=request.form.get('password')).first()
        if user and user.is_active:
            session.update({'user_id': user.id, 'user_email': user.email, 'first_name': user.first_name, 'last_name': user.last_name, 'is_admin': user.is_admin})
            return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            if not existing_user.is_active:
                # Reactivate the user
                existing_user.first_name = request.form.get('first_name')
                existing_user.last_name = request.form.get('last_name')
                existing_user.password = request.form.get('password')
                existing_user.is_active = True
                db.session.commit()
                return redirect(url_for('login'))
            else:
                # User already active
                return "User already exists and is active", 400
        else:
            # Create new user
            is_admin = (email == 'fitz3663@gmail.com' or User.query.first() is None)
            new_user = User(
                first_name=request.form.get('first_name'), last_name=request.form.get('last_name'),
                email=email, password=request.form.get('password'), is_admin=is_admin
            )
            db.session.add(new_user)
            db.session.commit()
            return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.clear(); return redirect(url_for('login'))

# --- CHAT & HISTORY ---
@app.route('/')
def index():
    if 'user_id' not in session: return redirect(url_for('login'))
    chats = ChatHistory.query.filter_by(user_id=session['user_id']).order_by(ChatHistory.timestamp.desc()).all()
    seen, history = set(), []
    for c in chats:
        if c.chat_session_id not in seen:
            history.append(c); seen.add(c.chat_session_id)
    files = LibraryFile.query.all()
    return render_template('index.html', first_name=session.get('first_name'), history=history, files=files)

@app.route('/chat', methods=['POST'])
def chat():
    if 'current_chat_id' not in session: session['current_chat_id'] = str(uuid.uuid4())
    user_msg = request.json.get('message')
    response = search_documents_web(user_msg)
    new_chat = ChatHistory(user_id=session['user_id'], chat_session_id=session['current_chat_id'], user_message=user_msg, bot_response=response)
    db.session.add(new_chat); db.session.commit()
    return jsonify({"answer": response})

@app.route('/get_session/<session_id>')
def get_session(session_id):
    chats = ChatHistory.query.filter_by(chat_session_id=session_id).order_by(ChatHistory.timestamp.asc()).all()
    session['current_chat_id'] = session_id 
    return jsonify([{"user": c.user_message, "bot": c.bot_response} for c in chats])

@app.route('/new_chat')
def new_chat():
    session.pop('current_chat_id', None)
    return redirect(url_for('index'))

@app.route('/rename_chat', methods=['POST'])
def rename_chat():
    data = request.json
    chat = ChatHistory.query.filter_by(chat_session_id=data.get('session_id')).first()
    if chat:
        chat.user_message = data.get('new_name')
        db.session.commit(); return jsonify({"status": "success"})
    return jsonify({"status": "error"}), 404

@app.route('/delete_chat', methods=['POST'])
def delete_chat():
    ChatHistory.query.filter_by(chat_session_id=request.json.get('session_id')).delete()
    db.session.commit(); return jsonify({"status": "success"})

# --- KNOWLEDGE SYNC ---
@app.route('/add_knowledge')
def add_knowledge():
    if 'user_id' not in session: return redirect(url_for('login'))
    return render_template('add_item.html', full_name=f"{session.get('first_name')} {session.get('last_name')}")

@app.route('/save_item', methods=['POST'])
def save_item():
    entry_id = str(uuid.uuid4())
    notes = request.form.get('notes')
    contributor = request.form.get('user_name')
    label_tag = request.form.get('label_tag')
    db.session.add(LibraryEntry(id=entry_id, notes=notes, label=label_tag, user_id=session['user_id']))
    db.session.commit()
    content = f"LABEL: {label_tag}\nCONTRIBUTOR: {contributor}\nNOTES: {notes}"
    collection.add(ids=[entry_id], embeddings=model.encode([content]).tolist(), documents=[content], metadatas=[{"source": f"Contributor: {contributor} - {label_tag}"}])
    return '<script>window.close();</script>'

@app.route('/my_knowledge')
def my_knowledge():
    if 'user_id' not in session: return redirect(url_for('login'))
    entries = LibraryEntry.query.filter_by(user_id=session['user_id']).all()
    return render_template('my_knowledge.html', entries=entries)

@app.route('/edit_my_knowledge/<entry_id>', methods=['POST'])
def edit_my_knowledge(entry_id):
    entry = db.session.get(LibraryEntry, entry_id)
    if entry and entry.user_id == session.get('user_id'):
        new_notes = request.json.get('notes')
        entry.notes = new_notes
        db.session.commit()
        contributor = f"{session.get('first_name')} {session.get('last_name')}"
        content = f"LABEL: {entry.label}\nCONTRIBUTOR: {contributor}\nNOTES: {new_notes}"
        collection.update(ids=[entry_id], embeddings=model.encode([content]).tolist(), documents=[content])
        return jsonify({"status": "success"})
    return jsonify({"status": "denied"}), 403

# --- ADMIN PANEL ---
@app.route('/admin')
def admin_dashboard():
    if not session.get('is_admin'): return "Denied", 403
    return render_template('admin.html', users=User.query.filter_by(is_active=True).all(), files=LibraryFile.query.all(), entries=LibraryEntry.query.all())

@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
def admin_delete_user(user_id):
    user = db.session.get(User, user_id)
    if user and user.email != session.get('user_email'):
        user.is_active = False
        db.session.commit()
        return jsonify({"status": "success"})
    return jsonify({"status": "error"}), 400

@app.route('/toggle_admin/<int:user_id>', methods=['POST'])
def toggle_admin(user_id):
    user = db.session.get(User, user_id)
    if user and user.email != session.get('user_email'):
        user.is_admin = not user.is_admin
        db.session.commit(); return jsonify({"status": "success"})
    return jsonify({"status": "error"}), 400

@app.route('/admin/upload', methods=['POST'])
def admin_upload():
    file = request.files.get('file')
    if file:
        fname = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        
        # Check if file already exists
        existing = LibraryFile.query.filter_by(filename=fname).first()
        if existing:
            # Remove old file from disk
            old_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
            if os.path.exists(old_path):
                os.remove(old_path)
            # Remove from Chroma
            try:
                collection.delete(where={"source": fname})
            except Exception as e:
                print(f"Error deleting from Chroma: {e}")
            # Remove from db
            db.session.delete(existing)
            db.session.commit()
        
        # Save new file
        file.save(path)
        db.session.add(LibraryFile(filename=fname))
        db.session.commit()
        
        # Ingest the PDF into Chroma
        try:
            reader = PdfReader(path)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and len(text.strip()) > 50:  # Ignore empty/short pages
                    chunk_id = f"{fname}_pg_{i}"
                    embedding = model.encode([text]).tolist()
                    collection.add(
                        ids=[chunk_id],
                        embeddings=embedding,
                        documents=[text],
                        metadatas=[{"source": fname, "page": i + 1}]
                    )
        except Exception as e:
            print(f"Error ingesting {fname}: {e}")
    
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/delete_file/<int:file_id>', methods=['POST'])
def admin_delete_file(file_id):
    f_rec = db.session.get(LibraryFile, file_id)
    if f_rec:
        path = os.path.join(app.config['UPLOAD_FOLDER'], f_rec.filename)
        if os.path.exists(path): os.remove(path)
        db.session.delete(f_rec)
        db.session.commit()
        # Remove from Chroma
        try:
            collection.delete(where={"source": f_rec.filename})
        except Exception as e:
            print(f"Error deleting from Chroma: {e}")
        return jsonify({"status": "success"})
    return jsonify({"status": "error"})

@app.route('/admin/edit_entry/<entry_id>', methods=['POST'])
def admin_edit_entry(entry_id):
    if not session.get('is_admin'): return jsonify({"status": "denied"}), 403
    entry = db.session.get(LibraryEntry, entry_id)
    if entry:
        new_notes = request.json.get('notes')
        entry.notes = new_notes
        db.session.commit()
        contributor = f"{entry.contributor.first_name} {entry.contributor.last_name}"
        content = f"LABEL: {entry.label}\nCONTRIBUTOR: {contributor}\nNOTES: {new_notes}"
        collection.update(ids=[entry_id], embeddings=model.encode([content]).tolist(), documents=[content])
        return jsonify({"status": "success"})
    return jsonify({"status": "error"}), 404

@app.route('/admin/delete_entry/<entry_id>', methods=['POST'])
def admin_delete_entry(entry_id):
    if not session.get('is_admin'): return jsonify({"status": "denied"}), 403
    entry = db.session.get(LibraryEntry, entry_id)
    if entry:
        db.session.delete(entry)
        db.session.commit()
        collection.delete(ids=[entry_id])
        return jsonify({"status": "success"})
    return jsonify({"status": "error"}), 404

@app.route('/library/<filename>')
def get_pdf(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)