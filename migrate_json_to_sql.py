import json
import os
import uuid  # Required for generating session IDs
from datetime import datetime
from models import db, User, ChatHistory
from web_app import app

# File paths
USER_FILE = "users.json"
CHAT_FILE = "chat_history.json"

def load_json(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            try:
                return json.load(f)
            except:
                return []
    return []

def migrate():
    with app.app_context():
        # 1. Create tables with the NEW structure (including chat_session_id)
        db.create_all()
        print("--- Database tables initialized ---")

        # 2. Migrate Users
        json_users = load_json(USER_FILE)
        for u in json_users:
            exists = User.query.filter_by(email=u['email']).first()
            if not exists:
                new_user = User(
                    first_name=u.get('first_name', 'User'),
                    last_name=u.get('last_name', ''),
                    email=u['email'],
                    password=u['password'],
                    is_admin=u.get('is_admin', False)
                )
                db.session.add(new_user)
                print(f"Migrating User: {u['email']}")
        
        db.session.commit()
        print("--- User migration complete ---")

        # 3. Migrate Chat History
        json_chats = load_json(CHAT_FILE)
        for c in json_chats:
            user = User.query.filter_by(email=c['email']).first()
            if user:
                # Convert timestamp string to Python datetime object
                try:
                    ts = datetime.strptime(c['timestamp'], "%Y-%m-%d %H:%M:%S")
                except:
                    ts = datetime.utcnow()

                # CRITICAL FIX: Assign a unique session ID to each old message
                # This treats every old JSON message as its own "thread" for now
                new_chat = ChatHistory(
                    user_id=user.id,
                    chat_session_id=str(uuid.uuid4()), 
                    user_message=c['user'],
                    bot_response=c['bot'],
                    timestamp=ts
                )
                db.session.add(new_chat)
        
        db.session.commit()
        print("--- Chat history migration complete ---")
        print("Migration finished successfully! You can now delete your .json files.")

if __name__ == "__main__":
    migrate()