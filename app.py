import os
import uuid
import json
import chromadb
from google import genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# 1. Setup & Credentials
load_dotenv()

# 2. THE MASTER LIST
master_data_list = [] 
DATA_FILE = "library_data.json" # Requirement: Defined data file

# 3. System Initialization
print("Initializing Liaison Library Bot (Technical Challenge Build)...")
try:
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection("liaison_library")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    from data_ingestion import chunk_pdf
except Exception as e:
    print(f"Initialization Error: {e}")

# Requirement: Load data from file on startup
def load_data_from_file():
    """Requirement: Check for existence of JSON and load it; start empty if missing."""
    global master_data_list
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r') as f:
                master_data_list = json.load(f)
            print(f"Startup: Successfully loaded {len(master_data_list)} items from {DATA_FILE}.")
        except Exception as e:
            print(f"Error loading {DATA_FILE}: {e}")
            master_data_list = []
    else:
        print(f"Startup: {DATA_FILE} not found. Starting with an empty dataset.")
        master_data_list = []

# Requirement: Save data immediately on modification
def save_data_to_file():
    """Requirement: Write the structured dataset to the file immediately."""
    try:
        with open(DATA_FILE, 'w') as f:
            json.dump(master_data_list, f, indent=4)
        print(f"File Sync: {DATA_FILE} updated.")
    except Exception as e:
        print(f"Save Error: {e}")

# Run the load function at startup
load_data_from_file()

def add_document():
    file_path = input("Enter the full path to your PDF: ").strip()
    if not os.path.exists(file_path):
        print("Error: File not found.")
        return

    try:
        print(f"Processing {file_path}...")
        chunks = chunk_pdf(file_path)
        for chunk in chunks:
            item_dict = {
                "id": str(uuid.uuid4()),
                "source": chunk["source_file"],
                "page": chunk["page_number"],
                "content": chunk["content_text"]
            }
            master_data_list.append(item_dict)

            collection.add(
                embeddings=model.encode([item_dict["content"]]).tolist(),
                documents=[item_dict["content"]],
                metadatas=[{"source": item_dict["source"], "page": item_dict["page"]}],
                ids=[item_dict["id"]]
            )
        
        # Requirement: Save immediately after modification
        save_data_to_file()
        print(f"Successfully added {len(chunks)} items.")
    except Exception as e:
        print(f"Error adding document: {e}")

def list_items():
    if not master_data_list:
        print("\nYour internal library list is empty.")
        return

    print(f"\n--- Current Library Items ({len(master_data_list)} total) ---")
    for index, item in enumerate(master_data_list[:10]):
        source_name = os.path.basename(str(item['source']))
        print(f"[{index + 1}] Source: {source_name} | Page: {item['page']}")
        print(f"    Snippet: {item['content'][:75].strip()}...")
    
    if len(master_data_list) > 10:
        print(f"... and {len(master_data_list) - 10} more items.")
    print("-----------------------------------------------")

def clear_library():
    confirm = input("Are you sure you want to delete ALL data? (y/n): ")
    if confirm.lower() == 'y':
        all_data = collection.get()
        if all_data['ids']:
            collection.delete(ids=all_data['ids'])
        
        # Requirement: Modify the data and write back immediately
        master_data_list.clear()
        if os.path.exists(DATA_FILE):
            os.remove(DATA_FILE)
        print("Library and data file cleared.")

def search_documents():
    query = input("\nWhat is your clinical question? ")
    if not query: return
    
    api_key = os.getenv("GEMINI_API_KEY")
    gen_client = genai.Client(api_key=api_key, http_options={'api_version': 'v1'})
    
    query_embedding = model.encode([query])
    results = collection.query(query_embeddings=query_embedding.tolist(), n_results=5)
    
    if not results['documents'][0]:
        print("No matches found.")
        return

    context = ""
    for i, doc in enumerate(results['documents'][0]):
        meta = results['metadatas'][0][i]
        context += f"Source: {meta.get('source')}, Page: {meta.get('page')}\nContent: {doc}\n\n"

    prompt = f"Using ONLY context: {context}\n\nQuery: {query}"
    
    print("Consulting Gemini...")
    response = gen_client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
    print(f"\n--- AI RESPONSE ---\n{response.text}\n")

def main():
    while True:
        print("\n--- Liaison Library Bot Menu ---")
        print("1. Add a Document")
        print("2. List All Items (Loaded from JSON)")
        print("3. Search Library (RAG)")
        print("4. Clear All Library Data")
        print("5. Exit")
        
        choice = input("Select an option (1-5): ")
        if choice == '1': add_document()
        elif choice == '2': list_items()
        elif choice == '3': search_documents()
        elif choice == '4': clear_library()
        elif choice == '5': break

if __name__ == "__main__":
    main()