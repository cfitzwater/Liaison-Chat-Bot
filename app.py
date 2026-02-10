import os
import uuid
import json
import chromadb
from google import genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# 1. Setup & Credentials
load_dotenv()

# 2. THE MASTER LIST - Requirement: Collection of items in a list of dictionaries
master_data_list = [] 

# 3. System Initialization
print("Initializing Liaison Library Bot...")
try:
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection("liaison_library")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    from data_ingestion import chunk_pdf
    print("Systems Ready.")
except Exception as e:
    print(f"Initialization Error: {e}")

def sync_data():
    """Maps ChromaDB data into the master_data_list of dictionaries."""
    global master_data_list
    try:
        raw_data = collection.get()
        if raw_data and raw_data['ids']:
            master_data_list = [] 
            for i in range(len(raw_data['ids'])):
                meta = raw_data['metadatas'][i] if raw_data['metadatas'] else {}
                item_dict = {
                    "id": raw_data['ids'][i],
                    "source": meta.get('source', 'N/A'),
                    "page": meta.get('page', 'N/A'),
                    "content": raw_data['documents'][i]
                }
                master_data_list.append(item_dict)
            print(f"Sync Success: {len(master_data_list)} items are now in memory.")
        else:
            master_data_list = []
            print("Sync: Database is currently empty.")
    except Exception as e:
        print(f"Sync Error: {e}")

def list_items():
    """Requirement: Iterate through the list of dictionaries to display items."""
    if not master_data_list:
        print("\nYour internal library list is empty. Add a document first!")
        return

    print(f"\n--- Current Library Items ({len(master_data_list)} total) ---")
    for index, item in enumerate(master_data_list[:10]):
        source_name = os.path.basename(str(item['source']))
        print(f"[{index + 1}] Source: {source_name} | Page: {item['page']}")
        print(f"    Snippet: {item['content'][:75].strip()}...")
    
    if len(master_data_list) > 10:
        print(f"... and {len(master_data_list) - 10} more items.")
    print("-----------------------------------------------")

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
            collection.add(
                embeddings=model.encode([item_dict["content"]]).tolist(),
                documents=[item_dict["content"]],
                metadatas=[{"source": item_dict["source"], "page": item_dict["page"]}],
                ids=[item_dict["id"]]
            )
        sync_data() 
        print(f"Successfully added {len(chunks)} items.")
    except Exception as e:
        print(f"Error adding document: {e}")

def save_to_json():
    """Requirement: Export structured data to a JSON file."""
    if not master_data_list:
        print("\nNothing to export.")
        return
    
    filename = "library_export.json"
    try:
        with open(filename, 'w') as f:
            json.dump(master_data_list, f, indent=4)
        print(f"Exported {len(master_data_list)} items to {filename}")
    except Exception as e:
        print(f"Export Error: {e}")

def clear_library():
    """Wipes all data from memory, database, and JSON export."""
    confirm = input("Are you sure you want to delete ALL data? (y/n): ")
    if confirm.lower() == 'y':
        all_data = collection.get()
        if all_data['ids']:
            collection.delete(ids=all_data['ids'])
        master_data_list.clear()
        if os.path.exists("library_export.json"):
            os.remove("library_export.json")
        print("Library cleared. You now have a blank slate.")

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

    prompt = f"Using ONLY the following context, answer the query. Cite sources.\nContext: {context}\n\nQuery: {query}"
    
    print("Consulting Gemini...")
    response = gen_client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
    print(f"\n--- AI RESPONSE ---\n{response.text}\n")

def main():
    # Initial sync on startup
    sync_data()
    
    while True:
        print("\n--- Liaison Library Bot Menu ---")
        print("1. Add a Document")
        print("2. List All Items (Structured View)")
        print("3. Search Library (RAG)")
        print("4. Export to JSON")
        print("5. Clear All Library Data") # Fixed the label here!
        print("6. Exit")
        
        choice = input("Select an option (1-6): ")
        if choice == '1': add_document()
        elif choice == '2': list_items()
        elif choice == '3': search_documents()
        elif choice == '4': save_to_json()
        elif choice == '5': clear_library() # Fixed the function call here!
        elif choice == '6': break

if __name__ == "__main__":
    main()