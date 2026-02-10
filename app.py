import os
import uuid
import chromadb
from google import genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# 1. Load the secret API key from your .env file
load_dotenv()

# 2. Initialize the AI Brain and Database
print("Initializing Liaison Library Bot systems...")
try:
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection("liaison_library")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Import chunking logic from your data_ingestion.py
    from data_ingestion import chunk_pdf
    print("Systems Ready.\n")
except ImportError:
    print("Warning: data_ingestion.py not found. Option 1 will be disabled.")
except Exception as e:
    print(f"Initialization Error: {e}")

def add_document():
    """Option 1: Ingests a PDF into the vector database."""
    file_path = input("Enter the full path to your PDF: ").strip()
    if not os.path.exists(file_path):
        print("Error: File not found. Check your spelling and case sensitivity.")
        return

    try:
        print(f"Processing {file_path}...")
        chunks = chunk_pdf(file_path)
        
        documents = [c["content_text"] for c in chunks]
        metadatas = [{"source": c["source_file"], "page": c["page_number"]} for c in chunks]
        ids = [str(uuid.uuid4()) for _ in chunks]
        
        # Create searchable embeddings
        embeddings = model.encode(documents)
        collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Success! Added {len(chunks)} sections to your library.")
    except Exception as e:
        print(f"Failed to add document: {e}")

def search_documents():
    """Option 3: Search using RAG and Gemini 2.5 Flash."""
    query = input("\nWhat is your clinical or CMS question? ")
    if not query: return

    # Auto-populate the key from .env
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in your .env file.")
        return
    
    gen_client = genai.Client(api_key=api_key, http_options={'api_version': 'v1'})
    
    # Semantic search in ChromaDB
    query_embedding = model.encode([query])
    results = collection.query(query_embeddings=query_embedding.tolist(), n_results=5)

    if not results['documents'][0]:
        print("No relevant information found in the library.")
        return

    # Build the context for the AI
    context = ""
    for i, doc in enumerate(results['documents'][0]):
        source = results['metadatas'][0][i].get('source', 'N/A')
        page = results['metadatas'][0][i].get('page', 'N/A')
        context += f"Source: {source}, Page: {page}\nContent: {doc}\n\n"

    prompt = f"""
    You are the Liaison Library Bot. Use the context below to answer the user.
    Cite the Source and Page number for every fact you provide.
    Context: {context}
    Query: {query}
    """
    
    print("Consulting Gemini 2.5 Flash...")
    response = gen_client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
    print(f"\n--- AI RESPONSE ---\n{response.text}\n------------------")

def main():
    while True:
        print("\n--- Liaison Library Bot Menu ---")
        print("1. Add a new document (PDF)")
        print("2. Check Library Size")
        print("3. Search Clinical/CMS Library")
        print("4. Exit")
        
        choice = input("Select an option (1-4): ")
        if choice == '1': add_document()
        elif choice == '2':
            count = len(collection.get()['documents'])
            print(f"Your library currently has {count} data chunks.")
        elif choice == '3': search_documents()
        elif choice == '4': break
        else: print("Invalid selection.")

if __name__ == "__main__":
    main()