import os
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# 1. HARD-LOCK PATHS (Prevents amnesia)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LIBRARY_DIR = os.path.join(BASE_DIR, "library")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

print(f"--- Data Ingestion ---")
print(f"Reading from: {LIBRARY_DIR}")
print(f"Saving to: {CHROMA_PATH}")

# 2. Initialize ChromaDB and Model
client = chromadb.PersistentClient(path=CHROMA_PATH)

# We delete and recreate to ensure a clean sync of the NEW documents
try:
    client.delete_collection("liaison_library")
except:
    pass

collection = client.get_or_create_collection("liaison_library")
model = SentenceTransformer('all-MiniLM-L6-v2')

def process_pdfs():
    if not os.path.exists(LIBRARY_DIR):
        print(f"ERROR: Library folder not found at {LIBRARY_DIR}")
        return

    pdf_files = [f for f in os.listdir(LIBRARY_DIR) if f.endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF files found in the library folder.")
        return

    for filename in pdf_files:
        print(f"Indexing: {filename}...")
        path = os.path.join(LIBRARY_DIR, filename)
        
        try:
            reader = PdfReader(path)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and len(text.strip()) > 50: # Ignore empty/short pages
                    chunk_id = f"{filename}_pg_{i}"
                    embedding = model.encode([text]).tolist()
                    
                    collection.add(
                        ids=[chunk_id],
                        embeddings=embedding,
                        documents=[text],
                        metadatas=[{"source": filename, "page": i + 1}]
                    )
            print(f"Done: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    process_pdfs()
    print(f"--- Ingestion Complete ---")
    print(f"Total items in AI memory: {collection.count()}")