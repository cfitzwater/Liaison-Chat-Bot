import os
import time
import random
import chromadb
from google import genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# 1. HARD-LOCK PATHS
load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

# 2. Initialize Components
print(f"--- AI Search Engine Initializing ---")
print(f"Target Database: {CHROMA_PATH}")

try:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection("liaison_library")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"Status: {collection.count()} clinical chunks available in memory.")
except Exception as e:
    print(f"CRITICAL DATABASE ERROR: {e}")

def search_documents_web(query):
    # 3. Embedding & Vector Search
    query_embedding = model.encode([query])
    results = collection.query(
        query_embeddings=query_embedding.tolist(), 
        n_results=6 
    )
    
    # 4. Check if Search found anything
    if not results or not results['documents'] or not results['documents'][0]:
        return "I'm sorry, I couldn't find any information in the clinical library regarding that query."

    # 5. Build Context with Smart Citations
    context = ""
    for i in range(len(results['documents'][0])):
        doc_text = results['documents'][0][i]
        meta = results['metadatas'][0][i]
        source = meta.get('source', 'Unknown')
        
        if "Contributor:" in source:
            citation = f"**{source}**"
        else:
            clean_file = os.path.basename(source)
            page = meta.get('page', 'N/A')
            citation = f"[Source: {clean_file}, Page: {page}](/library/{clean_file}#page={page})"
        
        context += f"SOURCE_TAG: {citation}\nCONTENT: {doc_text}\n\n"

    # 6. Gemini API Connection
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: 
        return "Error: GEMINI_API_KEY missing from environment."
        
    gen_client = genai.Client(api_key=api_key)
    
    prompt = (
        f"You are a Clinical Liaison Bot. Answer the query using ONLY the context provided.\n\n"
        f"INSTRUCTION: Cite your source by copying the exact 'SOURCE_TAG' at the end of your reply.\n\n"
        f"Context:\n{context}\n"
        f"Query: {query}"
    )

    # 7. Execution (Gemini 2.5 Flash)
    for attempt in range(2):
        try:
            response = gen_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config={'temperature': 0.1}
            )
            return response.text
        except Exception as e:
            print(f"\n--- API ATTEMPT {attempt + 1} LOG ---")
            print(f"SOURCES FOUND: {results['metadatas'][0]}")
            print(f"ERROR: {e}")
            
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                time.sleep(5)
                continue
                
            return f"Clinical Engine Error: {str(e)}"

    return "The system is currently over its free-tier capacity. Please try again in 30 seconds."

# 8. EXPORTS (This must be outside any function or string)
__all__ = ['search_documents_web', 'collection', 'model', 'CHROMA_PATH']