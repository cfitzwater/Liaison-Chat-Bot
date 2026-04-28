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
    if not query or not query.strip():
        return "Please enter a valid query."
        
    # 3. Embedding & Vector Search
    try:
        query_embedding = model.encode([query])
        results = collection.query(
            query_embeddings=query_embedding.tolist(), 
            n_results=6 
        )
    except Exception as e:
        print(f"Database Query Error: {e}")
        return "Error retrieving context from the database."
    
    # 4. Check if Search found anything
    if not results or not results['documents'] or not results['documents'][0]:
        return "I'm sorry, I couldn't find any information in the clinical library regarding that query."

    # 5. Build Context with Smart Citations
    context = ""
    metadatas = results.get('metadatas')
    for i in range(len(results['documents'][0])):
        doc_text = results['documents'][0][i]
        meta = (metadatas[0][i] if metadatas and metadatas[0] else None) or {}
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
        f"You are a professional and highly accurate Clinical Liaison Assistant.\n"
        f"Your role is to answer clinical and administrative questions for healthcare liaisons based ONLY on the provided context.\n\n"
        f"STRICT RULES:\n"
        f"1. Base your answer STRICTLY on the context provided. Do not use outside knowledge or hallucinate.\n"
        f"2. If the context does not contain the answer, explicitly state: 'I'm sorry, I cannot find the answer to that in the current clinical library.'\n"
        f"3. Format your response clearly using bullet points and bold text for readability.\n"
        f"4. You MUST cite your sources by appending the exact 'SOURCE_TAG' directly after the relevant information.\n\n"
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
            return response.text or "Error: Empty response from AI."
        except Exception as e:
            print(f"\n--- API ATTEMPT {attempt + 1} LOG ---")
            print(f"SOURCES FOUND: {results.get('metadatas')}")
            print(f"ERROR: {e}")
            
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "503" in str(e) or "UNAVAILABLE" in str(e):
                time.sleep(5)
                continue
                
            return f"Clinical Engine Error: {str(e)}"

    return "The system is currently over its free-tier capacity. Please try again in 30 seconds."

# 8. EXPORTS (This must be outside any function or string)
__all__ = ['search_documents_web', 'collection', 'model', 'CHROMA_PATH']