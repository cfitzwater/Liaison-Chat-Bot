import os
import time
import random
import chromadb
from google import genai
from google.api_core import exceptions
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# 1. Setup & Environment
load_dotenv()

# 2. Initialize Global AI Components
print("Initializing AI Search Engine (Liaison Library)...")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("liaison_library")
model = SentenceTransformer('all-MiniLM-L6-v2')

def search_documents_web(query):
    """
    Restored search function using gemini-2.5-flash and pre-formatted links.
    """
    # 3. Create query embedding
    query_embedding = model.encode([query])
    
    # 4. Retrieve top 4 relevant chunks
    results = collection.query(
        query_embeddings=query_embedding.tolist(), 
        n_results=4
    )
    
    if not results['documents'][0]:
        return "I'm sorry, I couldn't find any information in the clinical library."

    # 5. Build context & PRE-FORMAT LINKS (The stuff that makes them clickable)
    context = ""
    for i, doc in enumerate(results['documents'][0]):
        meta = results['metadatas'][0][i]
        clean_filename = os.path.basename(meta.get('source', 'Unknown'))
        page_num = meta.get('page', 'Unknown')
        
        # This string is what the AI will copy/paste
        citation_link = f"[Source: {clean_filename}, Page: {page_num}](/library/{clean_filename}#page={page_num})"
        
        context += f"REFERENCE LINK: {citation_link}\nCONTENT: {doc}\n\n"

    # 6. Initialize Gemini 2.5 Flash
    api_key = os.getenv("GEMINI_API_KEY")
    gen_client = genai.Client(api_key=api_key)
    
    prompt = (
        f"You are a Clinical Liaison Bot. Answer the query using ONLY the provided content.\n\n"
        f"INSTRUCTION: You must cite your work by copy-pasting the 'REFERENCE LINK' "
        f"provided in the context at the end of your answer. Keep it clickable.\n\n"
        f"Context:\n{context}\n"
        f"Query: {query}"
    )

    # 7. Execution Loop (Ensuring we use the 2.5 model)
    for attempt in range(4):
        try:
            response = gen_client.models.generate_content(
                model='gemini-2.5-flash', # RESTORED TO 2.5
                contents=prompt,
                config={'temperature': 0.1}
            )
            return response.text
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
                continue
            return f"Error: {str(e)}"

    return "The system is currently busy. Please try again in a moment."