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
    # 3. Create query embedding
    query_embedding = model.encode([query])
    
    # 4. Retrieve top 4 relevant chunks
    results = collection.query(
        query_embeddings=query_embedding.tolist(), 
        n_results=4
    )
    
    if not results['documents'][0]:
        return "I'm sorry, I couldn't find any information in the clinical library."

    # 5. Build context with "Smart Citations"
    context = ""
    for i in range(len(results['documents'][0])):
        doc_text = results['documents'][0][i]
        meta = results['metadatas'][0][i]
        source = meta.get('source', 'Unknown')
        
        # LOGIC: Check if this is a manual entry vs a PDF file
        if "Contributor:" in source:
            # Manual entry - just show the name
            citation_label = f"**Source: {source}**"
        else:
            # PDF File - Create the clickable deep link
            clean_file = os.path.basename(source)
            page = meta.get('page', 'N/A')
            citation_label = f"[Source: {clean_file}, Page: {page}](http://127.0.0.1:5000/library/{clean_file}#page={page})"
        
        context += f"CITATION TO USE: {citation_label}\nCONTENT: {doc_text}\n\n"

    # 6. Initialize Gemini 2.5 Flash
    api_key = os.getenv("GEMINI_API_KEY")
    gen_client = genai.Client(api_key=api_key)
    
    prompt = (
        f"You are a Clinical Liaison Bot. Answer the query using ONLY the provided content.\n\n"
        f"INSTRUCTION: You MUST cite your work by copy-pasting the exact 'CITATION TO USE' "
        f"provided in the context at the very end of your answer. Keep it clickable.\n\n"
        f"Context:\n{context}\n"
        f"Query: {query}"
    )

    # 7. Execution Loop with Exponential Backoff
    for attempt in range(4):
        try:
            response = gen_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config={'temperature': 0.1}
            )
            return response.text
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait_time)
                continue
            return f"Error connecting to AI: {str(e)}"

    return "The system is currently busy. Please try again in a moment."