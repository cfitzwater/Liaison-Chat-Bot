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
    RAG search function with cleaned filenames and clickable citations.
    """
    # 3. Create query embedding
    query_embedding = model.encode([query])
    
    # 4. Retrieve top 4 relevant chunks (Ensures we catch content from deep pages)
    results = collection.query(
        query_embeddings=query_embedding.tolist(), 
        n_results=4
    )
    
    if not results['documents'][0]:
        return "I'm sorry, I couldn't find any information in the clinical library regarding that topic."

    # 5. Build context for Gemini & Clean File Paths
    context = ""
    for i, doc in enumerate(results['documents'][0]):
        meta = results['metadatas'][0][i]
        
        # EXTRACT FILENAME ONLY: Turns '/home/user/Desktop/File.pdf' into 'File.pdf'
        full_path = meta.get('source', 'Unknown')
        clean_filename = os.path.basename(full_path) 
        
        context += f"Source: {clean_filename}, Page: {meta.get('page')}\nContent: {doc}\n\n"

    # 6. Call Gemini 2.5 Flash API with specific Hyperlink instructions
    api_key = os.getenv("GEMINI_API_KEY")
    gen_client = genai.Client(api_key=api_key)
    
    prompt = (
        f"You are a Clinical Liaison Assistant. Using the provided context, answer the clinical query.\n\n"
        f"IMPORTANT CITATION RULE: You MUST cite your sources at the end of your response using this "
        f"exact Markdown link format so they are clickable:\n"
        f"[Source: FILENAME, Page: PAGENUM](/library/FILENAME#page=PAGENUM)\n\n"
        f"If a snippet mentions information is on another page, look through all provided context "
        f"snippets to see if that page's content is present before saying it is missing.\n\n"
        f"Context: {context}\n\n"
        f"Query: {query}"
    )

    # 7. Execution Loop with Exponential Backoff for Rate Limits
    for attempt in range(4):
        try:
            response = gen_client.models.generate_content(
                model='gemini-2.5-flash', # Set to your verified model
                contents=prompt
            )
            return response.text
        except Exception as e:
            # Catch Rate Limits (429) and Resource Exhausted errors
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"Rate limit hit. Retrying in {wait_time:.1f}s (Attempt {attempt + 1})...")
                time.sleep(wait_time)
                continue
            return f"Error communicating with AI: {str(e)}"

    return "The AI is currently at its quota limit. Please try again in 60 seconds."