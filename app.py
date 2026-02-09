# app.py

import chromadb
from sentence_transformers import SentenceTransformer
from data_ingestion import chunk_pdf
import uuid
import os
import google.generativeai as genai

# Initialize ChromaDB client and collection
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("liaison_library")

# Initialize sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def add_document():
    """Prompts for a file path, processes the PDF, and adds it to the collection."""
    file_path = input("Enter the path to the PDF document: ")
    try:
        chunks = chunk_pdf(file_path)
        if not chunks:
            print("No text could be extracted from the document.")
            return

        # Generate embeddings for each chunk
        embeddings = model.encode([chunk["content_text"] for chunk in chunks])

        # Prepare data for ChromaDB
        documents = [chunk["content_text"] for chunk in chunks]
        metadatas = [{"source": chunk["source_file"], "page": chunk["page_number"]} for chunk in chunks]
        ids = [str(uuid.uuid4()) for _ in chunks]

        # Add to collection
        collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Successfully added {len(chunks)} chunks from {file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

def display_all_chunks():
    """Displays all the document chunks stored in the collection."""
    print("\n--- All Document Chunks ---")
    results = collection.get()
    if not results['documents']:
        print("No document chunks have been added yet.")
        return

    for i, doc in enumerate(results['documents']):
        print(f"ID: {results['ids'][i]}")
        print(f"Source: {results['metadatas'][i].get('source', 'N/A')}, Page: {results['metadatas'][i].get('page', 'N/A')}")
        print(f"Content: {doc}")
        print("-" * 20)

def search_documents():
    """Searches the collection for a given query and generates a contextual answer."""
    query = input("Enter your search query: ")
    if not query:
        print("Query cannot be empty.")
        return

    # Check for API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        try:
            api_key = input("Please enter your Gemini API key: ")
            os.environ["GEMINI_API_KEY"] = api_key
        except (EOFError, KeyboardInterrupt):
            print("\nAPI key input cancelled. Exiting search.")
            return

    genai.configure(api_key=api_key)

    # Generate embedding for the query
    query_embedding = model.encode([query])

    # Perform search in ChromaDB
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=5
    )

    if not results['documents'][0]:
        print("No relevant document chunks found.")
        return

    # Prepare context for the LLM
    context = ""
    for i, doc in enumerate(results['documents'][0]):
        source = results['metadatas'][0][i].get('source', 'N/A')
        page = results['metadatas'][0][i].get('page', 'N/A')
        context += f"Source: {source}, Page: {page}\nContent: {doc}\n\n"

    # Construct the prompt
    prompt = f"""
    You are the Liaison Library Bot. Your task is to answer the user's query based *only* on the provided context.
    Do not use any external knowledge.
    For every piece of information you use, you MUST cite the source file and page number.

    Context:
    ---
    {context}
    ---

    Query: "{query}"

    Answer:
    """

    try:
        # Call the Gemini API
        llm = genai.GenerativeModel('gemini2.5-pro')
        response = llm.generate_content(prompt)

        print("\n--- Generated Answer ---")
        print(response.text)
        print("\n--- Sources ---")
        for i, doc in enumerate(results['documents'][0]):
            print(f"Source: {results['metadatas'][0][i].get('source', 'N/A')}, Page: {results['metadatas'][0][i].get('page', 'N/A')}")

    except Exception as e:
        print(f"\nAn error occurred while generating the answer: {e}")
        print("Please check your API key and network connection.")


def main():
    """The main function to run the CLI application."""
    while True:
        print("\n--- Liaison Library Bot ---")
        print("1. Add a new document")
        print("2. View all document chunks")
        print("3. Search documents")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            add_document()
        elif choice == '2':
            display_all_chunks()
        elif choice == '3':
            search_documents()
        elif choice == '4':
            print("Exiting the application. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")

if __name__ == "__main__":
    main()
