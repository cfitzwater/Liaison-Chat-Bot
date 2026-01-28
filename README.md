# Project Design Document: Liaison Library Bot

## Section 1: Project Overview
**Project Title:** Liaison Library Bot  
**Application Purpose:** The Liaison Library Bot is a Retrieval-Augmented Generation (RAG) application designed to serve as an intelligent interface for the "The Real Liaison Team Chat" document repository. In healthcare liaison work, critical information is often buried in disparate PDFs, insurance grids, and SOPs. This application indexes those unstructured files into a vector database, allowing users to ask natural language questions and receive conversational answers grounded specifically in the department's approved documentation. This ensures "source-of-truth" accuracy while eliminating manual search time.

**Intended User:** Clinical Liaisons and Registered Nurse Liaisons who need immediate, cited access to facility protocols and insurance eligibility guidelines during the patient intake process.

## Section 2: Core Features
* **Automated Document Ingestion:** A Python-based pipeline that parses and "chunks" unstructured data from PDF, Word, and Excel files into searchable segments.
* **Semantic Search via Vector Database:** Utilizes a vector store (such as ChromaDB or FAISS) to find relevant information based on the intent of a query rather than just exact keyword matching.
* **Contextual Answer Generation:** Leverages a Large Language Model (LLM) to synthesize a coherent response based only on the retrieved document chunks.
* **Source Attribution & Citations:** For every response generated, the bot provides a direct reference to the specific file name and page number to ensure clinical compliance and verification.

## Section 3: Data Model
The following table defines the structure of a single **Document Chunk** (the primary data record) used by the system for retrieval:

| Field Name | Data Type | Description |
| :--- | :--- | :--- |
| chunk_id | UUID | A unique identifier for the specific segment of text. |
| source_file | String | The name of the original file (e.g., "2026_Eligibility_Manual.pdf"). |
| content_text | Text | The actual string of text extracted from the document. |
| embedding_vector | Array (Float) | The high-dimensional numerical representation (vector) of the text's meaning. |
| page_number | Integer | The specific page from the source document where the text resides. |
| last_indexed | DateTime | The timestamp of when the document was last processed by the pipeline. |
