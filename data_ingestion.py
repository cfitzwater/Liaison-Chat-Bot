# data_ingestion.py

from pypdf import PdfReader

def chunk_pdf(file_path: str) -> list[dict]:
    """
    Reads a PDF, extracts text, and splits it into chunks by page.

    Args:
        file_path: The path to the PDF file.

    Returns:
        A list of dictionaries, where each dictionary represents a chunk
        with its source file, page number, and content.
    """
    reader = PdfReader(file_path)
    chunks = []
    for i, page in enumerate(reader.pages):
        chunks.append({
            "source_file": file_path,
            "page_number": i + 1,
            "content_text": page.extract_text() or ""
        })
    return chunks
