from pypdf import PdfReader
import os
import re


def extract_text_from_pdfs(folder_path):

    documents = []

    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return documents

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):

            file_path = os.path.join(folder_path, file)
            print(f"Reading: {file}")

            reader = PdfReader(file_path)
            text = ""

            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"

            documents.append(text)

    return documents


def chunk_text(text, max_chunk_size=1000, overlap=100):
    """
    Sentence-aware chunking with overlap.
    Splits on sentence boundaries instead of raw character count,
    preventing requirements and sentences from being cut mid-way.
    Overlap ensures context continuity across chunks.
    """

    # Split into sentences using punctuation boundaries
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_endings.split(text.strip())

    chunks = []
    current_chunk = ""

    for sentence in sentences:

        # If adding this sentence keeps us under limit, append
        if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
            current_chunk += (" " if current_chunk else "") + sentence

        else:
            # Save current chunk if non-empty
            if current_chunk:
                chunks.append(current_chunk.strip())

            # Start new chunk with overlap from end of previous chunk
            if overlap > 0 and current_chunk:
                overlap_text = current_chunk[-overlap:]
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk = sentence

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks