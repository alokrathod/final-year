import os
from modules.pdf_processor import extract_text_from_pdfs, chunk_text
from modules.retriever import build_index, save_index, load_index
from agentic_runner import agentic_pipeline


PDF_FOLDER = "data/srs_pdfs"


def initialize_rag():

    if os.path.exists("rag_index.faiss") and os.path.exists("rag_chunks.pkl"):
        print("Loading existing RAG index...")
        return load_index()

    print("Building RAG index for the first time...")

    documents = extract_text_from_pdfs(PDF_FOLDER)

    if not documents:
        print("No PDFs found. Running without RAG.")
        return None, None

    all_chunks = []
    for doc in documents:
        chunks = chunk_text(doc)
        all_chunks.extend(chunks)

    index = build_index(all_chunks)
    save_index(index, all_chunks)

    print("RAG index built and saved.\n")

    return index, all_chunks


def main():

    # Step 1: Initialize RAG
    index, chunks = initialize_rag()

    # Step 2: Take user input
    user_input = input("Enter user requirements:\n")

    # Step 3: Run agentic pipeline
    final_srs, ai_suggested = agentic_pipeline(
        user_input,
        index=index,
        chunks=chunks
    )

    print("\n===== FINAL SRS AFTER ITERATION =====\n")
    print(final_srs)

    if ai_suggested:
        print(f"\n===== AI-SUGGESTED REQUIREMENTS ({len(ai_suggested)}) =====")
        for i, req in enumerate(ai_suggested, 1):
            print(f"  {i}. {req}")
        print("\nThese are highlighted in bold with [AI-SUGGESTED] in the PDF.")


if __name__ == "__main__":
    main()