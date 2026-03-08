import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def _normalize(embeddings):
    """L2-normalize embeddings for cosine similarity via IndexFlatIP."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-10)


def build_index(chunks):
    """
    Builds a FAISS index using Inner Product on L2-normalized vectors,
    which is equivalent to cosine similarity.
    Previously used IndexFlatL2 which is NOT equivalent to cosine similarity
    for sentence embeddings.
    """

    embeddings = embedding_model.encode(chunks).astype("float32")
    embeddings = _normalize(embeddings)

    dimension = embeddings.shape[1]

    # IndexFlatIP + normalized vectors = cosine similarity
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


def save_index(index, chunks):

    faiss.write_index(index, "rag_index.faiss")

    with open("rag_chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)


def load_index():

    index = faiss.read_index("rag_index.faiss")

    with open("rag_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    return index, chunks


def retrieve(query, index, chunks, k=3):
    """
    Retrieves top-k most semantically similar chunks using cosine similarity.
    """

    query_vector = embedding_model.encode([query]).astype("float32")
    query_vector = _normalize(query_vector)

    distances, indices = index.search(query_vector, k)

    return "\n\n".join([chunks[i] for i in indices[0] if i < len(chunks)])