from modules.llm import call_llm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ── Config ─────────────────────────────────────────
HIGH_THRESHOLD = 0.75
LOW_THRESHOLD = 0.60

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# ── Prompt ─────────────────────────────────────────
def build_prompt(requirement):
    return f"""
Explain the meaning of this requirement in ONE clear sentence.

Requirement:
{requirement}

Rules:
- Only ONE interpretation
- Do NOT list multiple meanings
- Do NOT explain
- Keep it concise

Output ONLY the sentence.
"""


# ── Cosine similarity ──────────────────────────────
def get_similarity(a, b):
    emb1 = embedding_model.encode([a])[0]
    emb2 = embedding_model.encode([b])[0]
    return cosine_similarity([emb1], [emb2])[0][0]


# ── Agreement function ─────────────────────────────
def is_same_meaning(sim):
    if sim >= HIGH_THRESHOLD:
        return True
    elif sim <= LOW_THRESHOLD:
        return False
    else:
        return True  # borderline → treat as same


# ── Q1 Computation ─────────────────────────────────
def compute_specificity_Q1(requirements):

    if not requirements:
        return 1.0

    nr = len(requirements)
    nui = 0

    print(f"\n[Q1] Evaluating specificity for {nr} requirements...\n")

    for i, req in enumerate(requirements, 1):

        print(f"REQ {i}: {req}")

        prompt = build_prompt(req)

        # ── Get interpretations ─────────────────
        interp_phi3 = call_llm(prompt, model="phi3", temperature=0).strip()
        interp_llama = call_llm(prompt, model="llama3", temperature=0).strip()
        interp_mistral = call_llm(prompt, model="mistral", temperature=0).strip()

        print("  Phi-3:   ", interp_phi3)
        print("  Llama3:  ", interp_llama)
        print("  Mistral: ", interp_mistral)

        # ── Pairwise similarities ───────────────
        sim_12 = get_similarity(interp_phi3, interp_llama)
        sim_13 = get_similarity(interp_phi3, interp_mistral)
        sim_23 = get_similarity(interp_llama, interp_mistral)

        print(f"  sim(Phi3, Llama3):  {sim_12:.3f}")
        print(f"  sim(Phi3, Mistral): {sim_13:.3f}")
        print(f"  sim(Llama3, Mistral): {sim_23:.3f}")

        # ── Agreement counting ──────────────────
        agreements = 0

        if is_same_meaning(sim_12):
            agreements += 1
        if is_same_meaning(sim_13):
            agreements += 1
        if is_same_meaning(sim_23):
            agreements += 1

        print(f"  Agreements: {agreements}/3")

        # ── Decision ───────────────────────────
        if agreements >= 2:
            print("  → UNAMBIGUOUS\n")
            nui += 1
        else:
            print("  → AMBIGUOUS\n")

    Q1 = nui / nr

    print("=" * 50)
    print(f"Q1 Specificity Score: {Q1:.4f}")
    print("=" * 50)

    return round(Q1, 4)