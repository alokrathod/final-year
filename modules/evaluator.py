import re
import json
import numpy as np

from modules.llm import call_llm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

REDUNDANCY_THRESHOLD = 0.88

# Likert scale reference (matches Krishna et al. 2024)
LIKERT_SCALE = {
    1: "Strongly Disagree - Falls far below expected standards",
    2: "Disagree - Requires significant improvement",
    3: "Neutral - Meets expected standards but misses some details",
    4: "Agree - Generally meets or slightly exceeds standards",
    5: "Strongly Agree - Excellent, fully meets or exceeds standards"
}


# ---------------------------------------------------
# Extract requirements (used for redundancy check)
# ---------------------------------------------------

def extract_srs_requirements(srs_text):
    """
    Extracts lines containing 'shall' from the SRS.
    Used for redundancy detection via cosine similarity.
    """
    lines = srs_text.split("\n")
    requirements = []
    for line in lines:
        if "shall" in line.lower():
            requirements.append(line.strip())
    return requirements


# ---------------------------------------------------
# Redundancy metric (kept formula-based)
# Cosine similarity is more reliable than LLM
# for detecting near-duplicate requirements.
# Mapped to Likert scale at the end.
# ---------------------------------------------------

def compute_redundancy(requirements):
    """
    Detects semantically redundant requirements using
    cosine similarity. Result mapped to Likert 1-5.
    """
    if len(requirements) < 2:
        return 5

    embeddings = embedding_model.encode(requirements)
    similarity_matrix = cosine_similarity(embeddings)

    redundant = 0
    visited = set()

    for i in range(len(requirements)):
        for j in range(i + 1, len(requirements)):
            if similarity_matrix[i][j] > REDUNDANCY_THRESHOLD:
                if j not in visited:
                    redundant += 1
                    visited.add(j)

    # Percentage of non-redundant requirements
    score_pct = (1 - redundant / len(requirements)) * 100

    # Map percentage to Likert 1.0-5.0 (float)
    if score_pct >= 95:
        return 5.0
    elif score_pct >= 85:
        return 4.5
    elif score_pct >= 75:
        return 4.0
    elif score_pct >= 60:
        return 3.0
    elif score_pct >= 40:
        return 2.0
    else:
        return 1.0


# ---------------------------------------------------
# Unified LLM judge — Likert scale (1-5)
# Metrics: completeness, correctness, consistency,
#          clarity, structure_compliance,
#          verifiability, traceability
# Aligned with Krishna et al. (2024) evaluation framework
# ---------------------------------------------------

def llm_judge_single(user_input, srs_text):
    """
    Single LLM evaluation call returning decimal Likert scores (1.0-5.0).
    Called multiple times and averaged by llm_judge() for stability.

    Metrics aligned with Krishna et al. (2024):
    - Per-requirement: correctness, clarity (unambiguity),
                       verifiability, traceability
    - Document-wide:   completeness, consistency,
                       structure_compliance

    Likert Scale (decimal allowed):
    1.0 = Strongly Disagree (far below standards)
    2.0 = Disagree (needs significant improvement)
    3.0 = Neutral (meets standards, misses some details)
    4.0 = Agree (meets/slightly exceeds standards)
    5.0 = Strongly Agree (excellent, fully meets standards)

    Decimal values (e.g. 3.5, 4.2) are encouraged for fine-grained
    differentiation between SRS documents.
    """

    prompt = f"""
You are a strict IEEE 830 SRS quality auditor.
Evaluate the SRS below using a continuous 5-point Likert scale.

=== LIKERT SCALE ===
1.0 = Strongly Disagree: Falls far below expected standards
2.0 = Disagree: Requires significant improvement
3.0 = Neutral: Meets expected standards but misses some details
4.0 = Agree: Generally meets or slightly exceeds standards
5.0 = Strongly Agree: Excellent, fully meets or exceeds standards

IMPORTANT: Use decimal values (e.g. 3.5, 4.2, 4.7) to express
fine-grained differences. Do NOT round to whole numbers unless
the document perfectly meets or perfectly fails a criterion.
For example:
- Almost all requirements are correct but 1-2 are slightly off → 4.2
- Mostly clear but 3 requirements use vague terms → 3.6
- All sections present but one is thin on detail → 4.5

=== EVALUATION CRITERIA ===

PER-REQUIREMENT METRICS (evaluate across all requirements):

CORRECTNESS (1-5):
- Do the Functional Requirements accurately represent features
  the system must possess, faithful to the user input?
- Are Non-Functional Requirements (performance, security,
  usability, reliability, maintainability, portability)
  standard best practices for this system type?
- Penalize only requirements that are factually wrong or
  from a completely unrelated domain.
- Score 5 if all requirements are accurate and faithful.

CLARITY (1-5):
- Is each requirement written with one and only one possible
  interpretation? (unambiguity)
- Penalize vague terms like "fast", "user-friendly",
  "adequate", "appropriate", "sufficient" where they are
  not qualified with measurable criteria.
- Score 5 if all requirements are clear and unambiguous.

VERIFIABILITY (1-5):
- Can each requirement be verified using finite, cost-effective
  techniques once the system is built?
- Penalize requirements that cannot be tested or measured
  objectively (e.g. "the system shall be good").
- Score 5 if all requirements are testable and measurable.

TRACEABILITY (1-5):
- Can each Functional Requirement be clearly traced back to
  a specific need stated in the user input?
- Penalize functional requirements with no clear origin
  in the user input.
- Do NOT penalize Non-Functional Requirements for traceability
  as they are implied by the domain.
- Score 5 if all functional requirements trace to user input.

DOCUMENT-WIDE METRICS:

COMPLETENESS (1-5):
- Does the SRS cover all functions, describe responses to
  all inputs, provide organizational clarity, and avoid
  placeholder text like "..."?
- Are all user-stated requirements addressed?
- Score 5 if the document is fully complete with no gaps.

CONSISTENCY (1-5):
- Are there no subsets of individual requirements that
  conflict with each other?
- Minor overlaps are NOT conflicts.
- Score 5 if no contradictions exist anywhere in the SRS.

STRUCTURE COMPLIANCE (1-5):
- Does the SRS contain all required IEEE 830 sections?
  Required: Introduction, Overall Description,
  Functional Requirements, Non-Functional Requirements,
  External Interface Requirements, System Constraints,
  Assumptions and Dependencies
- Score 5 if all 7 sections are present and well-organized.
- Deduct 1 point per missing section.

=== OUTPUT FORMAT ===

Return ONLY valid JSON with no markdown, no backticks, no explanation.
All scores MUST be decimal floats (e.g. 4.2, 3.7) — NOT integers:

{{
  "correctness": <float 1.0-5.0>,
  "clarity": <float 1.0-5.0>,
  "verifiability": <float 1.0-5.0>,
  "traceability": <float 1.0-5.0>,
  "completeness": <float 1.0-5.0>,
  "consistency": <float 1.0-5.0>,
  "structure_compliance": <float 1.0-5.0>,
  "reasoning": {{
    "correctness": "<one sentence with specific observations>",
    "clarity": "<one sentence with specific observations>",
    "verifiability": "<one sentence with specific observations>",
    "traceability": "<one sentence with specific observations>",
    "completeness": "<one sentence with specific observations>",
    "consistency": "<one sentence with specific observations>",
    "structure_compliance": "<one sentence with specific observations>"
  }}
}}

=== INPUT ===

User Requirements:
{user_input}

Generated SRS:
{srs_text}
"""

    response = call_llm(prompt, temperature=0.3)  # slight temp for variance

    # Strip markdown fences if present
    cleaned = re.sub(r"```json|```", "", response).strip()

    # Fix fraction expressions like "47 / 50" → converted to 0-5 float
    # Mistral sometimes returns scores as fractions instead of decimals
    cleaned = re.sub(
        r':\s*(\d+)\s*/\s*(\d+)',
        lambda m: f': {round(int(m.group(1)) / int(m.group(2)) * 5, 2)}',
        cleaned
    )

    try:
        data = json.loads(cleaned)
    except Exception as e:
        print(f"[Evaluator] LLM judge parse failed: {e}")
        print(response)
        return {
            "correctness": 1.0,
            "clarity": 1.0,
            "verifiability": 1.0,
            "traceability": 1.0,
            "completeness": 1.0,
            "consistency": 1.0,
            "structure_compliance": 1.0,
            "reasoning": {}
        }

    # Clamp all scores to [1.0, 5.0]
    for key in ["correctness", "clarity", "verifiability", "traceability",
                "completeness", "consistency", "structure_compliance"]:
        data[key] = min(max(float(data.get(key, 1.0)), 1.0), 5.0)

    return data


SCORE_KEYS = ["correctness", "clarity", "verifiability", "traceability",
              "completeness", "consistency", "structure_compliance"]


def llm_judge(user_input, srs_text, n_runs=3):
    """
    Runs llm_judge_single n_runs times and averages the scores.
    Averaging across multiple runs reduces LLM variance and produces
    decimal scores that differentiate between SRS documents more clearly.
    Reasoning is taken from the last run.
    """

    all_scores = {key: [] for key in SCORE_KEYS}
    last_reasoning = {}

    for run in range(n_runs):
        print(f"    [Evaluator] Run {run + 1}/{n_runs}...")
        result = llm_judge_single(user_input, srs_text)
        for key in SCORE_KEYS:
            all_scores[key].append(result.get(key, 1.0))
        last_reasoning = result.get("reasoning", {})

    # Average scores across runs, rounded to 2 decimal places
    averaged = {
        key: round(sum(vals) / len(vals), 2)
        for key, vals in all_scores.items()
    }
    averaged["reasoning"] = last_reasoning

    return averaged


# ---------------------------------------------------
# Main evaluation function
# ---------------------------------------------------

def evaluate_srs(user_input, srs_text, extracted_requirements=None):
    """
    Full evaluation pipeline returning Likert scores (1-5).

    Metrics (aligned with Krishna et al. 2024):
    Per-requirement: correctness, clarity, verifiability, traceability
    Document-wide:   completeness, consistency, structure_compliance,
                     redundancy (embedding-based, mapped to Likert)
    """

    # LLM judge for 7 metrics
    judge_result = llm_judge(user_input, srs_text)

    # Log reasoning for transparency
    reasoning = judge_result.get("reasoning", {})
    if reasoning:
        print("\n  [Evaluator Reasoning]")
        for metric, reason in reasoning.items():
            print(f"    {metric}: {reason}")

    # Redundancy via cosine similarity mapped to Likert
    requirements = extract_srs_requirements(srs_text)
    redundancy_likert = compute_redundancy(requirements)

    scores = {
        # Per-requirement metrics
        "correctness":          judge_result["correctness"],
        "clarity":              judge_result["clarity"],
        "verifiability":        judge_result["verifiability"],
        "traceability":         judge_result["traceability"],
        # Document-wide metrics
        "completeness":         judge_result["completeness"],
        "consistency":          judge_result["consistency"],
        "structure_compliance": judge_result["structure_compliance"],
        "redundancy":           redundancy_likert,
    }

    return scores