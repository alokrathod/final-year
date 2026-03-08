# Automated SRS Generation & Evaluation System

An agentic AI pipeline that converts plain English user requirements into a complete, IEEE 830-compliant Software Requirements Specification (SRS) document — with iterative self-improvement, multi-metric quality evaluation, and AI-suggestion detection.

> Runs fully locally using **Mistral 7B via Ollama** — no cloud APIs, no data leaves your machine.

---

## What It Does

1. Takes a plain English description of a software system from the user
2. Extracts atomic requirements (Functional & Non-Functional)
3. Retrieves IEEE 830 writing style from a local PDF corpus using RAG (FAISS)
4. Generates a complete IEEE 830 SRS document
5. Evaluates it on 8 quality metrics using an LLM-as-Judge framework (Likert 1–5)
6. Iteratively improves the SRS until the composite quality score reaches the threshold
7. Detects and highlights AI-suggested requirements in the final PDF
8. Exports the final SRS as a formatted PDF with convergence plot

---

## Results

Evaluated on a Library Management System — compared against ChatGPT-5.2 and Gemini-3 outputs using the same evaluator:

| Metric               | Our System | ChatGPT-5.2 | Gemini-3 |
| -------------------- | ---------- | ----------- | -------- |
| correctness          | **4.9**    | **4.9**     | 4.8      |
| clarity              | 4.4        | **4.6**     | 4.4      |
| verifiability        | **4.7**    | 4.6         | 4.6      |
| traceability         | 4.7        | **4.9**     | 4.5      |
| completeness         | 4.6        | **4.7**     | 4.4      |
| consistency          | **4.7**    | 4.4         | 4.5      |
| structure_compliance | **5.0**    | **5.0**     | **5.0**  |
| redundancy           | **5.0**    | 4.5         | **5.0**  |
| **Average**          | **4.75**   | 4.71        | 4.60     |

**Our system (local Mistral 7B) beats the average of both frontier cloud models.**

---

## Project Structure

```
├── main.py                        # Entry point
├── agentic_runner.py              # Main pipeline orchestrator
├── pdf_exporter.py                # PDF generation with AI-suggestion highlights
├── evaluate_external_srs.py      # Comparative evaluation script
│
├── modules/
│   ├── llm.py                     # Ollama/Mistral gateway
│   ├── extractor.py               # NL → structured requirements
│   ├── generator.py               # Requirements → IEEE 830 SRS
│   ├── evaluator.py               # 8-metric LLM-as-Judge evaluation
│   ├── planner.py                 # Improvement plan generation
│   ├── retriever.py               # FAISS vector search
│   ├── pdf_processor.py           # PDF → text chunks for RAG
│   └── ai_suggestions.py         # AI-added requirement detection
│
├── agents/
│   ├── evaluation_agent.py
│   ├── extraction_agent.py
│   ├── improvement_agent.py
│   └── structuring_agent.py
│
├── data/
│   └── srs_pdfs/                  # Place IEEE SRS PDF corpus here (gitignored)
│
├── rag_index.faiss                # Auto-generated (gitignored)
├── rag_chunks.pkl                 # Auto-generated (gitignored)
├── iteration_logs.json            # Auto-generated (gitignored)
├── final_srs.pdf                  # Auto-generated (gitignored)
├── convergence_plot.png           # Auto-generated (gitignored)
│
├── external_srs_chatgpt.txt       # For comparison (gitignored)
├── external_srs_gemini.txt        # For comparison (gitignored)
├── input_requirements.txt         # For comparison script (gitignored)
│
├── requirements.txt
└── .gitignore
```

---

## Installation

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com) installed and running
- Mistral model pulled

```bash
ollama pull mistral
ollama serve
```

### Install Python dependencies

```bash
pip install -r requirements.txt
```

### requirements.txt

```
requests
faiss-cpu
sentence-transformers
scikit-learn
numpy
pypdf
reportlab
matplotlib
```

---

## Usage

### Run the main pipeline

```bash
python main.py
```

You will be prompted to enter your requirements in plain English. Example:

```
Enter user requirements:
I want to build a web-based library management system. Users should be able to
register and login using email and password. Users can search for books by title,
author, or ISBN. Users can borrow up to 3 books at a time...
```

The system will:

- Extract requirements
- Build/load the RAG index from `data/srs_pdfs/`
- Generate the initial SRS
- Iteratively evaluate and improve until composite score >= 4.7
- Export `final_srs.pdf`, `iteration_logs.json`, and `convergence_plot.png`

---

### Run comparative evaluation

Place SRS text files from other tools in the project root:

```
external_srs_chatgpt.txt
external_srs_gemini.txt
input_requirements.txt      ← the original user input used for all systems
```

Then run:

```bash
python evaluate_external_srs.py
```

Outputs:

- `srs_comparison.json` — all scores
- `srs_comparison.png` — bar chart comparison

---

## How It Works

### Pipeline (6 stages)

```
User Input (plain text)
        │
        ▼
[1] Extractor         →  Atomic requirements list (JSON)
        │
        ▼
[2] RAG Retriever     →  IEEE 830 style/format context (FAISS)
        │
        ▼
[3] Generator         →  Full IEEE 830 SRS document
        │
        ▼
[4] Evaluator         →  8 Likert scores (3-run averaged)
        │
        ▼
[5] Improvement Loop  →  Plan → Improve → Re-evaluate (max 5 iterations)
   (if score < 4.7)        ↑___________________|
        │
        ▼
[6] Export            →  PDF + iteration_logs.json + convergence_plot.png
```

### Evaluation Metrics (Likert 1–5)

Aligned with the Krishna et al. (2024) IEEE SRS evaluation framework:

| Metric               | Type            | Weight | Description                                   |
| -------------------- | --------------- | ------ | --------------------------------------------- |
| correctness          | Per-requirement | 0.20   | Faithful to user input, no wrong facts        |
| clarity              | Per-requirement | 0.15   | Unambiguous, one interpretation only          |
| verifiability        | Per-requirement | 0.15   | Can be objectively tested                     |
| traceability         | Per-requirement | 0.15   | Each FR traces to user input                  |
| completeness         | Document-wide   | 0.15   | All user needs covered, no gaps               |
| consistency          | Document-wide   | 0.10   | No contradictions between requirements        |
| structure_compliance | Document-wide   | 0.05   | All IEEE 830 sections present                 |
| redundancy           | Document-wide   | 0.05   | No duplicate requirements (cosine similarity) |

- **7 metrics** scored by LLM-as-Judge (3 runs averaged to reduce variance)
- **Redundancy** scored by embedding-based cosine similarity (more reliable than LLM for near-duplicates)
- **Composite threshold**: weighted average >= 4.7 to stop iteration

### RAG — Style Only, No Domain Contamination

RAG retrieves only **formatting and writing style** from the IEEE SRS corpus — never domain content. A two-stage sanitization process:

1. FAISS retrieves top-3 chunks using a fixed style query (`"IEEE 830 SRS format writing style requirements structure"`)
2. A separate LLM call strips all domain content and extracts only structural patterns (heading formats, numbering conventions, "shall" syntax)

This prevents the generator from picking up irrelevant content from unrelated SRS documents in the corpus.

### AI-Suggested Requirements Detection

Requirements added by the LLM (not explicitly stated by the user) are automatically detected and highlighted in the PDF:

1. **Cosine similarity filter** — any generated requirement with max similarity < 0.60 to all user requirements is flagged
2. **LLM relevance judge** — flags each candidate as RELEVANT (e.g. implied best practice) or IRRELEVANT
3. Confirmed relevant ones are bolded with a `[AI-SUGGESTED]` label in the exported PDF

### Score Regression Prevention

Each iteration tracks the best-scoring SRS. If a new iteration scores lower than the previous best, the system automatically reverts to the best version before generating the next improvement plan. The final exported document is always the highest-scoring version seen across all iterations.

---

## Configuration

Key constants in `agentic_runner.py`:

```python
COMPOSITE_THRESHOLD = 4.7   # Stop iterating when weighted average reaches this
max_iterations = 5           # Maximum improvement iterations

METRIC_WEIGHTS = {
    "correctness":          0.20,
    "clarity":              0.15,
    "verifiability":        0.15,
    "traceability":         0.15,
    "completeness":         0.15,
    "consistency":          0.10,
    "structure_compliance": 0.05,
    "redundancy":           0.05,
}
```

LLM settings in `modules/llm.py`:

```python
MODEL_NAME = "mistral"                              # Any Ollama model
OLLAMA_URL = "http://localhost:11434/api/generate"  # Ollama server URL
```

---

## Output Files

| File                   | Description                                         |
| ---------------------- | --------------------------------------------------- |
| `final_srs.pdf`        | Complete IEEE 830 SRS with [AI-SUGGESTED] labels    |
| `final_srs.txt`        | Plain text version of the final SRS                 |
| `iteration_logs.json`  | Per-iteration scores and composite for all runs     |
| `convergence_plot.png` | Line chart of all 8 metric scores across iterations |
| `rag_index.faiss`      | Saved FAISS index (auto-built on first run)         |
| `rag_chunks.pkl`       | Saved text chunks corresponding to the index        |
| `srs_comparison.png`   | Bar chart comparing all three systems (if ran)      |
| `srs_comparison.json`  | Raw comparison scores (if ran)                      |

---

## Research Context

This system extends the **manual LLM-based SRS validation approach of Krishna et al. (2024)** into a fully automated, iterative evaluation framework:

| Dimension       | Krishna et al. (2024)    | Our System                        |
| --------------- | ------------------------ | --------------------------------- |
| Evaluators      | 4 human experts          | Automated LLM judge               |
| Metrics         | 8 (includes conciseness) | 8 (includes structure compliance) |
| Scoring         | Unweighted mean          | Weighted composite                |
| Runs            | 4 raters, one-time       | 3 LLM runs, averaged              |
| Feedback loop   | None                     | Iterative (up to 5 rounds)        |
| Scalability     | Manual, hours            | Automated, minutes                |
| Reproducibility | Low (human variance)     | High (deterministic prompt)       |

---

## License

MIT License — free to use, modify, and distribute.
