import json
import matplotlib.pyplot as plt

from modules.extractor import extract_requirements
from modules.generator import generate_srs
from modules.evaluator import evaluate_srs
from modules.llm import call_llm
from modules.retriever import retrieve
from modules.ai_suggestions import find_ai_suggested_requirements
from pdf_exporter import export_srs_to_pdf_with_legend
from modules.planner import generate_improvement_plan

# ---------------------------
# Average scoring 
# ---------------------------

# Likert scale: 1-5
# Threshold = 4.5 composite average — the weighted average of all metrics
# must reach this before iteration stops.
COMPOSITE_THRESHOLD = 4.5


def compute_average_score(scores):
    """
    Computes simple average Likert score (1.0 - 5.0).
    """
    if not scores:
        return 0.0
    return round(sum(scores.values()) / len(scores), 2)


# ------------------------------------------------
# RAG query — style and format only
# ------------------------------------------------

# RAG is used exclusively for IEEE 830 writing style and format patterns.
RAG_STYLE_QUERY = "IEEE 830 SRS format writing style requirements structure"


# ------------------------------------------------
# Plot convergence graph
# ------------------------------------------------

def plot_iterations(memory):

    if not memory:
        print("No data to plot.")
        return

    iterations = [entry["iteration"] for entry in memory]
    metrics = [k for k in memory[0]["scores"].keys() if k != "hallucination_count"]

    plt.figure(figsize=(10, 6))

    for metric in metrics:
        values = [entry["scores"].get(metric, 0) for entry in memory]
        plt.plot(iterations, values, marker='o', label=metric)

    composite_values = [entry.get("composite_score", 0) for entry in memory]
    plt.plot(
        iterations, composite_values,
        marker='s', linestyle='--', linewidth=2,
        label="composite (weighted)", color='black'
    )

    plt.xlabel("Iteration")
    plt.ylabel("Likert Score (1-5)")
    plt.title("Agentic SRS Improvement Convergence")
    plt.ylim(0.5, 5.5)
    plt.yticks([1, 2, 3, 4, 5], ["1\nStrongly\nDisagree", "2\nDisagree", "3\nNeutral", "4\nAgree", "5\nStrongly\nAgree"])
    plt.axhline(y=4.5, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="per-metric threshold (4.5)")
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("convergence_plot.png", dpi=150)
    plt.show()

    print("Convergence plot saved as convergence_plot.png")


# ------------------------------------------------
# Save logs
# ------------------------------------------------

def save_iteration_log(memory):
    with open("iteration_logs.json", "w") as f:
        json.dump(memory, f, indent=4)
    print("Iteration logs saved as iteration_logs.json")


# ------------------------------------------------
# Improve SRS
# ------------------------------------------------

def improve_srs(current_srs, improvement_plan, passing_metrics=None, failing_metrics=None):

    passing_str = ", ".join(passing_metrics) if passing_metrics else "none"
    failing_str = ", ".join(f"{k} ({v}/5.0)" for k, v in failing_metrics.items()) if failing_metrics else "none"

    prompt = f"""
You are a requirements engineering expert doing a TARGETED improvement of an SRS document.

=== IMPROVEMENT FOCUS ===
FAILING metrics that need improvement: {failing_str}
PASSING metrics — DO NOT touch anything that affects these: {passing_str}

=== IMPROVEMENT PLAN ===
{improvement_plan}

=== CURRENT SRS ===
{current_srs}

=== STRICT RULES ===
1. ONLY fix the specific issues identified in the improvement plan for FAILING metrics.
2. Do NOT rewrite sections that are already working well.
3. Do NOT invent new requirements or features.
4. Preserve all requirement IDs (FR-1, NFR-1, etc.) exactly.
5. Preserve the full IEEE 830 structure — do not remove any sections.
6. Make the minimum changes necessary to address the failing metrics.
7. If a requirement is already clear, verifiable, and correct — leave it EXACTLY as is.

Return the FULL improved SRS only. No explanation, no commentary.
"""
    # Low temperature = deterministic, conservative edits only
    return call_llm(prompt, temperature=0.1)


# ------------------------------------------------
# Agentic pipeline
# ------------------------------------------------

def agentic_pipeline(
    user_input,
    index=None,
    chunks=None,
    max_iterations=5,
    composite_threshold=4.5
):

    memory = []
    iteration = 1

    # Step 1: Extract atomic requirements
    extracted = extract_requirements(user_input)

    print(f"\nExtracted {len(extracted)} requirements:\n")

    for req in extracted:
        print(f"{req['id']}: {req['text']}")

    if not extracted:
        print("⚠️  Warning: Extraction failed or returned empty. Proceeding with raw user input as fallback.")
        extracted = [{"id": "REQ-001", "text": user_input}]
    else:
        print(f"\nExtracted {len(extracted)} atomic requirements.")

    # Step 2: RAG context retrieval 
    rag_context = None
    if index and chunks:
        print(f"RAG query (style/format): {RAG_STYLE_QUERY}")
        rag_context = retrieve(RAG_STYLE_QUERY, index, chunks)
        print("RAG style context retrieved.")

    # Step 3: Generate initial SRS
    current_srs = generate_srs(extracted, rag_context)
    print("\n===== INITIAL GENERATED SRS =====\n")
    print(current_srs)

    # Step 4: Iterative improvement loop
    # Track best SRS so far — revert if a new iteration scores lower
    best_srs = current_srs
    best_score = -1

    while iteration <= max_iterations:

        print(f"\n{'='*50}")
        print(f"  Iteration {iteration} — Evaluation")
        print(f"{'='*50}")

        scores = evaluate_srs(user_input, current_srs, extracted_requirements=extracted)
        avg_score = compute_average_score(scores)

        memory.append({
            "iteration": iteration,
            "scores": scores,
            "avg_score": avg_score
        })

        print("\nScores:")
        for k, v in scores.items():
            print(f"  {k}: {v}")
        print(f"\n  Composite (weighted Likert): {avg_score} / 5.0")

        # --- Best score tracking: revert to best if regression detected ---
        if avg_score > best_score:
            best_score = avg_score
            best_srs = current_srs
            print(f"  ✔  New best score: {best_score}/5.0 — saved.")
        else:
            print(f"  ⚠️  Score regressed ({avg_score} < {best_score}). Reverting to best SRS.")
            current_srs = best_srs

        # Check: weighted composite average must be >= composite_threshold
        if avg_score >= composite_threshold:
            print(f"\n✅ Average threshold ({composite_threshold}/5.0) reached at iteration {iteration}.")
            break

        print(f"\n  Average score {avg_score}/5.0 — needs +{round(composite_threshold - avg_score, 2)} to reach threshold ({composite_threshold}).")

        # Identify passing vs failing metrics to guide targeted improvement
        failing_metrics = {k: v for k, v in scores.items() if v < composite_threshold}
        passing_metrics = [k for k, v in scores.items() if v >= composite_threshold]

        if failing_metrics:
            print(f"  Weakest metrics (will target for improvement):")
            for k, v in sorted(failing_metrics.items(), key=lambda x: x[1]):
                print(f"    ✗  {k}: {v}/5.0")

        plan = generate_improvement_plan(scores, current_srs, user_input)
        print("\n===== IMPROVEMENT PLAN =====\n")
        print(plan)

        improved = improve_srs(
            current_srs,
            plan,
            passing_metrics=passing_metrics,
            failing_metrics=failing_metrics
        )
        print(f"\n===== IMPROVED SRS (Iteration {iteration}) =====\n")
        print(improved)

        current_srs = improved
        iteration += 1

    # Always use best scoring SRS for export, not necessarily last iteration
    current_srs = best_srs

    if iteration > max_iterations:
        print(f"\n⚠️  Max iterations ({max_iterations}) reached. Best composite score: {best_score}/5.0")

    # ── Step 5: Detect AI-suggested requirements ──────────────
    from modules.evaluator import extract_srs_requirements
    srs_reqs = extract_srs_requirements(current_srs)

    ai_suggested = find_ai_suggested_requirements(
        extracted_requirements=extracted,
        srs_requirements=srs_reqs,
        user_input=user_input
    )

    if ai_suggested:
        print(f"\n📌 {len(ai_suggested)} AI-suggested requirement(s) will be highlighted in the PDF.")
    else:
        print("\n📌 No AI-suggested requirements found.")

    # ── Step 6: Export PDF with highlights ────────────────────
    save_iteration_log(memory)

    # Save final SRS as plain text for external comparison script
    with open("final_srs.txt", "w") as f:
        f.write(current_srs)
    print("Final SRS saved as final_srs.txt")

    export_srs_to_pdf_with_legend(
        current_srs,
        filename="final_srs.pdf",
        ai_suggested_reqs=ai_suggested
    )

    print("\nFinal SRS exported as final_srs.pdf")

    plot_iterations(memory)

    return current_srs, ai_suggested