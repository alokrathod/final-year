import json
import matplotlib.pyplot as plt
import numpy as np

from modules.evaluator import evaluate_srs


# ---------------------------------------------------
# Evaluate a single SRS file
# ---------------------------------------------------

def evaluate_external(user_input_file, srs_file, label="SRS"):

    with open(user_input_file, "r") as f:
        user_input = f.read()

    with open(srs_file, "r") as f:
        srs_text = f.read()

    print(f"\n{'='*50}")
    print(f"  Evaluating: {label}")
    print(f"{'='*50}")

    scores = evaluate_srs(user_input, srs_text)

    print(f"\n  Results for {label}:")
    for key, value in scores.items():
        print(f"    {key}: {value}")

    return scores


# ---------------------------------------------------
# Plot comparison bar chart
# ---------------------------------------------------

def plot_comparison(results: dict, save_path="srs_comparison.png"):
    """
    results: dict of { label: scores_dict }
    e.g. {
        "Our System": {...},
        "ChatGPT-5.2":    {...},
        "Gemini-3":     {...}
    }
    Plots one subplot per metric, all in a single figure.
    """

    labels = list(results.keys())
    metrics = [
        "correctness",
        "clarity",
        "verifiability",
        "traceability",
        "completeness",
        "consistency",
        "structure_compliance",
        "redundancy",
    ]

    metric_display = {
        "correctness":          "Correctness",
        "clarity":              "Clarity\n(Unambiguity)",
        "verifiability":        "Verifiability",
        "traceability":         "Traceability",
        "completeness":         "Completeness",
        "consistency":          "Consistency",
        "structure_compliance": "Structure\nCompliance",
        "redundancy":           "Redundancy",
    }

    # Colors per system
    colors = ["#2196F3", "#FF5722", "#4CAF50"]

    n_metrics = len(metrics)
    n_cols = 4
    n_rows = (n_metrics + n_cols - 1) // n_cols  # ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.flatten()

    x = np.arange(len(labels))
    bar_width = 0.5

    for idx, metric in enumerate(metrics):

        ax = axes[idx]

        values = [results[label].get(metric, 0) for label in labels]

        bars = ax.bar(
            x,
            values,
            width=bar_width,
            color=colors[:len(labels)],
            edgecolor="white",
            linewidth=0.8
        )

        # Value labels on top of each bar
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold"
            )

        ax.set_title(metric_display[metric], fontsize=11, fontweight="bold", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim(0, 5.8)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_yticklabels(["1\nStr.\nDisagree", "2\nDisagree", "3\nNeutral", "4\nAgree", "5\nStr.\nAgree"], fontsize=7)
        ax.set_ylabel("Likert Score (1-5)", fontsize=9)
        ax.axhline(y=4.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide ALL unused subplots cleanly
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    # Add figure-level legend (avoids tight_layout conflict)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(labels))
    ]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        fontsize=12,
        title="Systems",
        title_fontsize=12,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02)
    )

    fig.suptitle(
        "SRS Quality Comparison: Our System vs ChatGPT-5.2 vs Gemini-3",
        fontsize=15,
        fontweight="bold",
        y=1.01
    )

    fig.subplots_adjust(top=0.93, bottom=0.08, hspace=0.5, wspace=0.4)
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------
# Radar / Spider chart
# ---------------------------------------------------


# ---------------------------------------------------
# Save scores to JSON
# ---------------------------------------------------

def save_comparison_json(results, save_path="srs_comparison.json"):
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Comparison scores saved as: {save_path}")


# ---------------------------------------------------
# Main
# ---------------------------------------------------

if __name__ == "__main__":

    USER_INPUT_FILE = "input_requirements.txt"

    # Load our system scores directly from iteration_logs.json
    # (already computed during the agentic pipeline — no need to re-evaluate)
    print("\n" + "="*50)
    print("  Loading: Our System (from iteration_logs.json)")
    print("="*50)
    with open("iteration_logs.json", "r") as f:
        iteration_logs = json.load(f)
    # Take scores from the last iteration (best/final scores)
    scores_ours = iteration_logs[-1]["scores"]
    print(f"\n  Results for Our System:")
    for key, value in scores_ours.items():
        print(f"    {key}: {value}")

    # Evaluate ChatGPT and Gemini SRS documents
    scores_chatgpt = evaluate_external(USER_INPUT_FILE, "external_srs_chatgpt.txt", label="ChatGPT-5.2")
    scores_gemini  = evaluate_external(USER_INPUT_FILE, "external_srs_gemini.txt",  label="Gemini-3")

    results = {
        "Our System": scores_ours,
        "ChatGPT-5.2":    scores_chatgpt,
        "Gemini-3":     scores_gemini,
    }

    # Save raw scores
    save_comparison_json(results)

    # Plot bar comparison chart
    plot_comparison(results, save_path="srs_comparison.png")

    # Print summary table
    print("\n\n===== SUMMARY TABLE =====\n")
    metrics = list(scores_ours.keys())
    header = f"{'Metric':<25} {'Our System':>12} {'ChatGPT-5.2':>12} {'Gemini-3':>12}"
    print(header)
    print("-" * len(header))
    for m in metrics:
        print(
            f"{m:<25} "
            f"{scores_ours.get(m, 0):>12.1f} "
            f"{scores_chatgpt.get(m, 0):>12.1f} "
            f"{scores_gemini.get(m, 0):>12.1f}"
        )