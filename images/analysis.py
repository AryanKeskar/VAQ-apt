import json
import os
import re
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "image_resize" / "experiment_results"


def extract_cosine_similarities(results):
    """Return all cosine similarity values from a JSON experiment payload."""
    similarities = []
    if not isinstance(results, list):
        return similarities

    for item in results:
        evaluations = item.get("evaluations", [])
        if not isinstance(evaluations, list):
            continue

        for evaluation in evaluations:
            value = evaluation.get("cosine similarity")
            if value is not None:
                try:
                    similarities.append(float(value))
                except (TypeError, ValueError):
                    continue
    return similarities


def parse_experiment_filename(filename):
    """Parse names like context_experiment_USER_124_PATCH_288x288.json."""
    match = re.match(r"context_experiment_USER_(\d+)_PATCH_(\d+)x(\d+)\.json$", filename)
    if match is None:
        return None, None
    patch_size = int(match.group(1))
    context_size = int(match.group(2))
    return patch_size, context_size


def load_summary(results_dir=RESULTS_DIR):
    """Aggregate mean cosine similarity by patch size and context crop size."""
    summary = {}
    file_count = 0

    for json_path in sorted(Path(results_dir).glob("*.json")):
        patch_size, context_size = parse_experiment_filename(json_path.name)
        if patch_size is None or context_size is None:
            continue

        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        values = extract_cosine_similarities(data)
        if not values:
            continue

        summary.setdefault(patch_size, {}).setdefault(context_size, []).extend(values)
        file_count += 1

    averaged_summary = {
        patch_size: {
            context_size: sum(values) / len(values)
            for context_size, values in context_data.items()
        }
        for patch_size, context_data in summary.items()
    }

    return averaged_summary, file_count


def print_summary(summary):
    """Pretty-print aggregated results."""
    print("\nAverage cosine similarity by patch size and context crop size:\n")
    print(f"{'patch_size':>10} {'context_size':>12} {'mean_similarity':>15}")
    print("-" * 42)

    for patch_size in sorted(summary):
        for context_size in sorted(summary[patch_size]):
            mean_similarity = summary[patch_size][context_size]
            print(f"{patch_size:>10} {context_size:>12} {mean_similarity:>15.4f}")


def plot_summary(summary, output_path=RESULTS_DIR / "zoom_context_similarity_summary.png"):
    """Plot mean cosine similarity curves for each patch size against context crop size."""
    if plt is None:
        raise ImportError("matplotlib is required to create the comparison plot.")

    all_context_sizes = sorted({context_size for patch_data in summary.values() for context_size in patch_data})
    patch_sizes = sorted(summary)

    fig, ax = plt.subplots(figsize=(10, 6))

    for patch_size in patch_sizes:
        xs = []
        ys = []
        for context_size in all_context_sizes:
            if context_size in summary[patch_size]:
                xs.append(context_size)
                ys.append(summary[patch_size][context_size])

        if xs:
            ax.plot(xs, ys, marker="o", linewidth=2, label=f"Zoom patch {patch_size}")

    ax.set_title("Average cosine similarity vs context crop size")
    ax.set_xlabel("Context crop size (pixels)")
    ax.set_ylabel("Mean cosine similarity")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(title="Patch size")
    ax.set_xticks(all_context_sizes)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"\nSaved plot to: {output_path}")


def main():
    summary, file_count = load_summary()
    if not summary:
        raise ValueError(f"No experiment JSON files matched the expected naming pattern in {RESULTS_DIR}")

    print(f"Processed {file_count} experiment files.")
    print_summary(summary)

    plot_summary(summary)


if __name__ == "__main__":
    main()
