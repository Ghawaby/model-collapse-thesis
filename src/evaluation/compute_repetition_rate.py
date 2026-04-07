from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Config — edit these if your paths differ
# ---------------------------------------------------------------------------

GENERATED_DIR = Path("data/generated")
OUTPUT_SUMMARY = Path("results/plots/recursive_metrics_summary.json")
OUTPUT_REPETITION = Path("results/evaluations/repetition_rate_summary.json")
OUTPUT_PLOT = Path("results/plots/repetition_rate_across_generations.png")

NGRAM_SIZE = 3  # trigrams

GEN0_REFERENCE_SAMPLES = Path("data/reference/gen0_reference/gen0_reference_samples.jsonl")

BASELINE_FILES: List[Tuple[int, Path]] = [
    (1, GENERATED_DIR / "generation1_eval_samples.jsonl"),
    (2, GENERATED_DIR / "generation2_eval_samples.jsonl"),
    (3, GENERATED_DIR / "generation3_eval_samples.jsonl"),
    (4, GENERATED_DIR / "generation4_eval_samples.jsonl"),
    (5, GENERATED_DIR / "generation5_eval_samples.jsonl"),
]

CORRECTED_FILES: List[Tuple[int, Path]] = [
    (1, GENERATED_DIR / "corrected_generation1_eval_samples.jsonl"),
    (2, GENERATED_DIR / "corrected_generation2_eval_samples.jsonl"),
    (3, GENERATED_DIR / "corrected_generation3_eval_samples.jsonl"),
    (4, GENERATED_DIR / "corrected_generation4_eval_samples.jsonl"),
    (5, GENERATED_DIR / "corrected_generation5_eval_samples.jsonl"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_token_ids(row: Dict[str, Any]) -> Optional[List[int]]:
    candidates = [
        "generated_completion_token_ids",
        "current_completion_token_ids",
        "completion_token_ids",
        "generated_full_token_ids",
        "current_full_token_ids",
        "full_token_ids",
        "token_ids",
        "gen0_completion_token_ids",
        "human_completion_token_ids",
    ]
    for key in candidates:
        if key in row:
            return [int(x) for x in row[key]]
    return None


def repetition_rate(token_ids: Sequence[int], n: int = NGRAM_SIZE) -> float:
    if len(token_ids) < n:
        return 0.0
    ngrams = [tuple(token_ids[i: i + n]) for i in range(len(token_ids) - n + 1)]
    total = len(ngrams)
    unique = len(set(ngrams))
    return 1.0 - (unique / total)


def compute_mean_repetition_rate(rows: List[Dict[str, Any]]) -> Optional[float]:
    rates = []
    for row in rows:
        ids = extract_token_ids(row)
        if ids:
            rates.append(repetition_rate(ids))
    if not rates:
        return None
    return sum(rates) / len(rates)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_repetition_rate(
    gen0_rate: Optional[float],
    baseline_entries: List[Dict[str, Any]],
    corrected_entries: List[Dict[str, Any]],
) -> None:
    plt.figure(figsize=(8, 5))

    x_baseline = []
    y_baseline = []
    if gen0_rate is not None:
        x_baseline.append(0)
        y_baseline.append(gen0_rate)
    for entry in baseline_entries:
        x_baseline.append(entry["generation"])
        y_baseline.append(entry["repetition_rate"])

    if x_baseline:
        plt.plot(x_baseline, y_baseline, marker="o", linewidth=2, label="Baseline recursive")

    x_corrected = []
    y_corrected = []
    if gen0_rate is not None:
        x_corrected.append(0)
        y_corrected.append(gen0_rate)
    for entry in corrected_entries:
        x_corrected.append(entry["generation"])
        y_corrected.append(entry["repetition_rate"])

    if x_corrected:
        plt.plot(x_corrected, y_corrected, marker="o", linewidth=2, linestyle="--", label="Corrected anchored")

    all_gens = sorted({x for x in x_baseline + x_corrected})
    plt.xticks(all_gens, [f"Gen{g}" for g in all_gens])

    plt.xlabel("Generation")
    plt.ylabel("Repetition rate")
    plt.title("Recursive degradation: Repetition rate across generations")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to: {OUTPUT_PLOT}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    results: Dict[str, Any] = {
        "ngram_size": NGRAM_SIZE,
        "description": (
            "Repetition rate = fraction of n-grams in generated output that are duplicates. "
            "Higher means more repetitive. Computed per sample then averaged across the corpus."
        ),
        "gen0_baseline": None,
        "baseline_recursive": [],
        "corrected_anchored": [],
    }

    # Gen0
    gen0_rate = None
    if GEN0_REFERENCE_SAMPLES.exists():
        gen0_rows = load_jsonl(GEN0_REFERENCE_SAMPLES)
        gen0_rate = compute_mean_repetition_rate(gen0_rows)
        results["gen0_baseline"] = {"generation": 0, "repetition_rate": gen0_rate}
        print(f"Gen0 reference  repetition_rate = {gen0_rate:.6f}" if gen0_rate is not None else "Gen0: could not compute")
    else:
        print(f"Gen0 reference file not found: {GEN0_REFERENCE_SAMPLES}")

    # Baseline
    print("\nBaseline recursive branch:")
    for gen_num, path in BASELINE_FILES:
        if not path.exists():
            print(f"  Gen{gen_num}: file not found, skipping")
            continue
        rows = load_jsonl(path)
        rate = compute_mean_repetition_rate(rows)
        if rate is None:
            print(f"  Gen{gen_num}: could not extract token ids")
            continue
        results["baseline_recursive"].append({"generation": gen_num, "repetition_rate": rate, "source_file": str(path)})
        print(f"  Gen{gen_num}: repetition_rate = {rate:.6f}")

    # Corrected
    print("\nCorrected anchored branch:")
    for gen_num, path in CORRECTED_FILES:
        if not path.exists():
            print(f"  Gen{gen_num}: file not found, skipping")
            continue
        rows = load_jsonl(path)
        rate = compute_mean_repetition_rate(rows)
        if rate is None:
            print(f"  Gen{gen_num}: could not extract token ids")
            continue
        results["corrected_anchored"].append({"generation": gen_num, "repetition_rate": rate, "source_file": str(path)})
        print(f"  Gen{gen_num}: repetition_rate = {rate:.6f}")

    # Save repetition summary
    OUTPUT_REPETITION.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_REPETITION.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved repetition rate summary to: {OUTPUT_REPETITION}")

    # Plot
    plot_repetition_rate(gen0_rate, results["baseline_recursive"], results["corrected_anchored"])

    # Merge into recursive_metrics_summary.json
    if OUTPUT_SUMMARY.exists():
        with OUTPUT_SUMMARY.open("r", encoding="utf-8") as f:
            summary = json.load(f)

        rep_lookup: Dict[str, Dict[int, float]] = {
            "Baseline recursive": {},
            "Corrected anchored": {},
        }

        if gen0_rate is not None:
            rep_lookup["Baseline recursive"][0] = gen0_rate
            rep_lookup["Corrected anchored"][0] = gen0_rate

        for entry in results["baseline_recursive"]:
            rep_lookup["Baseline recursive"][entry["generation"]] = entry["repetition_rate"]

        for entry in results["corrected_anchored"]:
            rep_lookup["Corrected anchored"][entry["generation"]] = entry["repetition_rate"]

        for row in summary:
            branch = row.get("branch", "")
            gen = row.get("generation")
            if branch in rep_lookup and gen in rep_lookup[branch]:
                row["repetition_rate"] = rep_lookup[branch][gen]

        with OUTPUT_SUMMARY.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print("Updated recursive_metrics_summary.json with repetition_rate column.")
    else:
        print(f"recursive_metrics_summary.json not found at {OUTPUT_SUMMARY}, skipping merge.")

    print("\nDone.")


if __name__ == "__main__":
    main()