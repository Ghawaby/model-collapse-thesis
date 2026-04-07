from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot recursive-generation metrics across generations."
    )

    parser.add_argument(
        "--baseline-files",
        nargs="*",
        default=[],
        help=(
            "List of evaluation JSON files for the baseline recursive branch. "
            "If omitted, the script auto-discovers gen0_fresh_greedy_check.json and generation*_vs_gen0.json."
        ),
    )
    parser.add_argument(
        "--corrected-files",
        nargs="*",
        default=[],
        help=(
            "Optional list of evaluation JSON files for the corrected branch. "
            "Use later when you run the corrected experiment."
        ),
    )
    parser.add_argument(
        "--evaluations-dir",
        type=str,
        default="results/evaluations",
        help="Directory where evaluation JSON files are stored.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/plots",
        help="Directory where plots will be saved.",
    )
    parser.add_argument(
        "--baseline-label",
        type=str,
        default="Baseline recursive",
        help="Legend label for baseline branch.",
    )
    parser.add_argument(
        "--corrected-label",
        type=str,
        default="Corrected anchored",
        help="Legend label for corrected branch.",
    )
    parser.add_argument(
        "--title-prefix",
        type=str,
        default="Recursive degradation",
        help="Prefix used in figure titles.",
    )

    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_generation_number(payload: Dict[str, Any], path: Path) -> int:
    generation_name = str(payload.get("generation_name", "")).lower()

    if "gen0" in generation_name:
        return 0

    match = re.search(r"generation[_ ]?(\d+)|generation(\d+)", generation_name)
    if match:
        groups = match.groups()
        for g in groups:
            if g is not None:
                return int(g)

    stem = path.stem.lower()

    if "gen0" in stem:
        return 0

    match = re.search(r"generation[_\-]?(\d+)", stem)
    if match:
        return int(match.group(1))

    raise ValueError(f"Could not infer generation number from file: {path}")


def generation_label(gen_num: int) -> str:
    return f"Gen{gen_num}"


def auto_discover_baseline_files(evaluations_dir: Path) -> List[Path]:
    candidates: List[Path] = []

    gen0_file = evaluations_dir / "gen0_fresh_greedy_check.json"
    if gen0_file.exists():
        candidates.append(gen0_file)

    for path in sorted(evaluations_dir.glob("generation*_vs_gen0.json")):
        candidates.append(path)

    return candidates


def load_series(files: Sequence[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for path in files:
        payload = load_json(path)
        gen_num = infer_generation_number(payload, path)

        metrics = payload.get("metrics", {})
        current = metrics.get("current", {})

        rows.append(
            {
                "generation": gen_num,
                "label": generation_label(gen_num),
                "perplexity": float(current.get("perplexity", 0.0)),
                "mauve": float(current.get("mauve", 0.0)),
                "tail_retention": float(current.get("tail_token_rate", 0.0)),
                "distinct_2": float(current.get("distinct_2", 0.0)),
                "source_file": str(path),
            }
        )

    rows.sort(key=lambda x: x["generation"])
    return rows


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_metric(
    baseline_rows: Sequence[Dict[str, Any]],
    corrected_rows: Sequence[Dict[str, Any]],
    metric_key: str,
    metric_label: str,
    title: str,
    output_path: Path,
    baseline_label: str,
    corrected_label: str,
) -> None:
    plt.figure(figsize=(8, 5))

    if baseline_rows:
        x_baseline = [row["generation"] for row in baseline_rows]
        y_baseline = [row[metric_key] for row in baseline_rows]
        plt.plot(
            x_baseline,
            y_baseline,
            marker="o",
            linewidth=2,
            label=baseline_label,
        )

    if corrected_rows:
        x_corrected = [row["generation"] for row in corrected_rows]
        y_corrected = [row[metric_key] for row in corrected_rows]
        plt.plot(
            x_corrected,
            y_corrected,
            marker="o",
            linewidth=2,
            linestyle="--",
            label=corrected_label,
        )

    all_rows = list(baseline_rows) + list(corrected_rows)
    if all_rows:
        generations = sorted({row["generation"] for row in all_rows})
        plt.xticks(generations, [generation_label(g) for g in generations])

    plt.xlabel("Generation")
    plt.ylabel(metric_label)
    plt.title(title)
    plt.grid(True, alpha=0.3)

    if baseline_rows or corrected_rows:
        plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_summary_table(
    baseline_rows: Sequence[Dict[str, Any]],
    corrected_rows: Sequence[Dict[str, Any]],
    output_path: Path,
    baseline_label: str,
    corrected_label: str,
) -> None:
    rows: List[Dict[str, Any]] = []

    for row in baseline_rows:
        rows.append(
            {
                "branch": baseline_label,
                "generation": row["generation"],
                "generation_label": row["label"],
                "perplexity": row["perplexity"],
                "mauve": row["mauve"],
                "tail_retention": row["tail_retention"],
                "distinct_2": row["distinct_2"],
                "source_file": row["source_file"],
            }
        )

    for row in corrected_rows:
        rows.append(
            {
                "branch": corrected_label,
                "generation": row["generation"],
                "generation_label": row["label"],
                "perplexity": row["perplexity"],
                "mauve": row["mauve"],
                "tail_retention": row["tail_retention"],
                "distinct_2": row["distinct_2"],
                "source_file": row["source_file"],
            }
        )

    rows.sort(key=lambda x: (x["branch"], x["generation"]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()

    evaluations_dir = Path(args.evaluations_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    if args.baseline_files:
        baseline_files = [Path(p) for p in args.baseline_files]
    else:
        baseline_files = auto_discover_baseline_files(evaluations_dir)

    corrected_files = [Path(p) for p in args.corrected_files] if args.corrected_files else []

    for path in baseline_files + corrected_files:
        if not path.exists():
            raise FileNotFoundError(f"Missing evaluation file: {path}")

    baseline_rows = load_series(baseline_files)
    corrected_rows = load_series(corrected_files)

    # Prepend shared Gen0 point to corrected branch so both lines start at Gen0
    if corrected_rows:
        gen0_entry = next((r for r in baseline_rows if r["generation"] == 0), None)
        if gen0_entry and not any(r["generation"] == 0 for r in corrected_rows):
            corrected_rows = [gen0_entry] + corrected_rows

    if not baseline_rows and not corrected_rows:
        raise ValueError("No evaluation files found to plot.")

    plot_metric(
        baseline_rows=baseline_rows,
        corrected_rows=corrected_rows,
        metric_key="perplexity",
        metric_label="Perplexity",
        title=f"{args.title_prefix}: Perplexity across generations",
        output_path=output_dir / "perplexity_across_generations.png",
        baseline_label=args.baseline_label,
        corrected_label=args.corrected_label,
    )

    plot_metric(
        baseline_rows=baseline_rows,
        corrected_rows=corrected_rows,
        metric_key="mauve",
        metric_label="MAUVE",
        title=f"{args.title_prefix}: MAUVE across generations",
        output_path=output_dir / "mauve_across_generations.png",
        baseline_label=args.baseline_label,
        corrected_label=args.corrected_label,
    )

    plot_metric(
        baseline_rows=baseline_rows,
        corrected_rows=corrected_rows,
        metric_key="tail_retention",
        metric_label="Tail retention",
        title=f"{args.title_prefix}: Tail retention across generations",
        output_path=output_dir / "tail_retention_across_generations.png",
        baseline_label=args.baseline_label,
        corrected_label=args.corrected_label,
    )

    plot_metric(
        baseline_rows=baseline_rows,
        corrected_rows=corrected_rows,
        metric_key="distinct_2",
        metric_label="Distinct-2",
        title=f"{args.title_prefix}: Distinct-2 across generations",
        output_path=output_dir / "distinct2_across_generations.png",
        baseline_label=args.baseline_label,
        corrected_label=args.corrected_label,
    )

    save_summary_table(
        baseline_rows=baseline_rows,
        corrected_rows=corrected_rows,
        output_path=output_dir / "recursive_metrics_summary.json",
        baseline_label=args.baseline_label,
        corrected_label=args.corrected_label,
    )

    print("\nDone.")
    print(f"Plots saved to: {output_dir}")
    print("Files created:")
    print(f"  - {output_dir / 'perplexity_across_generations.png'}")
    print(f"  - {output_dir / 'mauve_across_generations.png'}")
    print(f"  - {output_dir / 'tail_retention_across_generations.png'}")
    print(f"  - {output_dir / 'distinct2_across_generations.png'}")
    print(f"  - {output_dir / 'recursive_metrics_summary.json'}")


if __name__ == "__main__":
    main()