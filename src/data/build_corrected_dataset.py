from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the corrected next-generation dataset from synthetic data, tail buffer, and repair pool."
    )

    parser.add_argument(
        "--reference-dir",
        type=str,
        default="data/reference/gen0_reference",
        help="Directory created by build_gen0_reference.py",
    )
    parser.add_argument(
        "--evaluation-report",
        type=str,
        required=True,
        help="JSON report created by evaluate_generation_vs_gen0.py",
    )
    parser.add_argument(
        "--current-synthetic-samples",
        type=str,
        required=True,
        help="JSONL file of current generation samples.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to write corrected training token chunks. One JSON list per line.",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default="",
        help="Optional explicit path for metadata JSON. If omitted, output_path + '.meta.json' is used.",
    )
    parser.add_argument(
        "--fallback-tokenizer",
        type=str,
        default="gpt2",
        help="Reserved for compatibility; not strictly needed here.",
    )

    parser.add_argument(
        "--target-size",
        type=int,
        default=0,
        help="Total number of training rows to output. 0 means match current synthetic row count.",
    )

    # Base fractions
    parser.add_argument("--synthetic-base-fraction", type=float, default=0.70)
    parser.add_argument("--tail-base-fraction", type=float, default=0.15)
    parser.add_argument("--repair-base-fraction", type=float, default=0.15)

    # Deficit scaling
    parser.add_argument("--tail-scale", type=float, default=0.35)
    parser.add_argument("--repair-scale-mauve", type=float, default=0.20)
    parser.add_argument("--repair-scale-distinct2", type=float, default=0.20)
    parser.add_argument("--repair-scale-ppl", type=float, default=0.10)

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )

    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl_tokens(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row["token_ids"], ensure_ascii=False) + "\n")


def get_bigrams(token_ids: Sequence[int]) -> set[Tuple[int, int]]:
    if len(token_ids) < 2:
        return set()
    return set((token_ids[i], token_ids[i + 1]) for i in range(len(token_ids) - 1))


def unique_bigram_count(token_ids: Sequence[int]) -> int:
    return len(get_bigrams(token_ids))


def normalize_current_synthetic_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts current synthetic sample rows into training rows with token_ids/text.
    Prefer full sequence ids for training, but also keep completion ids for repair analysis.
    """
    normalized: List[Dict[str, Any]] = []

    full_id_candidates = [
        "generated_full_token_ids",
        "gen0_full_token_ids",
        "current_full_token_ids",
        "human_full_token_ids",
        "full_token_ids",
        "token_ids",
    ]

    completion_id_candidates = [
        "generated_completion_token_ids",
        "gen0_completion_token_ids",
        "current_completion_token_ids",
        "human_completion_token_ids",
        "completion_token_ids",
    ]

    text_candidates = [
        "generated_full_text",
        "gen0_full_text",
        "current_full_text",
        "human_full_text",
        "full_text",
        "text",
    ]

    completion_text_candidates = [
        "generated_completion_text",
        "gen0_completion_text",
        "current_completion_text",
        "human_completion_text",
        "completion_text",
    ]

    for row in rows:
        full_ids = None
        for key in full_id_candidates:
            if key in row:
                full_ids = [int(x) for x in row[key]]
                break

        completion_ids = None
        for key in completion_id_candidates:
            if key in row:
                completion_ids = [int(x) for x in row[key]]
                break

        text = None
        for key in text_candidates:
            if key in row:
                text = row[key]
                break

        completion_text = None
        for key in completion_text_candidates:
            if key in row:
                completion_text = row[key]
                break

        if full_ids is None and completion_ids is None:
            raise ValueError("Synthetic row has neither full ids nor completion ids.")

        # If full sequence is missing, fall back to completion ids for training too.
        if full_ids is None:
            full_ids = completion_ids

        normalized.append(
            {
                "source": "synthetic",
                "prompt_id": row.get("prompt_id"),
                "token_ids": full_ids,
                "completion_token_ids": completion_ids if completion_ids is not None else full_ids,
                "text": text if text is not None else completion_text,
            }
        )

    return normalized


def normalize_human_pool_rows(rows: Sequence[Dict[str, Any]], source_name: str) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []

    for row in rows:
        token_ids = row.get("token_ids")
        if token_ids is None:
            raise ValueError(f"{source_name} row missing token_ids")

        normalized.append(
            {
                "source": source_name,
                "chunk_index": row.get("chunk_index"),
                "token_ids": [int(x) for x in token_ids],
                "text": row.get("text", ""),
                "tail_count": row.get("tail_count", 0),
                "tail_density": row.get("tail_density", 0.0),
                "tail_token_ids_present": row.get("tail_token_ids_present", []),
            }
        )

    return normalized


def normalize_reference_completion_rows(rows: Sequence[Dict[str, Any]], source_name: str) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []

    id_candidates = [
        "gen0_completion_token_ids",
        "generated_completion_token_ids",
        "completion_token_ids",
        "human_completion_token_ids",
    ]

    for row in rows:
        ids = None
        for key in id_candidates:
            if key in row:
                ids = [int(x) for x in row[key]]
                break

        if ids is None:
            raise ValueError(f"{source_name} row missing completion token ids")

        normalized.append(
            {
                "source": source_name,
                "prompt_id": row.get("prompt_id"),
                "completion_token_ids": ids,
            }
        )

    return normalized


def compute_missing_items_relative_to_gen0(
    gen0_reference_rows: Sequence[Dict[str, Any]],
    current_synthetic_rows: Sequence[Dict[str, Any]],
    tail_set: set[int],
) -> Dict[str, Any]:
    gen0_completion_ids = [row["completion_token_ids"] for row in gen0_reference_rows]
    current_completion_ids = [row["completion_token_ids"] for row in current_synthetic_rows]

    gen0_tail_tokens = set()
    current_tail_tokens = set()

    for seq in gen0_completion_ids:
        for tok in seq:
            if tok in tail_set:
                gen0_tail_tokens.add(tok)

    for seq in current_completion_ids:
        for tok in seq:
            if tok in tail_set:
                current_tail_tokens.add(tok)

    missing_tail_tokens = gen0_tail_tokens - current_tail_tokens

    gen0_bigrams = set()
    current_bigrams = set()

    for seq in gen0_completion_ids:
        gen0_bigrams |= get_bigrams(seq)

    for seq in current_completion_ids:
        current_bigrams |= get_bigrams(seq)

    missing_bigrams = gen0_bigrams - current_bigrams

    return {
        "missing_tail_tokens": missing_tail_tokens,
        "missing_bigrams": missing_bigrams,
        "gen0_tail_token_count": len(gen0_tail_tokens),
        "current_tail_token_count": len(current_tail_tokens),
        "gen0_bigram_count": len(gen0_bigrams),
        "current_bigram_count": len(current_bigrams),
    }


def score_repair_row(
    row: Dict[str, Any],
    missing_tail_tokens: set[int],
    missing_bigrams: set[Tuple[int, int]],
) -> Dict[str, Any]:
    token_ids = row["token_ids"]

    row_tokens = set(token_ids)
    row_bigrams = get_bigrams(token_ids)

    missing_tail_coverage = len(row_tokens & missing_tail_tokens)
    missing_bigram_coverage = len(row_bigrams & missing_bigrams)
    diversity_gain = unique_bigram_count(token_ids)
    tail_density = float(row.get("tail_density", 0.0))

    score = (
        3.0 * missing_tail_coverage
        + 1.5 * missing_bigram_coverage
        + 0.05 * diversity_gain
        + 2.0 * tail_density
    )

    enriched = dict(row)
    enriched["repair_score"] = score
    enriched["missing_tail_coverage"] = missing_tail_coverage
    enriched["missing_bigram_coverage"] = missing_bigram_coverage
    enriched["diversity_gain"] = diversity_gain
    return enriched


def sample_or_repeat(rows: Sequence[Dict[str, Any]], k: int, rng: random.Random) -> List[Dict[str, Any]]:
    if k <= 0:
        return []

    rows = list(rows)
    if not rows:
        return []

    if k <= len(rows):
        return rng.sample(rows, k)

    result = list(rows)
    while len(result) < k:
        result.append(rng.choice(rows))
    return result[:k]


def top_or_repeat(rows: Sequence[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    if k <= 0:
        return []

    rows = list(rows)
    if not rows:
        return []

    if k <= len(rows):
        return rows[:k]

    result = list(rows)
    idx = 0
    while len(result) < k:
        result.append(rows[idx % len(rows)])
        idx += 1
    return result[:k]


def compute_component_counts(
    target_size: int,
    ppl_deficit: float,
    mauve_deficit: float,
    tail_deficit: float,
    distinct2_deficit: float,
    synthetic_base_fraction: float,
    tail_base_fraction: float,
    repair_base_fraction: float,
    tail_scale: float,
    repair_scale_mauve: float,
    repair_scale_distinct2: float,
    repair_scale_ppl: float,
) -> Dict[str, Any]:
    synth_weight = synthetic_base_fraction
    tail_weight = tail_base_fraction + tail_scale * tail_deficit
    repair_weight = (
        repair_base_fraction
        + repair_scale_mauve * mauve_deficit
        + repair_scale_distinct2 * distinct2_deficit
        + repair_scale_ppl * ppl_deficit
    )

    total_weight = synth_weight + tail_weight + repair_weight
    if total_weight <= 0:
        synth_weight, tail_weight, repair_weight = 0.70, 0.15, 0.15
        total_weight = 1.0

    synth_fraction = synth_weight / total_weight
    tail_fraction = tail_weight / total_weight
    repair_fraction = repair_weight / total_weight

    synth_count = int(round(target_size * synth_fraction))
    tail_count = int(round(target_size * tail_fraction))
    repair_count = target_size - synth_count - tail_count

    # Small correction if rounding causes issues
    if repair_count < 0:
        repair_count = 0
        overflow = synth_count + tail_count - target_size
        if overflow > 0:
            synth_count = max(0, synth_count - overflow)

    return {
        "weights": {
            "synthetic_weight": synth_weight,
            "tail_weight": tail_weight,
            "repair_weight": repair_weight,
        },
        "fractions": {
            "synthetic_fraction": synth_fraction,
            "tail_fraction": tail_fraction,
            "repair_fraction": repair_fraction,
        },
        "counts": {
            "synthetic_count": synth_count,
            "tail_count": tail_count,
            "repair_count": repair_count,
        },
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    reference_dir = Path(args.reference_dir)
    evaluation_report_path = Path(args.evaluation_report)
    current_synthetic_path = Path(args.current_synthetic_samples)
    output_path = Path(args.output_path)

    metadata_path = Path(args.metadata_path) if args.metadata_path else Path(str(output_path) + ".meta.json")

    if not reference_dir.exists():
        raise FileNotFoundError(f"Reference dir not found: {reference_dir}")
    if not evaluation_report_path.exists():
        raise FileNotFoundError(f"Evaluation report not found: {evaluation_report_path}")
    if not current_synthetic_path.exists():
        raise FileNotFoundError(f"Current synthetic samples not found: {current_synthetic_path}")

    print("Loading evaluation report...")
    evaluation_report = load_json(evaluation_report_path)

    print("Loading reference resources...")
    tail_tokens_payload = load_json(reference_dir / "tail_tokens.json")
    tail_buffer_raw = load_jsonl(reference_dir / "tail_sample_buffer.jsonl")
    repair_pool_raw = load_jsonl(reference_dir / "repair_pool.jsonl")
    gen0_reference_raw = load_jsonl(reference_dir / "gen0_reference_samples.jsonl")

    print("Loading current synthetic samples...")
    current_synthetic_raw = load_jsonl(current_synthetic_path)

    tail_set = set(int(entry["token_id"]) for entry in tail_tokens_payload["tail_tokens"])

    current_synthetic_rows = normalize_current_synthetic_rows(current_synthetic_raw)
    tail_buffer_rows = normalize_human_pool_rows(tail_buffer_raw, "tail_buffer")
    repair_pool_rows = normalize_human_pool_rows(repair_pool_raw, "repair_pool")
    gen0_reference_rows = normalize_reference_completion_rows(gen0_reference_raw, "gen0_reference")

    if args.target_size > 0:
        target_size = args.target_size
    else:
        target_size = len(current_synthetic_rows)

    deficits = evaluation_report.get("deficits", {})
    ppl_deficit = float(deficits.get("ppl_deficit", 0.0))
    mauve_deficit = float(deficits.get("mauve_deficit", 0.0))
    tail_deficit = float(deficits.get("tail_deficit", 0.0))
    distinct2_deficit = float(deficits.get("distinct2_deficit", 0.0))

    print("Computing underrepresented items relative to Gen0...")
    missing_info = compute_missing_items_relative_to_gen0(
        gen0_reference_rows=gen0_reference_rows,
        current_synthetic_rows=current_synthetic_rows,
        tail_set=tail_set,
    )

    missing_tail_tokens = missing_info["missing_tail_tokens"]
    missing_bigrams = missing_info["missing_bigrams"]

    print("Scoring repair pool...")
    scored_repair_rows = [
        score_repair_row(row, missing_tail_tokens, missing_bigrams)
        for row in repair_pool_rows
    ]
    scored_repair_rows.sort(
        key=lambda r: (
            r["repair_score"],
            r["missing_tail_coverage"],
            r["missing_bigram_coverage"],
            r["diversity_gain"],
            -int(r.get("chunk_index", 0) or 0),
        ),
        reverse=True,
    )

    print("Computing component counts...")
    counts_info = compute_component_counts(
        target_size=target_size,
        ppl_deficit=ppl_deficit,
        mauve_deficit=mauve_deficit,
        tail_deficit=tail_deficit,
        distinct2_deficit=distinct2_deficit,
        synthetic_base_fraction=args.synthetic_base_fraction,
        tail_base_fraction=args.tail_base_fraction,
        repair_base_fraction=args.repair_base_fraction,
        tail_scale=args.tail_scale,
        repair_scale_mauve=args.repair_scale_mauve,
        repair_scale_distinct2=args.repair_scale_distinct2,
        repair_scale_ppl=args.repair_scale_ppl,
    )

    synthetic_count = counts_info["counts"]["synthetic_count"]
    tail_count = counts_info["counts"]["tail_count"]
    repair_count = counts_info["counts"]["repair_count"]

    print("Selecting synthetic rows...")
    selected_synthetic = sample_or_repeat(current_synthetic_rows, synthetic_count, rng)

    print("Selecting tail rows...")
    # Tail buffer file is already tail-oriented, so taking top rows is fine.
    tail_buffer_rows_sorted = sorted(
        tail_buffer_rows,
        key=lambda r: (
            float(r.get("tail_density", 0.0)),
            int(r.get("tail_count", 0)),
            -int(r.get("chunk_index", 0) or 0),
        ),
        reverse=True,
    )
    selected_tail = top_or_repeat(tail_buffer_rows_sorted, tail_count)

    print("Selecting repair rows...")
    selected_repair = top_or_repeat(scored_repair_rows, repair_count)

    corrected_rows: List[Dict[str, Any]] = []
    corrected_rows.extend(selected_synthetic)
    corrected_rows.extend(selected_tail)
    corrected_rows.extend(selected_repair)

    rng.shuffle(corrected_rows)

    print("Saving corrected dataset...")
    write_jsonl_tokens(output_path, corrected_rows)

    metadata = {
        "reference_dir": str(reference_dir),
        "evaluation_report": str(evaluation_report_path),
        "current_synthetic_samples": str(current_synthetic_path),
        "output_path": str(output_path),
        "target_size": target_size,
        "seed": args.seed,
        "deficits": {
            "ppl_deficit": ppl_deficit,
            "mauve_deficit": mauve_deficit,
            "tail_deficit": tail_deficit,
            "distinct2_deficit": distinct2_deficit,
        },
        "component_weights": counts_info["weights"],
        "component_fractions": counts_info["fractions"],
        "component_counts": counts_info["counts"],
        "available_pool_sizes": {
            "synthetic_rows": len(current_synthetic_rows),
            "tail_buffer_rows": len(tail_buffer_rows),
            "repair_pool_rows": len(repair_pool_rows),
        },
        "missing_relative_to_gen0": {
            "missing_tail_token_count": len(missing_tail_tokens),
            "missing_bigram_count": len(missing_bigrams),
            "gen0_tail_token_count": missing_info["gen0_tail_token_count"],
            "current_tail_token_count": missing_info["current_tail_token_count"],
            "gen0_bigram_count": missing_info["gen0_bigram_count"],
            "current_bigram_count": missing_info["current_bigram_count"],
        },
        "selected_examples": {
            "synthetic_prompt_ids_head": [
                row.get("prompt_id") for row in selected_synthetic[:10]
            ],
            "tail_chunk_indices_head": [
                row.get("chunk_index") for row in selected_tail[:10]
            ],
            "repair_chunk_indices_head": [
                row.get("chunk_index") for row in selected_repair[:10]
            ],
            "top_repair_scores_head": [
                {
                    "chunk_index": row.get("chunk_index"),
                    "repair_score": row.get("repair_score"),
                    "missing_tail_coverage": row.get("missing_tail_coverage"),
                    "missing_bigram_coverage": row.get("missing_bigram_coverage"),
                    "diversity_gain": row.get("diversity_gain"),
                }
                for row in selected_repair[:10]
            ],
        },
    }

    write_json(metadata_path, metadata)

    print("\nDone.")
    print(f"Saved corrected dataset to: {output_path}")
    print(f"Saved metadata to: {metadata_path}")
    print("Component counts:")
    print(f"  synthetic = {synthetic_count}")
    print(f"  tail = {tail_count}")
    print(f"  repair = {repair_count}")


if __name__ == "__main__":
    main()