from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a generation against the Gen0 reference package."
    )

    parser.add_argument(
        "--reference-dir",
        type=str,
        default="data/reference/gen0_reference",
        help="Directory created by build_gen0_reference.py",
    )
    parser.add_argument(
        "--current-samples",
        type=str,
        required=True,
        help="JSONL file containing current generation outputs on the fixed prompt set.",
    )
    parser.add_argument(
        "--current-checkpoint",
        type=str,
        required=True,
        help="Checkpoint of the current generation model for perplexity evaluation.",
    )
    parser.add_argument(
        "--heldout-path",
        type=str,
        default="data/processed/wikitext2_64/test_tokens_64.txt",
        help="Held-out human token chunks for perplexity evaluation.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="results/evaluations/current_vs_gen0.json",
        help="Where to save the evaluation report.",
    )
    parser.add_argument(
        "--fallback-tokenizer",
        type=str,
        default="gpt2",
        help="Used if tokenizer is not saved in checkpoint folders.",
    )
    parser.add_argument(
        "--generation-name",
        type=str,
        default="current_generation",
        help="Label stored in the output report.",
    )
    parser.add_argument(
        "--ppl-batch-size",
        type=int,
        default=32,
        help="Batch size for perplexity evaluation.",
    )
    parser.add_argument(
        "--mauve-max-text-length",
        type=int,
        default=256,
        help="Max text length passed to MAUVE.",
    )

    # Thresholds for drift flags
    parser.add_argument("--ppl-tol", type=float, default=0.05)
    parser.add_argument("--mauve-tol", type=float, default=0.05)
    parser.add_argument("--tail-tol", type=float, default=0.05)
    parser.add_argument("--distinct2-tol", type=float, default=0.05)

    return parser.parse_args()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


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
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_token_line(line: str) -> Optional[List[int]]:
    line = line.strip()
    if not line:
        return None

    if line.startswith("[") and line.endswith("]"):
        parsed = json.loads(line)
        if not isinstance(parsed, list):
            raise ValueError("Expected JSON list of token ids.")
        return [int(x) for x in parsed]

    normalized = line.replace(",", " ")
    parts = [p for p in normalized.split() if p]
    return [int(p) for p in parts]


def load_token_chunks(path: Path) -> List[List[int]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    chunks: List[List[int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            chunk = parse_token_line(line)
            if chunk is None or len(chunk) == 0:
                continue
            if not all(isinstance(x, int) for x in chunk):
                raise ValueError(f"Non-integer token id in {path} at line {line_num}")
            chunks.append(chunk)

    if not chunks:
        raise ValueError(f"No token chunks loaded from {path}")

    return chunks


def load_tokenizer(path: Path, fallback_tokenizer: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(path))
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(fallback_tokenizer)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def safe_model_load(checkpoint_path: Path, device: torch.device):
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    try:
        model = AutoModelForCausalLM.from_pretrained(str(checkpoint_path), dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(str(checkpoint_path), torch_dtype=dtype)

    model.to(device)
    model.eval()
    return model


def compute_perplexity(
    checkpoint_path: Path,
    token_chunks: Sequence[Sequence[int]],
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    model = safe_model_load(checkpoint_path, device)

    total_nll = 0.0
    total_predicted_tokens = 0

    seq_len = len(token_chunks[0])
    for chunk in token_chunks:
        if len(chunk) != seq_len:
            raise ValueError("All held-out chunks must have the same length for this evaluator.")

    for start in tqdm(range(0, len(token_chunks), batch_size), desc=f"Perplexity @ {checkpoint_path.name}"):
        batch = token_chunks[start : start + batch_size]
        input_ids = torch.tensor(batch, dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids, device=device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )

        # Causal LM loss predicts next token, so effective count is seq_len - 1 per row.
        predicted_tokens = input_ids.shape[0] * (input_ids.shape[1] - 1)
        total_nll += outputs.loss.item() * predicted_tokens
        total_predicted_tokens += predicted_tokens

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    avg_nll = total_nll / max(1, total_predicted_tokens)
    ppl = math.exp(avg_nll)

    return {
        "avg_nll": avg_nll,
        "perplexity": ppl,
        "num_chunks": len(token_chunks),
        "predicted_tokens": total_predicted_tokens,
    }


def normalize_sample_rows(rows: Sequence[Dict[str, Any]], tokenizer) -> Dict[int, Dict[str, Any]]:
    """
    Converts different possible JSONL schemas into:
    {
      prompt_id: {
        "prompt_id": int,
        "completion_token_ids": List[int],
        "completion_text": str
      }
    }
    """
    normalized: Dict[int, Dict[str, Any]] = {}

    id_candidates = [
        "completion_token_ids",
        "generated_completion_token_ids",
        "gen_completion_token_ids",
        "gen0_completion_token_ids",
        "current_completion_token_ids",
        "human_completion_token_ids",
    ]

    text_candidates = [
        "completion_text",
        "generated_completion_text",
        "gen_completion_text",
        "gen0_completion_text",
        "current_completion_text",
        "human_completion_text",
    ]

    for row in rows:
        if "prompt_id" not in row:
            continue

        prompt_id = int(row["prompt_id"])

        token_ids = None
        for key in id_candidates:
            if key in row:
                token_ids = row[key]
                break

        text = None
        for key in text_candidates:
            if key in row:
                text = row[key]
                break

        if token_ids is None and text is None:
            raise ValueError(f"Row with prompt_id={prompt_id} has neither completion ids nor completion text.")

        if token_ids is None and text is not None:
            token_ids = tokenizer.encode(text, add_special_tokens=False)

        if text is None and token_ids is not None:
            text = tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)

        normalized[prompt_id] = {
            "prompt_id": prompt_id,
            "completion_token_ids": [int(x) for x in token_ids],
            "completion_text": text,
        }

    return normalized


def align_rows(
    human_rows: Dict[int, Dict[str, Any]],
    gen0_rows: Dict[int, Dict[str, Any]],
    current_rows: Dict[int, Dict[str, Any]],
) -> Tuple[List[int], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    common_prompt_ids = sorted(set(human_rows) & set(gen0_rows) & set(current_rows))

    if not common_prompt_ids:
        raise ValueError("No overlapping prompt_ids across human, Gen0, and current generation rows.")

    human_aligned = [human_rows[p] for p in common_prompt_ids]
    gen0_aligned = [gen0_rows[p] for p in common_prompt_ids]
    current_aligned = [current_rows[p] for p in common_prompt_ids]

    return common_prompt_ids, human_aligned, gen0_aligned, current_aligned


def distinct_n_from_token_ids(token_sequences: Sequence[Sequence[int]], n: int = 2) -> float:
    total_ngrams = 0
    unique_ngrams = set()

    for seq in token_sequences:
        if len(seq) < n:
            continue
        for i in range(len(seq) - n + 1):
            ng = tuple(seq[i : i + n])
            unique_ngrams.add(ng)
            total_ngrams += 1

    if total_ngrams == 0:
        return 0.0

    return len(unique_ngrams) / total_ngrams


def compute_tail_stats(token_sequences: Sequence[Sequence[int]], tail_set: set[int]) -> Dict[str, float]:
    total_tokens = 0
    tail_hits = 0
    unique_tail_tokens_seen = set()

    for seq in token_sequences:
        total_tokens += len(seq)
        for tok in seq:
            if tok in tail_set:
                tail_hits += 1
                unique_tail_tokens_seen.add(tok)

    tail_rate = tail_hits / max(1, total_tokens)
    tail_vocab_coverage = len(unique_tail_tokens_seen) / max(1, len(tail_set))

    return {
        "total_tokens": total_tokens,
        "tail_hits": tail_hits,
        "tail_token_rate": tail_rate,
        "tail_vocab_coverage": tail_vocab_coverage,
    }


def compute_mauve_score(
    human_texts: Sequence[str],
    model_texts: Sequence[str],
    max_text_length: int,
) -> float:
    try:
        import mauve
    except ImportError as e:
        raise ImportError(
            "MAUVE is not installed. Install it with: pip install mauve-text"
        ) from e

    device_id = 0 if torch.cuda.is_available() else -1

    result = mauve.compute_mauve(
        p_text=list(human_texts),
        q_text=list(model_texts),
        device_id=device_id,
        max_text_length=max_text_length,
        verbose=False,
    )
    return float(result.mauve)


def ratio_or_none(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def normalized_deficit_higher_is_worse(current_value: float, baseline_value: float) -> float:
    """
    Example: perplexity. If current > baseline, that's degradation.
    """
    if baseline_value == 0:
        return 0.0
    return max(0.0, (current_value - baseline_value) / baseline_value)


def normalized_deficit_lower_is_worse(current_value: float, baseline_value: float) -> float:
    """
    Example: MAUVE, tail rate, Distinct-2. If current < baseline, that's degradation.
    """
    if baseline_value == 0:
        return 0.0
    return max(0.0, (baseline_value - current_value) / baseline_value)


def main() -> None:
    args = parse_args()

    reference_dir = Path(args.reference_dir)
    current_samples_path = Path(args.current_samples)
    current_checkpoint = Path(args.current_checkpoint)
    heldout_path = Path(args.heldout_path)
    output_path = Path(args.output_path)

    if not reference_dir.exists():
        raise FileNotFoundError(f"Reference dir not found: {reference_dir}")
    if not current_samples_path.exists():
        raise FileNotFoundError(f"Current samples file not found: {current_samples_path}")
    if not current_checkpoint.exists():
        raise FileNotFoundError(f"Current checkpoint not found: {current_checkpoint}")
    if not heldout_path.exists():
        raise FileNotFoundError(f"Held-out token file not found: {heldout_path}")

    config = load_json(reference_dir / "config.json")
    summary_stats = load_json(reference_dir / "summary_stats.json")
    tail_tokens_payload = load_json(reference_dir / "tail_tokens.json")

    # Get Gen0 checkpoint from reference config unless user moved things manually.
    gen0_checkpoint = Path(config["gen0_checkpoint"])
    if not gen0_checkpoint.exists():
        raise FileNotFoundError(
            f"Gen0 checkpoint recorded in config does not exist anymore: {gen0_checkpoint}"
        )

    print("Loading tokenizer...")
    tokenizer = load_tokenizer(gen0_checkpoint, args.fallback_tokenizer)

    print("Loading reference rows...")
    human_rows_raw = load_jsonl(reference_dir / "human_eval_references.jsonl")
    gen0_rows_raw = load_jsonl(reference_dir / "gen0_reference_samples.jsonl")
    current_rows_raw = load_jsonl(current_samples_path)

    human_rows = normalize_sample_rows(human_rows_raw, tokenizer)
    gen0_rows = normalize_sample_rows(gen0_rows_raw, tokenizer)
    current_rows = normalize_sample_rows(current_rows_raw, tokenizer)

    common_prompt_ids, human_aligned, gen0_aligned, current_aligned = align_rows(
        human_rows,
        gen0_rows,
        current_rows,
    )

    print(f"Aligned prompts: {len(common_prompt_ids)}")

    human_texts = [row["completion_text"] for row in human_aligned]
    gen0_texts = [row["completion_text"] for row in gen0_aligned]
    current_texts = [row["completion_text"] for row in current_aligned]

    human_ids = [row["completion_token_ids"] for row in human_aligned]
    gen0_ids = [row["completion_token_ids"] for row in gen0_aligned]
    current_ids = [row["completion_token_ids"] for row in current_aligned]

    tail_token_ids = [entry["token_id"] for entry in tail_tokens_payload["tail_tokens"]]
    tail_set = set(int(x) for x in tail_token_ids)

    print("Computing lexical / tail metrics...")
    gen0_distinct2 = distinct_n_from_token_ids(gen0_ids, n=2)
    current_distinct2 = distinct_n_from_token_ids(current_ids, n=2)

    human_tail = compute_tail_stats(human_ids, tail_set)
    gen0_tail = compute_tail_stats(gen0_ids, tail_set)
    current_tail = compute_tail_stats(current_ids, tail_set)

    print("Computing MAUVE...")
    gen0_mauve = compute_mauve_score(
        human_texts=human_texts,
        model_texts=gen0_texts,
        max_text_length=args.mauve_max_text_length,
    )
    current_mauve = compute_mauve_score(
        human_texts=human_texts,
        model_texts=current_texts,
        max_text_length=args.mauve_max_text_length,
    )

    print("Loading held-out token chunks...")
    heldout_chunks = load_token_chunks(heldout_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Computing perplexity for current generation...")
    current_ppl = compute_perplexity(
        checkpoint_path=current_checkpoint,
        token_chunks=heldout_chunks,
        batch_size=args.ppl_batch_size,
        device=device,
    )

    print("Computing perplexity for Gen0 baseline...")
    gen0_ppl = compute_perplexity(
        checkpoint_path=gen0_checkpoint,
        token_chunks=heldout_chunks,
        batch_size=args.ppl_batch_size,
        device=device,
    )

    # Drift / deficits relative to Gen0
    ppl_deficit = normalized_deficit_higher_is_worse(
        current_value=current_ppl["perplexity"],
        baseline_value=gen0_ppl["perplexity"],
    )
    mauve_deficit = normalized_deficit_lower_is_worse(
        current_value=current_mauve,
        baseline_value=gen0_mauve,
    )
    tail_deficit = normalized_deficit_lower_is_worse(
        current_value=current_tail["tail_token_rate"],
        baseline_value=gen0_tail["tail_token_rate"],
    )
    distinct2_deficit = normalized_deficit_lower_is_worse(
        current_value=current_distinct2,
        baseline_value=gen0_distinct2,
    )

    report = {
        "generation_name": args.generation_name,
        "reference_dir": str(reference_dir),
        "current_samples": str(current_samples_path),
        "current_checkpoint": str(current_checkpoint),
        "gen0_checkpoint": str(gen0_checkpoint),
        "heldout_path": str(heldout_path),
        "num_aligned_prompts": len(common_prompt_ids),
        "reference_summary_stats": summary_stats,
        "metrics": {
            "gen0": {
                "perplexity": gen0_ppl["perplexity"],
                "avg_nll": gen0_ppl["avg_nll"],
                "mauve": gen0_mauve,
                "distinct_2": gen0_distinct2,
                "tail_token_rate": gen0_tail["tail_token_rate"],
                "tail_vocab_coverage": gen0_tail["tail_vocab_coverage"],
            },
            "current": {
                "perplexity": current_ppl["perplexity"],
                "avg_nll": current_ppl["avg_nll"],
                "mauve": current_mauve,
                "distinct_2": current_distinct2,
                "tail_token_rate": current_tail["tail_token_rate"],
                "tail_vocab_coverage": current_tail["tail_vocab_coverage"],
            },
            "human_reference": {
                "tail_token_rate": human_tail["tail_token_rate"],
                "tail_vocab_coverage": human_tail["tail_vocab_coverage"],
            },
        },
        "relative_to_gen0": {
            "perplexity_ratio": ratio_or_none(current_ppl["perplexity"], gen0_ppl["perplexity"]),
            "mauve_ratio": ratio_or_none(current_mauve, gen0_mauve),
            "tail_token_rate_ratio": ratio_or_none(
                current_tail["tail_token_rate"],
                gen0_tail["tail_token_rate"],
            ),
            "distinct_2_ratio": ratio_or_none(current_distinct2, gen0_distinct2),
        },
        "deficits": {
            "ppl_deficit": ppl_deficit,
            "mauve_deficit": mauve_deficit,
            "tail_deficit": tail_deficit,
            "distinct2_deficit": distinct2_deficit,
        },
        "drift_flags": {
            "perplexity_drift": ppl_deficit > args.ppl_tol,
            "mauve_drift": mauve_deficit > args.mauve_tol,
            "tail_drift": tail_deficit > args.tail_tol,
            "distinct2_drift": distinct2_deficit > args.distinct2_tol,
        },
        "thresholds": {
            "ppl_tol": args.ppl_tol,
            "mauve_tol": args.mauve_tol,
            "tail_tol": args.tail_tol,
            "distinct2_tol": args.distinct2_tol,
        },
    }

    print("Saving evaluation report...")
    write_json(output_path, report)

    print("\nDone.")
    print(f"Saved report to: {output_path}")
    print("Current metrics:")
    print(f"  Perplexity: {current_ppl['perplexity']:.4f}")
    print(f"  MAUVE: {current_mauve:.4f}")
    print(f"  Tail token rate: {current_tail['tail_token_rate']:.6f}")
    print(f"  Distinct-2: {current_distinct2:.6f}")


if __name__ == "__main__":
    main()