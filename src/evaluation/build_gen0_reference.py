from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Gen0 reference package for Gen0-anchored recursive correction."
    )

    parser.add_argument(
        "--train-path",
        type=str,
        default="data/processed/wikitext2_64/train_tokens_64.txt",
        help="Processed human train chunks.",
    )
    parser.add_argument(
        "--eval-path",
        type=str,
        default="data/processed/wikitext2_64/validation_tokens_64.txt",
        help="Processed validation chunks used for fixed prompts.",
    )
    parser.add_argument(
        "--gen0-checkpoint",
        type=str,
        default="checkpoints/generation0/best",
        help="Path to trained Generation 0 checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/reference/gen0_reference",
        help="Directory where the Gen0 reference package will be saved.",
    )

    parser.add_argument(
        "--fallback-tokenizer",
        type=str,
        default="gpt2",
        help="Used only if the tokenizer was not saved inside the checkpoint folder.",
    )

    parser.add_argument("--num-prompts", type=int, default=500)
    parser.add_argument("--prompt-len", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)

    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy decoding instead of sampling.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)

    parser.add_argument(
        "--tail-mass",
        type=float,
        default=0.05,
        help="Define tail tokens as the rarest tokens covering this fraction of train-token mass.",
    )
    parser.add_argument("--tail-buffer-size", type=int, default=2000)
    parser.add_argument(
        "--repair-pool-size",
        type=int,
        default=0,
        help="0 means keep all remaining human chunks in the repair pool.",
    )

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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
        raise ValueError(f"No chunks loaded from {path}")

    return chunks


def decode_ids(tokenizer, ids: Sequence[int]) -> str:
    return tokenizer.decode(list(ids), clean_up_tokenization_spaces=False)


def build_token_counter(chunks: Sequence[Sequence[int]]) -> Counter:
    counter: Counter = Counter()
    for chunk in chunks:
        counter.update(chunk)
    return counter


def build_tail_token_set(counter: Counter, tail_mass: float) -> Tuple[List[int], int, int]:
    if not (0.0 < tail_mass < 1.0):
        raise ValueError("--tail-mass must be between 0 and 1")

    total_mass = sum(counter.values())
    target_mass = max(1, math.ceil(total_mass * tail_mass))

    sorted_items = sorted(counter.items(), key=lambda kv: (kv[1], kv[0]))

    tail_token_ids: List[int] = []
    cumulative = 0

    for token_id, count in sorted_items:
        tail_token_ids.append(token_id)
        cumulative += count
        if cumulative >= target_mass:
            break

    return tail_token_ids, cumulative, total_mass


def get_tail_stats(chunk: Sequence[int], tail_set: set[int]) -> Dict[str, Any]:
    tail_hits = [tok for tok in chunk if tok in tail_set]
    tail_unique = sorted(set(tail_hits))
    tail_count = len(tail_hits)
    tail_density = tail_count / max(1, len(chunk))

    return {
        "contains_tail": tail_count > 0,
        "tail_count": tail_count,
        "tail_density": tail_density,
        "tail_token_ids_present": tail_unique,
    }


def sample_eval_chunks(
    eval_chunks: Sequence[Sequence[int]],
    num_prompts: int,
    prompt_len: int,
    max_new_tokens: int,
    seed: int,
) -> List[Tuple[int, List[int]]]:
    min_len = prompt_len + max_new_tokens
    eligible: List[Tuple[int, List[int]]] = []

    for idx, chunk in enumerate(eval_chunks):
        if len(chunk) >= min_len:
            eligible.append((idx, list(chunk)))

    if len(eligible) < num_prompts:
        raise ValueError(
            f"Not enough eligible eval chunks. Needed {num_prompts}, found {len(eligible)}. "
            f"Need chunk length >= {min_len}."
        )

    rng = random.Random(seed)
    selected = rng.sample(eligible, num_prompts)
    selected.sort(key=lambda x: x[0])
    return selected


def write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_prompt_and_human_reference_rows(
    selected_eval_chunks: Sequence[Tuple[int, Sequence[int]]],
    tokenizer,
    prompt_len: int,
    max_new_tokens: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    prompt_rows: List[Dict[str, Any]] = []
    human_rows: List[Dict[str, Any]] = []

    for prompt_id, (source_chunk_index, chunk) in enumerate(selected_eval_chunks):
        prompt_ids = list(chunk[:prompt_len])
        human_completion_ids = list(chunk[prompt_len : prompt_len + max_new_tokens])
        human_full_ids = prompt_ids + human_completion_ids

        prompt_rows.append(
            {
                "prompt_id": prompt_id,
                "source_chunk_index": source_chunk_index,
                "prompt_token_ids": prompt_ids,
                "prompt_text": decode_ids(tokenizer, prompt_ids),
            }
        )

        human_rows.append(
            {
                "prompt_id": prompt_id,
                "source_chunk_index": source_chunk_index,
                "human_completion_token_ids": human_completion_ids,
                "human_completion_text": decode_ids(tokenizer, human_completion_ids),
                "human_full_token_ids": human_full_ids,
                "human_full_text": decode_ids(tokenizer, human_full_ids),
            }
        )

    return prompt_rows, human_rows


def load_tokenizer(checkpoint_path: Path, fallback_tokenizer: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(fallback_tokenizer)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return tokenizer


def generate_gen0_reference_rows(
    model,
    tokenizer,
    prompt_rows: Sequence[Dict[str, Any]],
    batch_size: int,
    max_new_tokens: int,
    greedy: bool,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> List[Dict[str, Any]]:
    model.eval()
    rows: List[Dict[str, Any]] = []

    for start in tqdm(range(0, len(prompt_rows), batch_size), desc="Generating Gen0 reference samples"):
        batch = prompt_rows[start : start + batch_size]
        batch_input_ids = [row["prompt_token_ids"] for row in batch]

        padded = tokenizer.pad(
            {"input_ids": batch_input_ids},
            padding=True,
            return_tensors="pt",
        )

        input_ids = padded["input_ids"].to(device)
        attention_mask = padded["attention_mask"].to(device)

        generate_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        if greedy:
            generate_kwargs["do_sample"] = False
        else:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p

        with torch.no_grad():
            outputs = model.generate(**generate_kwargs)

        padded_prompt_len = input_ids.shape[1]

        for row, output_ids in zip(batch, outputs):
            gen_completion_ids = output_ids[padded_prompt_len:].tolist()
            gen_full_ids = list(row["prompt_token_ids"]) + gen_completion_ids

            rows.append(
                {
                    "prompt_id": row["prompt_id"],
                    "gen0_completion_token_ids": gen_completion_ids,
                    "gen0_completion_text": decode_ids(tokenizer, gen_completion_ids),
                    "gen0_full_token_ids": gen_full_ids,
                    "gen0_full_text": decode_ids(tokenizer, gen_full_ids),
                    "generated_new_tokens": len(gen_completion_ids),
                }
            )

    rows.sort(key=lambda x: x["prompt_id"])
    return rows


def build_train_rows(
    train_chunks: Sequence[Sequence[int]],
    tokenizer,
    tail_set: set[int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for idx, chunk in enumerate(tqdm(train_chunks, desc="Building train metadata")):
        stats = get_tail_stats(chunk, tail_set)

        rows.append(
            {
                "chunk_index": idx,
                "token_ids": list(chunk),
                "text": decode_ids(tokenizer, chunk),
                "contains_tail": stats["contains_tail"],
                "tail_count": stats["tail_count"],
                "tail_density": stats["tail_density"],
                "tail_token_ids_present": stats["tail_token_ids_present"],
            }
        )

    return rows


def build_tail_buffer_and_repair_pool(
    train_rows: Sequence[Dict[str, Any]],
    tail_buffer_size: int,
    repair_pool_size: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows_with_tail = [row for row in train_rows if row["contains_tail"]]

    rows_with_tail_sorted = sorted(
        rows_with_tail,
        key=lambda r: (r["tail_density"], r["tail_count"], -r["chunk_index"]),
        reverse=True,
    )

    tail_buffer = rows_with_tail_sorted[:tail_buffer_size]
    tail_buffer_indices = {row["chunk_index"] for row in tail_buffer}

    remaining = [row for row in train_rows if row["chunk_index"] not in tail_buffer_indices]

    if repair_pool_size > 0 and repair_pool_size < len(remaining):
        rng = random.Random(seed)
        repair_pool = rng.sample(remaining, repair_pool_size)
        repair_pool.sort(key=lambda r: r["chunk_index"])
    else:
        repair_pool = remaining

    return tail_buffer, repair_pool


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    train_path = Path(args.train_path)
    eval_path = Path(args.eval_path)
    checkpoint_path = Path(args.gen0_checkpoint)
    output_dir = Path(args.output_dir)

    ensure_dir(output_dir)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Gen0 checkpoint not found: {checkpoint_path}")

    print("Loading tokenizer...")
    tokenizer = load_tokenizer(checkpoint_path, args.fallback_tokenizer)

    print("Loading token chunks...")
    train_chunks = load_token_chunks(train_path)
    eval_chunks = load_token_chunks(eval_path)

    print("Building train token frequencies...")
    token_counter = build_token_counter(train_chunks)

    print("Defining tail token set...")
    tail_token_ids, realized_tail_mass, total_train_token_mass = build_tail_token_set(
        token_counter,
        args.tail_mass,
    )
    tail_set = set(tail_token_ids)

    print("Selecting fixed evaluation prompts...")
    selected_eval_chunks = sample_eval_chunks(
        eval_chunks=eval_chunks,
        num_prompts=args.num_prompts,
        prompt_len=args.prompt_len,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    prompt_rows, human_reference_rows = build_prompt_and_human_reference_rows(
        selected_eval_chunks=selected_eval_chunks,
        tokenizer=tokenizer,
        prompt_len=args.prompt_len,
        max_new_tokens=args.max_new_tokens,
    )

    print("Loading Gen0 model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint_path),
        torch_dtype=dtype,
    )
    model.to(device)

    gen0_reference_rows = generate_gen0_reference_rows(
        model=model,
        tokenizer=tokenizer,
        prompt_rows=prompt_rows,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        greedy=args.greedy,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
    )

    print("Building tail buffer and repair pool...")
    train_rows = build_train_rows(
        train_chunks=train_chunks,
        tokenizer=tokenizer,
        tail_set=tail_set,
    )

    tail_buffer, repair_pool = build_tail_buffer_and_repair_pool(
        train_rows=train_rows,
        tail_buffer_size=args.tail_buffer_size,
        repair_pool_size=args.repair_pool_size,
        seed=args.seed,
    )

    tail_tokens_payload = {
        "definition": {
            "method": "bottom_empirical_probability_mass",
            "tail_mass_requested": args.tail_mass,
            "realized_tail_mass_tokens": realized_tail_mass,
            "total_train_token_mass": total_train_token_mass,
            "realized_tail_mass_fraction": realized_tail_mass / max(1, total_train_token_mass),
        },
        "tail_tokens": [
            {
                "token_id": token_id,
                "count": int(token_counter[token_id]),
                "decoded": decode_ids(tokenizer, [token_id]),
            }
            for token_id in tail_token_ids
        ],
    }

    config = {
        "train_path": str(train_path),
        "eval_path": str(eval_path),
        "gen0_checkpoint": str(checkpoint_path),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "prompt_len": args.prompt_len,
        "max_new_tokens": args.max_new_tokens,
        "num_prompts": args.num_prompts,
        "decoding": {
            "greedy": args.greedy,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "batch_size": args.batch_size,
        },
        "tail_definition": tail_tokens_payload["definition"],
        "buffer_sizes": {
            "tail_buffer_size_requested": args.tail_buffer_size,
            "repair_pool_size_requested": args.repair_pool_size,
        },
    }

    summary_stats = {
        "num_train_chunks": len(train_chunks),
        "num_eval_chunks_total": len(eval_chunks),
        "num_prompts_selected": len(prompt_rows),
        "num_unique_train_tokens_observed": len(token_counter),
        "num_tail_tokens": len(tail_token_ids),
        "tail_buffer_size_actual": len(tail_buffer),
        "repair_pool_size_actual": len(repair_pool),
        "device_used": str(device),
        "avg_tail_density_train": sum(r["tail_density"] for r in train_rows) / max(1, len(train_rows)),
        "avg_tail_count_train": sum(r["tail_count"] for r in train_rows) / max(1, len(train_rows)),
        "avg_gen0_new_tokens": sum(r["generated_new_tokens"] for r in gen0_reference_rows)
        / max(1, len(gen0_reference_rows)),
    }

    print("Saving Gen0 reference package...")
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "summary_stats.json", summary_stats)
    write_json(output_dir / "tail_tokens.json", tail_tokens_payload)

    write_jsonl(output_dir / "eval_prompts.jsonl", prompt_rows)
    write_jsonl(output_dir / "human_eval_references.jsonl", human_reference_rows)
    write_jsonl(output_dir / "gen0_reference_samples.jsonl", gen0_reference_rows)
    write_jsonl(output_dir / "tail_sample_buffer.jsonl", tail_buffer)
    write_jsonl(output_dir / "repair_pool.jsonl", repair_pool)

    print("\nDone.")
    print(f"Saved Gen0 reference package to: {output_dir}")


if __name__ == "__main__":
    main()