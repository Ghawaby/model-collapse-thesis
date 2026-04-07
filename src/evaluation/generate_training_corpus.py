from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic training corpus from a checkpoint using fixed train prompts."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--prompt-source",
        type=str,
        default="data/processed/wikitext2_64/train_tokens_64.txt",
        help="Processed 64-token train chunks. The first prompt_len tokens of each row are used as prompts.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        required=True,
        help="Detailed JSONL output with prompts and generated ids/text.",
    )
    parser.add_argument(
        "--output-train-tokens",
        type=str,
        required=True,
        help="Training-ready token file for the next generation. One JSON token list per line.",
    )
    parser.add_argument("--fallback-tokenizer", type=str, default="gpt2")
    parser.add_argument("--prompt-len", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="0 means all rows")
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_token_line(line: str) -> Optional[List[int]]:
    line = line.strip()
    if not line:
        return None
    if line.startswith("[") and line.endswith("]"):
        parsed = json.loads(line)
        return [int(x) for x in parsed]
    parts = [p for p in line.replace(",", " ").split() if p]
    return [int(p) for p in parts]


def load_token_chunks(path: Path) -> List[List[int]]:
    rows: List[List[int]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            chunk = parse_token_line(raw)
            if chunk:
                rows.append(chunk)
    if not rows:
        raise ValueError(f"No token rows loaded from {path}")
    return rows


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_token_rows(path: Path, token_rows: Sequence[Sequence[int]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for row in token_rows:
            f.write(json.dumps(list(row), ensure_ascii=False) + "\n")


def load_tokenizer(checkpoint_path: Path, fallback_tokenizer: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(fallback_tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model(checkpoint_path: Path, device: torch.device):
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(str(checkpoint_path), dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(str(checkpoint_path), torch_dtype=dtype)
    model.to(device)
    model.eval()
    return model


def decode_ids(tokenizer, ids: Sequence[int]) -> str:
    return tokenizer.decode(list(ids), clean_up_tokenization_spaces=False)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    checkpoint = Path(args.checkpoint)
    prompt_source = Path(args.prompt_source)
    output_jsonl = Path(args.output_jsonl)
    output_train_tokens = Path(args.output_train_tokens)

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not prompt_source.exists():
        raise FileNotFoundError(f"Prompt source not found: {prompt_source}")

    print("Loading tokenizer...")
    tokenizer = load_tokenizer(checkpoint, args.fallback_tokenizer)

    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint, device)

    print("Loading prompt source...")
    rows = load_token_chunks(prompt_source)
    if args.limit > 0:
        rows = rows[: args.limit]

    prompt_rows: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        if len(row) < args.prompt_len:
            continue
        prompt_ids = row[: args.prompt_len]
        prompt_rows.append(
            {
                "row_id": i,
                "prompt_token_ids": prompt_ids,
                "prompt_text": decode_ids(tokenizer, prompt_ids),
            }
        )

    generated_rows: List[Dict[str, Any]] = []
    generated_token_rows: List[List[int]] = []

    for start in tqdm(range(0, len(prompt_rows), args.batch_size), desc="Generating synthetic training corpus"):
        batch = prompt_rows[start : start + args.batch_size]
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
            "max_new_tokens": args.max_new_tokens,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        if args.greedy:
            generate_kwargs["do_sample"] = False
        else:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = args.temperature
            generate_kwargs["top_p"] = args.top_p

        with torch.no_grad():
            outputs = model.generate(**generate_kwargs)

        padded_prompt_len = input_ids.shape[1]

        for prompt_row, output_ids in zip(batch, outputs):
            completion_ids = output_ids[padded_prompt_len:].tolist()

            if len(completion_ids) < args.max_new_tokens:
                completion_ids = completion_ids + [tokenizer.eos_token_id] * (args.max_new_tokens - len(completion_ids))
            else:
                completion_ids = completion_ids[: args.max_new_tokens]

            full_ids = list(prompt_row["prompt_token_ids"]) + completion_ids

            generated_rows.append(
                {
                    "row_id": prompt_row["row_id"],
                    "prompt_token_ids": prompt_row["prompt_token_ids"],
                    "prompt_text": prompt_row["prompt_text"],
                    "generated_completion_token_ids": completion_ids,
                    "generated_completion_text": decode_ids(tokenizer, completion_ids),
                    "generated_full_token_ids": full_ids,
                    "generated_full_text": decode_ids(tokenizer, full_ids),
                }
            )
            generated_token_rows.append(full_ids)

    print("Saving outputs...")
    write_jsonl(output_jsonl, generated_rows)
    write_token_rows(output_train_tokens, generated_token_rows)

    print("\nDone.")
    print(f"Saved detailed JSONL to: {output_jsonl}")
    print(f"Saved train token rows to: {output_train_tokens}")
    print(f"Rows generated: {len(generated_rows)}")


if __name__ == "__main__":
    main()