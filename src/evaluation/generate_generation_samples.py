from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate samples for a model on the fixed Gen0 evaluation prompt set."
    )

    parser.add_argument(
        "--reference-dir",
        type=str,
        default="data/reference/gen0_reference",
        help="Directory created by build_gen0_reference.py",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint of the model to generate from.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Where to save the generated JSONL file.",
    )
    parser.add_argument(
        "--generation-name",
        type=str,
        default="current_generation",
        help="Label stored in each output row.",
    )
    parser.add_argument(
        "--fallback-tokenizer",
        type=str,
        default="gpt2",
        help="Used if tokenizer was not saved inside the checkpoint folder.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="0 means use the batch size stored in the Gen0 reference config.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=0,
        help="0 means use the value stored in the Gen0 reference config.",
    )

    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Force greedy decoding. If omitted, decoding defaults to the Gen0 reference config.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=-1.0,
        help="If >= 0, override temperature from the Gen0 reference config.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=-1.0,
        help="If >= 0, override top-p from the Gen0 reference config.",
    )

    parser.add_argument(
        "--limit-prompts",
        type=int,
        default=0,
        help="Optional debug limit. 0 means use all prompts.",
    )

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


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def decode_ids(tokenizer, ids: List[int]) -> str:
    return tokenizer.decode(ids, clean_up_tokenization_spaces=False)


def resolve_generation_settings(
    args: argparse.Namespace,
    reference_config: Dict[str, Any],
) -> Dict[str, Any]:
    ref_decoding = reference_config.get("decoding", {})

    batch_size = args.batch_size if args.batch_size > 0 else int(ref_decoding.get("batch_size", 16))
    max_new_tokens = (
        args.max_new_tokens if args.max_new_tokens > 0 else int(reference_config.get("max_new_tokens", 32))
    )

    if args.greedy:
        greedy = True
    else:
        greedy = bool(ref_decoding.get("greedy", False))

    if args.temperature >= 0:
        temperature = args.temperature
    else:
        temperature = float(ref_decoding.get("temperature", 1.0))

    if args.top_p >= 0:
        top_p = args.top_p
    else:
        top_p = float(ref_decoding.get("top_p", 0.9))

    return {
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "greedy": greedy,
        "temperature": temperature,
        "top_p": top_p,
    }


def generate_rows(
    model,
    tokenizer,
    prompt_rows: List[Dict[str, Any]],
    generation_name: str,
    batch_size: int,
    max_new_tokens: int,
    greedy: bool,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    for start in tqdm(range(0, len(prompt_rows), batch_size), desc=f"Generating {generation_name} samples"):
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

        for prompt_row, output_ids in zip(batch, outputs):
            completion_ids = output_ids[padded_prompt_len:].tolist()
            full_ids = list(prompt_row["prompt_token_ids"]) + completion_ids

            results.append(
                {
                    "generation_name": generation_name,
                    "prompt_id": int(prompt_row["prompt_id"]),
                    "source_chunk_index": int(prompt_row["source_chunk_index"]),
                    "prompt_token_ids": prompt_row["prompt_token_ids"],
                    "prompt_text": prompt_row["prompt_text"],
                    "generated_completion_token_ids": completion_ids,
                    "generated_completion_text": decode_ids(tokenizer, completion_ids),
                    "generated_full_token_ids": full_ids,
                    "generated_full_text": decode_ids(tokenizer, full_ids),
                    "generated_new_tokens": len(completion_ids),
                }
            )

    results.sort(key=lambda x: x["prompt_id"])
    return results


def main() -> None:
    args = parse_args()

    reference_dir = Path(args.reference_dir)
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output_path)

    if not reference_dir.exists():
        raise FileNotFoundError(f"Reference dir not found: {reference_dir}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    config_path = reference_dir / "config.json"
    prompts_path = reference_dir / "eval_prompts.jsonl"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in reference dir: {config_path}")
    if not prompts_path.exists():
        raise FileNotFoundError(f"Missing eval_prompts.jsonl in reference dir: {prompts_path}")

    print("Loading reference config...")
    reference_config = load_json(config_path)

    print("Loading fixed evaluation prompts...")
    prompt_rows = load_jsonl(prompts_path)

    if args.limit_prompts > 0:
        prompt_rows = prompt_rows[: args.limit_prompts]

    settings = resolve_generation_settings(args, reference_config)

    print("Generation settings:")
    print(f"  batch_size = {settings['batch_size']}")
    print(f"  max_new_tokens = {settings['max_new_tokens']}")
    print(f"  greedy = {settings['greedy']}")
    print(f"  temperature = {settings['temperature']}")
    print(f"  top_p = {settings['top_p']}")

    print("Loading tokenizer...")
    tokenizer = load_tokenizer(checkpoint_path, args.fallback_tokenizer)

    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device)

    generated_rows = generate_rows(
        model=model,
        tokenizer=tokenizer,
        prompt_rows=prompt_rows,
        generation_name=args.generation_name,
        batch_size=settings["batch_size"],
        max_new_tokens=settings["max_new_tokens"],
        greedy=settings["greedy"],
        temperature=settings["temperature"],
        top_p=settings["top_p"],
        device=device,
    )

    print("Saving generated samples...")
    write_jsonl(output_path, generated_rows)

    print("\nDone.")
    print(f"Saved {len(generated_rows)} rows to: {output_path}")


if __name__ == "__main__":
    main()