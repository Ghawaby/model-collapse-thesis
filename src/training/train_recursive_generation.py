from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a recursive generation checkpoint on a provided token dataset."
    )

    parser.add_argument("--train-file", type=str, required=True)
    parser.add_argument(
        "--val-file",
        type=str,
        default="data/processed/wikitext2_64/validation_tokens_64.txt",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="data/processed/wikitext2_64/test_tokens_64.txt",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        required=True,
        help="Checkpoint to initialize from. For baseline recursion, this is the previous generation checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Example: checkpoints/generation1",
    )
    parser.add_argument(
        "--history-json",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--history-csv",
        type=str,
        required=True,
    )
    parser.add_argument("--fallback-tokenizer", type=str, default="gpt2")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_token_line(line: str) -> Optional[List[int]]:
    line = line.strip()
    if not line:
        return None
    if line.startswith("[") and line.endswith("]"):
        parsed = json.loads(line)
        return [int(x) for x in parsed]
    parts = [p for p in line.replace(",", " ").split() if p]
    return [int(p) for p in parts]


class TokenChunkDataset(Dataset):
    def __init__(self, path: Path) -> None:
        self.rows: List[List[int]] = []
        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                row = parse_token_line(raw)
                if row:
                    self.rows.append(row)
        if not self.rows:
            raise ValueError(f"No rows loaded from {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> List[int]:
        return self.rows[idx]


def collate_token_rows(batch: Sequence[Sequence[int]]) -> Dict[str, torch.Tensor]:
    input_ids = torch.tensor(batch, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids.clone(),
    }


def load_tokenizer(checkpoint_path: Path, fallback_tokenizer: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(fallback_tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(checkpoint_path: Path, device: torch.device):
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(str(checkpoint_path), dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(str(checkpoint_path), torch_dtype=dtype)
    model.to(device)
    return model


def evaluate_loss(model, dataloader: DataLoader, device: torch.device, desc: str) -> Dict[str, float]:
    model.eval()
    total_nll = 0.0
    total_predicted_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            predicted_tokens = batch["input_ids"].shape[0] * (batch["input_ids"].shape[1] - 1)
            total_nll += outputs.loss.item() * predicted_tokens
            total_predicted_tokens += predicted_tokens

    avg_nll = total_nll / max(1, total_predicted_tokens)
    ppl = math.exp(avg_nll)

    return {
        "avg_nll": avg_nll,
        "perplexity": ppl,
    }


def save_history_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_history_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    ensure_parent(path)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_file = Path(args.train_file)
    val_file = Path(args.val_file)
    test_file = Path(args.test_file)
    init_checkpoint = Path(args.init_checkpoint)
    output_dir = Path(args.output_dir)
    best_dir = output_dir / "best"
    last_dir = output_dir / "last"

    for path in [train_file, val_file, test_file, init_checkpoint]:
        if not path.exists():
            raise FileNotFoundError(f"Missing path: {path}")

    print("Loading tokenizer...")
    tokenizer = load_tokenizer(init_checkpoint, args.fallback_tokenizer)

    print("Loading datasets...")
    train_ds = TokenChunkDataset(train_file)
    val_ds = TokenChunkDataset(val_file)
    test_ds = TokenChunkDataset(test_file)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_token_rows,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_token_rows,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_token_rows,
    )

    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(init_checkpoint, device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    epoch_rows: List[Dict[str, Any]] = []
    best_val_nll = float("inf")

    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_nll = 0.0
        running_predicted_tokens = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            predicted_tokens = batch["input_ids"].shape[0] * (batch["input_ids"].shape[1] - 1)
            running_nll += loss.item() * predicted_tokens
            running_predicted_tokens += predicted_tokens

        train_avg_nll = running_nll / max(1, running_predicted_tokens)
        train_ppl = math.exp(train_avg_nll)

        val_metrics = evaluate_loss(model, val_loader, device, desc=f"Validate epoch {epoch}")

        epoch_row = {
            "epoch": epoch,
            "train_avg_nll": train_avg_nll,
            "train_perplexity": train_ppl,
            "val_avg_nll": val_metrics["avg_nll"],
            "val_perplexity": val_metrics["perplexity"],
        }
        epoch_rows.append(epoch_row)

        print(
            f"Epoch {epoch}: "
            f"train_ppl={train_ppl:.4f}, "
            f"val_ppl={val_metrics['perplexity']:.4f}"
        )

        if val_metrics["avg_nll"] < best_val_nll:
            best_val_nll = val_metrics["avg_nll"]
            ensure_dir(best_dir)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)

    print("Saving last checkpoint...")
    ensure_dir(last_dir)
    model.save_pretrained(last_dir)
    tokenizer.save_pretrained(last_dir)

    print("Evaluating best checkpoint on test split...")
    best_model = load_model(best_dir, device)
    test_metrics = evaluate_loss(best_model, test_loader, device, desc="Test best checkpoint")

    history_payload = {
        "train_file": str(train_file),
        "val_file": str(val_file),
        "test_file": str(test_file),
        "init_checkpoint": str(init_checkpoint),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "epochs": args.epochs,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "epoch_history": epoch_rows,
        "best_val_avg_nll": best_val_nll,
        "test_avg_nll": test_metrics["avg_nll"],
        "test_perplexity": test_metrics["perplexity"],
    }

    save_history_json(Path(args.history_json), history_payload)
    save_history_csv(Path(args.history_csv), epoch_rows)

    print("\nDone.")
    print(f"Best checkpoint: {best_dir}")
    print(f"Last checkpoint: {last_dir}")
    print(f"Test perplexity: {test_metrics['perplexity']:.4f}")


if __name__ == "__main__":
    main()
