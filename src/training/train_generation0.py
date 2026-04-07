from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer, get_linear_schedule_with_warmup


@dataclass
class TrainConfig:
    tokenizer_name: str = "gpt2"

    train_file: str = "data/processed/wikitext2_64/train_tokens_64.txt"
    val_file: str = "data/processed/wikitext2_64/validation_tokens_64.txt"
    test_file: str = "data/processed/wikitext2_64/test_tokens_64.txt"

    output_dir: str = "checkpoints/generation0"
    log_dir: str = "results/logs"

    block_size: int = 64
    batch_size: int = 16
    epochs: int = 10
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    seed: int = 42

    # Small GPT-2 style model for a manageable bachelor-thesis setup
    n_embd: int = 256
    n_layer: int = 4
    n_head: int = 4


class TokenBlockDataset(Dataset):
    def __init__(self, file_path: str, block_size: int) -> None:
        self.file_path = Path(file_path)
        self.block_size = block_size

        if not self.file_path.exists():
            raise FileNotFoundError(f"Missing dataset file: {self.file_path}")

        self.samples: list[list[int]] = []
        with self.file_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                token_ids = [int(x) for x in line.split()]
                if len(token_ids) != self.block_size:
                    raise ValueError(
                        f"{self.file_path} line {line_num} has {len(token_ids)} tokens, "
                        f"expected {self.block_size}."
                    )
                self.samples.append(token_ids)

        if not self.samples:
            raise ValueError(f"No usable samples found in {self.file_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        input_ids = torch.tensor(self.samples[idx], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate(model: GPT2LMHeadModel, dataloader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            total_loss += loss.item()
            total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)

    # Prevent overflow if loss is very large
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return avg_loss, perplexity


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_history_csv(history: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return

    fieldnames = list(history[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    cfg = TrainConfig()
    set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    best_dir = output_dir / "best"
    last_dir = output_dir / "last"
    log_dir = Path(cfg.log_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)
    last_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = TokenBlockDataset(cfg.train_file, cfg.block_size)
    val_dataset = TokenBlockDataset(cfg.val_file, cfg.block_size)
    test_dataset = TokenBlockDataset(cfg.test_file, cfg.block_size)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    model_config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=cfg.block_size,
        n_ctx=cfg.block_size,
        n_embd=cfg.n_embd,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    model = GPT2LMHeadModel(model_config)
    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    total_steps = cfg.epochs * len(train_loader)
    warmup_steps = int(cfg.warmup_ratio * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    save_json(asdict(cfg), output_dir / "train_config.json")

    history: list[dict] = []
    best_val_loss = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            if step % 200 == 0 or step == len(train_loader):
                print(
                    f"Epoch {epoch}/{cfg.epochs} | "
                    f"Step {step}/{len(train_loader)} | "
                    f"Train loss: {loss.item():.4f}"
                )

        train_loss = running_loss / len(train_loader)
        train_ppl = math.exp(train_loss) if train_loss < 20 else float("inf")

        val_loss, val_ppl = evaluate(model, val_loader, device)

        epoch_row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_perplexity": round(train_ppl, 6) if math.isfinite(train_ppl) else "inf",
            "val_loss": round(val_loss, 6),
            "val_perplexity": round(val_ppl, 6) if math.isfinite(val_ppl) else "inf",
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_row)

        print("-" * 60)
        print(f"Epoch {epoch} finished")
        print(f"Train loss: {train_loss:.4f} | Train perplexity: {train_ppl:.4f}")
        print(f"Val   loss: {val_loss:.4f} | Val   perplexity: {val_ppl:.4f}")
        print("-" * 60)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            save_json({"best_epoch": epoch, "best_val_loss": val_loss}, best_dir / "best_metrics.json")
            print(f"New best model saved to: {best_dir}")

    # Save final model from last epoch
    model.save_pretrained(last_dir)
    tokenizer.save_pretrained(last_dir)

    # Evaluate best model on test set
    best_model = GPT2LMHeadModel.from_pretrained(best_dir).to(device)
    test_loss, test_ppl = evaluate(best_model, test_loader, device)

    final_summary = {
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_perplexity": test_ppl if math.isfinite(test_ppl) else "inf",
    }

    save_json(final_summary, output_dir / "final_summary.json")
    save_json({"history": history}, log_dir / "generation0_history.json")
    save_history_csv(history, log_dir / "generation0_history.csv")

    print("=" * 60)
    print("Training complete")
    print(f"Best model folder: {best_dir}")
    print(f"Last model folder: {last_dir}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test perplexity: {test_ppl:.4f}" if math.isfinite(test_ppl) else "Test perplexity: inf")
    print("=" * 60)


if __name__ == "__main__":
    main()
