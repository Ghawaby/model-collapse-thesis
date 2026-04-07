from pathlib import Path

raw_dir = Path("data/raw/wikitext2")
out_file = Path("data/raw/wikitext2_combined.txt")

parts = ["train.txt", "validation.txt", "test.txt"]

with out_file.open("w", encoding="utf-8") as out:
    for name in parts:
        file_path = raw_dir / name
        out.write(f"\n===== {name} =====\n")
        with file_path.open("r", encoding="utf-8") as f:
            out.write(f.read())
            out.write("\n")

print(f"Saved combined file to: {out_file}")