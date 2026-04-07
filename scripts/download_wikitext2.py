from datasets import load_dataset
import os

def main():
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

    save_dir = "data/raw/wikitext2"
    os.makedirs(save_dir, exist_ok=True)

    for split in ["train", "validation", "test"]:
        output_path = os.path.join(save_dir, f"{split}.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            for row in dataset[split]:
                f.write(row["text"] + "\n")
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()