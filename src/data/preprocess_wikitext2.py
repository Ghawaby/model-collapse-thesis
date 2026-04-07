from transformers import AutoTokenizer
import os

TOKENIZER_NAME = "gpt2"
BLOCK_SIZE = 64

RAW_DIR = "data/raw/wikitext2"
PROCESSED_DIR = "data/processed/wikitext2_64"

def read_text_file(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return "\n".join(lines)

def chunk_tokens(token_ids, block_size):
    chunks = []
    for i in range(0, len(token_ids) - block_size + 1, block_size):
        chunk = token_ids[i:i + block_size]
        if len(chunk) == block_size:
            chunks.append(chunk)
    return chunks

def save_chunks(chunks, path):
    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(" ".join(map(str, chunk)) + "\n")

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    for split in ["train", "validation", "test"]:
        input_path = os.path.join(RAW_DIR, f"{split}.txt")
        output_path = os.path.join(PROCESSED_DIR, f"{split}_tokens_64.txt")

        text = read_text_file(input_path)
        token_ids = tokenizer.encode(text)

        chunks = chunk_tokens(token_ids, BLOCK_SIZE)
        save_chunks(chunks, output_path)

        print(f"{split}: {len(chunks)} chunks saved to {output_path}")

if __name__ == "__main__":
    main()