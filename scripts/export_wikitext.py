from datasets import load_dataset
import os

# Load WikiText-2
dataset = load_dataset("wikitext", "wikitext-2-v1")

# Ensure src exists
os.makedirs("src", exist_ok=True)

# Write cleaned training text
with open("src/wiki.train.tokens", "w", encoding="utf-8") as f:
    for line in dataset["train"]["text"]:
        line = line.strip()
        if line:
            f.write(line + "\n")

print("Export complete: src/wiki.train.tokens")
