#!/usr/bin/env python
"""
run once before training:
    python src/fetch.py

writes everything to src/data/train.txt (one big file) - 
the more diverse the data, the better the model handles multilingual input

Data sources used (all freely available, no API key needed):
  1. WikiText-103  (English Wikipedia, large)
  2. wikitext-2    (English Wikipedia, small fallback)
  3. Tatoeba sentences (multilingual — English, French, Spanish, German,
                        Portuguese, Italian, Japanese, Chinese, Arabic, Russian)
     via HuggingFace datasets
"""

import os
from pathlib import Path

OUT_DIR = Path("src/data")
OUT_FILE = OUT_DIR / "train.txt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_CHARS_PER_SOURCE = 5_000_000   # 5 MB per source — keeps model size sane

def write_chunk(fout, texts, source_name, max_chars=MAX_CHARS_PER_SOURCE):
    written = 0
    for t in texts:
        if not isinstance(t, str):
            continue
        t = t.strip()
        if not t:
            continue
        fout.write(t + "\n")
        written += len(t) + 1
        if written >= max_chars:
            break
    print(f"  [{source_name}] wrote ~{written/1e6:.2f} MB")
    return written


print("=== Fetching training data ===")
print(f"Output: {OUT_FILE}")

with open(OUT_FILE, "w", encoding="utf-8") as fout:

    # ── 1. WikiText-103 (English, large) ─────────────────────────────────────
    try:
        from datasets import load_dataset
        print("\n[1/3] WikiText-103 …")
        ds = load_dataset("wikitext", "wikitext-103-v1", split="train", trust_remote_code=True)
        write_chunk(fout, ds["text"], "wikitext-103")
    except Exception as e:
        print(f"  WikiText-103 failed ({e}); trying wikitext-2 …")
        try:
            ds = load_dataset("wikitext", "wikitext-2-v1", split="train", trust_remote_code=True)
            write_chunk(fout, ds["text"], "wikitext-2")
        except Exception as e2:
            print(f"  wikitext-2 also failed: {e2}")

    # ── 2. OpenWebText snippet (English web text) ─────────────────────────────
    try:
        print("\n[2/3] OpenWebText (English web) …")
        ds = load_dataset("openwebtext", split="train", trust_remote_code=True, streaming=True)
        texts = (row["text"] for row in ds)
        # stream up to MAX_CHARS_PER_SOURCE
        written = 0
        for t in texts:
            t = t.strip()
            if not t:
                continue
            fout.write(t + "\n")
            written += len(t) + 1
            if written >= MAX_CHARS_PER_SOURCE:
                break
        print(f"  [openwebtext] wrote ~{written/1e6:.2f} MB")
    except Exception as e:
        print(f"  OpenWebText failed: {e} (skipping)")

    # ── 3. Multilingual sentences (Tatoeba via opus_books / Helsinki) ─────────
    LANGUAGES = [
        ("fr", "French"),
        ("de", "German"),
        ("es", "Spanish"),
        ("it", "Italian"),
        ("pt", "Portuguese"),
        ("ru", "Russian"),
        ("ja", "Japanese"),
        ("zh", "Chinese"),
        ("ar", "Arabic"),
    ]
    print("\n[3/3] Multilingual (opus_books) …")
    for lang_code, lang_name in LANGUAGES:
        try:
            ds = load_dataset(
                "opus_books",
                f"en-{lang_code}",
                split="train",
                trust_remote_code=True,
            )
            # Each sample has a "translation" dict with "en" and lang_code keys
            texts = []
            for row in ds:
                t = row.get("translation", {})
                for v in t.values():
                    if isinstance(v, str):
                        texts.append(v)
            write_chunk(fout, texts, f"opus_books/{lang_name}", max_chars=1_000_000)
        except Exception as e:
            print(f"  opus_books/{lang_name} failed: {e} (skipping)")

total = OUT_FILE.stat().st_size
print(f"\n=== Done. Total: {total/1e6:.1f} MB written to {OUT_FILE} ===")
print("Now run: python src/myprogram.py train --work_dir work")