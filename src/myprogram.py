#!/usr/bin/env python
import os
import string
import random
import json
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import Counter, defaultdict

MODEL_FNAME = "model.json"

def _safe_read_text(path: str) -> str:
    """Read UTF-8 text robustly (ignore decode errors)."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _find_training_files() -> list[str]:
    """
    Find training text files in common locations.
    Supports:
      - data/*.txt
      - /job/data/*.txt (inside docker)
      - src/wiki.train.tokens (your current corpus export)
      - example/input.txt as last-resort fallback
    """
    candidates: list[str] = []

    # Common local folder
    if os.path.isdir("data"):
        for fn in os.listdir("data"):
            if fn.endswith(".txt") or fn.endswith(".tokens"):
                candidates.append(os.path.join("data", fn))

    # If you're running inside docker with mounted /job/data
    if os.path.isdir("/job/data"):
        for fn in os.listdir("/job/data"):
            if fn.endswith(".txt") or fn.endswith(".tokens"):
                candidates.append(os.path.join("/job/data", fn))

    # Your exported corpus (as shown in your screenshots)
    if os.path.isfile("src/wiki.train.tokens"):
        candidates.append("src/wiki.train.tokens")

    # Fallback to example inputs if nothing else exists
    if os.path.isfile("example/input.txt"):
        candidates.append("example/input.txt")

    # Deduplicate while preserving order
    seen = set()
    out = []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """

    def __init__(self):
        # Precomputed top-3 guesses for each context for fast prediction
        self.top3_uni: list[str] = []
        self.top3_bi: dict[str, list[str]] = {}   # context: 1 char -> list[str] length 3
        self.top3_tri: dict[str, list[str]] = {}  # context: 2 chars -> list[str] length 3

        # Default fallback guesses
        self.default_top3: list[str] = [" ", "e", "a"]


    @classmethod
    def load_training_data(cls) -> list[str]:
        """
        Returns a list of training texts (strings).
        """
        files = _find_training_files()
        texts: list[str] = []
        for path in files:
            try:
                texts.append(_safe_read_text(path))
            except OSError:
                continue
        return texts

    @classmethod
    def load_test_data(cls, fname: str) -> list[str]:
        data: list[str] = []
        with open(fname, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                inp = line[:-1]  # strip newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds: list[str], fname: str) -> None:
        with open(fname, "wt", encoding="utf-8") as f:
            for p in preds:
                f.write(f"{p}\n")

    def run_train(self, data: list[str], work_dir: str) -> None:
        """
        Train n-gram counts from training texts and precompute top-3 maps.
        data: list[str]
        """
        text = "\n".join(data)

        # If dataset is empty, keep defaults
        if not text:
            self.top3_uni = self.default_top3[:]
            self.top3_bi = {}
            self.top3_tri = {}
            return

        # Count unigrams, bigrams, trigrams over characters
        uni = Counter()
        bi = defaultdict(Counter)   # prev_char -> Counter(next_char)
        tri = defaultdict(Counter)  # prev2 (len 2) -> Counter(next_char)

        for ch in text:
            uni[ch] += 1

        for i in range(len(text) - 1):
            c1 = text[i]
            c2 = text[i + 1]
            bi[c1][c2] += 1

        for i in range(len(text) - 2):
            ctx = text[i : i + 2]
            nxt = text[i + 2]
            tri[ctx][nxt] += 1

        def best3(counter: Counter) -> list[str]:
            if not counter:
                return []
            chars = [c for (c, _) in counter.most_common(3)]
            while len(chars) < 3:
                chars.append(self.default_top3[len(chars)])
            return chars[:3]

        self.top3_uni = best3(uni)
        if len(self.top3_uni) < 3:
            self.top3_uni = (self.top3_uni + self.default_top3)[:3]

        self.top3_bi = {}
        for c1, ctr in bi.items():
            self.top3_bi[c1] = best3(ctr)

        self.top3_tri = {}
        for ctx, ctr in tri.items():
            self.top3_tri[ctx] = best3(ctr)

 
    def run_pred(self, data: list[str]) -> list[str]:
        """
        data: list[str] where each string is a prefix.
        Return list[str] where each element is exactly 3 characters (guesses).
        """
        preds: list[str] = []

        uni_fallback = self.top3_uni if self.top3_uni else self.default_top3

        for inp in data:
            guesses = None

            try:
                if len(inp) >= 2:
                    ctx2 = inp[-2:]
                    guesses = self.top3_tri.get(ctx2)

                if guesses is None and len(inp) >= 1:
                    ctx1 = inp[-1]
                    guesses = self.top3_bi.get(ctx1)

                if guesses is None:
                    guesses = uni_fallback

            except Exception:
                guesses = self.default_top3

            guesses = (guesses + self.default_top3)[:3]
            preds.append("".join(guesses))

        return preds

    def save(self, work_dir: str) -> None:
        os.makedirs(work_dir, exist_ok=True)
        payload = {
            "top3_uni": self.top3_uni,
            "top3_bi": self.top3_bi,
            "top3_tri": self.top3_tri,
            "default_top3": self.default_top3,
        }
        path = os.path.join(work_dir, MODEL_FNAME)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    @classmethod
    def load(cls, work_dir: str):
        model = MyModel()
        path = os.path.join(work_dir, MODEL_FNAME)

        # If no saved model yet, return defaults (won't crash graders)
        if not os.path.isfile(path):
            model.top3_uni = model.default_top3[:]
            return model

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        model.default_top3 = payload.get("default_top3") or model.default_top3
        model.top3_uni = payload.get("top3_uni") or model.default_top3
        model.top3_bi = payload.get("top3_bi") or {}
        model.top3_tri = payload.get("top3_tri") or {}
        return model


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("mode", choices=("train", "test"), help="what to run")
    parser.add_argument("--work_dir", help="where to save", default="work")
    parser.add_argument("--test_data", help="path to test data", default="example/input.txt")
    parser.add_argument("--test_output", help="path to write test predictions", default="pred.txt")
    args = parser.parse_args()

    if args.mode == "train":
        if not os.path.isdir(args.work_dir):
            print(f"Making working directory {args.work_dir}")
            os.makedirs(args.work_dir)

        print("Instantiating model")
        model = MyModel()

        print("Loading training data")
        train_data = MyModel.load_training_data()
        print(f"Loaded {len(train_data)} training text(s):")
        for i, _ in enumerate(train_data[:3]):
            print(f"  - training chunk {i+1}")
        if len(train_data) > 3:
            print(f"  ... (+{len(train_data)-3} more)")

        print("Training")
        model.run_train(train_data, args.work_dir)

        print("Saving model")
        model.save(args.work_dir)

    elif args.mode == "test":
        print("Loading model")
        model = MyModel.load(args.work_dir)

        print(f"Loading test data from {args.test_data}")
        test_data = MyModel.load_test_data(args.test_data)

        print("Making predictions")
        pred = model.run_pred(test_data)

        print(f"Writing predictions to {args.test_output}")
        assert len(pred) == len(test_data), f"Expected {len(test_data)} predictions but got {len(pred)}"
        model.write_pred(pred, args.test_output)

    else:
        raise NotImplementedError(f"Unknown mode {args.mode}")