#!/usr/bin/env python
import os
import pickle
# import json
import math
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import Counter, defaultdict

# MODEL_FNAME = "model.json"
MODEL_FNAME = "model.pkl"

MAX_ORDER = 4
TOP_K     = 3
WEIGHTS = [0.10, 0.20, 0.35, 0.35]
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

    if os.path.isdir("src/data"):
        for fn in sorted(os.listdir("src/data")):
            if fn.endswith((".txt", ".tokens", ".text")):
                candidates.append(os.path.join("src/data", fn))

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
        self.default_top3: list = [" ", "e", "t"]
        self.unigram_top3: dict = [" ", "e", "t"]
        self.lookup: dict[str, str] = {}


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

        text = " ".join(data)

        if not text.strip():
            return
        
        unigram_counter = Counter(text)
        self.default_top3 = [c for c, _ in unigram_counter.most_common(3)]
        self.unigram_top3 = self.default_top3[:]

        print(f"Training on {len(text):,} characters, max order={MAX_ORDER}")

        # one counter per order instead of uni/bi/tri
        counters = [defaultdict(Counter) for _ in range(MAX_ORDER)]

        n = len(text)

        for i in range(n - 1):
            nxt = text[i + 1]
            counters[0][""][nxt] += 1
            for order in range(MAX_ORDER):
                # order=0: unigram (context="")
                # order=1: bigram (context=last 1 char)
                # ...
                start = i - order + 1
                if start < 0:
                    continue

                ctx = text[start : i + 1]
                counters[order][ctx][nxt] += 1

        self.lookup = {}

        # Unigram fallback (empty context)
        uni_top3 = "".join(c for c, _ in counters[0][""].most_common(TOP_K))
        self.lookup[""] = (uni_top3 + "".join(self.default_top3))[:3]

        for order in range(1, MAX_ORDER):
            for ctx, ctr in counters[order].items():
                top = "".join(c for c, _ in ctr.most_common(TOP_K))
                # pad with default if fewer than 3 unique successors seen
                padded = top + "".join(self.default_top3)
                self.lookup[ctx] = padded[:3]

        print(f"lookup table: {len(self.lookup):,} entries")
        print("done")


 
    def run_pred(self, data: list[str]) -> list[str]:
        """
        data: list[str] where each string is a prefix.
        Return list[str] where each element is exactly 3 characters (guesses).
        """
        preds = []

        for inp in data:
            try:
                preds.append(self._predict_one(inp))
            except Exception:
                preds.append("".join(self.default_top3[:3]))

        return preds
    
    def _predict_one(self, inp: str) -> str:
        for order in range(MAX_ORDER - 1, 0, -1):
            if len(inp) >= order:
                ctx = inp[-order:]
                result = self.lookup.get(ctx)
                if result:
                    return result

        # unigram fallback
        result = self.lookup.get("")
        if result:
            return result

        return "".join(self.default_top3[:3])

    def save(self, work_dir: str) -> None:
        os.makedirs(work_dir, exist_ok=True)

        path = os.path.join(work_dir, MODEL_FNAME)

        payload = {
            "lookup": self.lookup,
            "default_top3": self.default_top3,
            "unigram_top3": self.unigram_top3,
        }

        with open(path, "wb") as f:   # binary mode
            pickle.dump(payload, f, protocol=4)

        print(f"saved to {path}")
            
    @classmethod
    def load(cls, work_dir: str):
        model = cls()
        path = os.path.join(work_dir, MODEL_FNAME)

        if not os.path.isfile(path):
            print(f"erm no model found at {path}; using defaults")
            return model

        with open(path, "rb") as f:
            payload = pickle.load(f)

        model.lookup = payload.get("lookup", {})
        model.default_top3 = payload.get("default_top3", model.default_top3)
        model.unigram_top3 = payload.get("unigram_top3", model.default_top3)

        print(f"loaded from {path}")
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