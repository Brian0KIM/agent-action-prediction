import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from action_router.features import render_granite_sample, session_group


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_labels(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {row["id"]: row["action"] for row in csv.DictReader(f)}


def build_texts(data_dir, fold, n_splits, max_history_events, limit):
    samples = load_jsonl(Path(data_dir) / "train.jsonl")
    labels = load_labels(Path(data_dir) / "train_labels.csv")
    texts = []
    y = []
    groups = []
    for sample in samples:
        sample_id = sample["id"]
        texts.append(render_granite_sample(sample, max_history_events=max_history_events))
        y.append(labels[sample_id])
        groups.append(session_group(sample_id))

    texts = np.array(texts, dtype=object)
    y = np.array(y, dtype=object)
    groups = np.array(groups, dtype=object)
    splits = list(GroupKFold(n_splits=n_splits).split(texts, y, groups))
    _, val_idx = splits[fold]
    if limit:
        val_idx = val_idx[:limit]
    return texts[val_idx].tolist()


def run_one_model(model_dir, texts, max_length, batch_size):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

    class TextDataset:
        def __init__(self, items, tokenizer):
            self.items = items
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            return self.tokenizer(self.items[idx], truncation=True, max_length=max_length, padding=False)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True, attn_implementation="eager"
    )  # ModernBERT: eager must match training or accuracy craters to ~0.14
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    loader = DataLoader(
        TextDataset(texts, tokenizer),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    rows = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                _ = model(**batch).logits
            rows += len(batch["input_ids"])
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed, rows, device.type


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--model-dirs", nargs="+", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-history-events", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=3000)
    args = parser.parse_args()

    texts = build_texts(args.data_dir, args.fold, args.n_splits, args.max_history_events, args.limit)
    total_elapsed = 0.0
    device = "unknown"
    for model_dir in args.model_dirs:
        elapsed, rows, device = run_one_model(model_dir, texts, args.max_length, args.batch_size)
        total_elapsed += elapsed
        print(f"model={model_dir} device={device} rows={rows} elapsed_sec={elapsed:.2f} rows_per_sec={rows / elapsed:.2f}")

    estimated_30k = total_elapsed * (30000 / len(texts))
    print(
        f"ensemble_models={len(args.model_dirs)} benchmark_rows={len(texts)} "
        f"total_elapsed_sec={total_elapsed:.2f} estimated_30k_sec={estimated_30k:.2f}"
    )


if __name__ == "__main__":
    main()
