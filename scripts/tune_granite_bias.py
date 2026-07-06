import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupKFold

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from action_router.constants import ACTION_CLASSES, LABEL2ID
from action_router.features import render_granite_sample, session_group


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_labels(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {row["id"]: row["action"] for row in csv.DictReader(f)}


def build_validation_data(data_dir, fold, n_splits, max_history_events, open_files_mode):
    samples = load_jsonl(Path(data_dir) / "train.jsonl")
    labels = load_labels(Path(data_dir) / "train_labels.csv")
    texts = []
    y = []
    groups = []
    for sample in samples:
        sample_id = sample["id"]
        texts.append(
            render_granite_sample(
                sample,
                max_history_events=max_history_events,
                open_files_mode=open_files_mode,
            )
        )
        y.append(LABEL2ID[labels[sample_id]])
        groups.append(session_group(sample_id))

    texts = np.array(texts, dtype=object)
    y = np.array(y, dtype=np.int64)
    groups = np.array(groups, dtype=object)
    splits = list(GroupKFold(n_splits=n_splits).split(texts, y, groups))
    _, val_idx = splits[fold]
    return texts[val_idx].tolist(), y[val_idx]


def predict_logits(model_dir, texts, max_length, batch_size):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    class TextDataset:
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            return tokenizer(self.items[idx], truncation=True, max_length=max_length, padding=False)

    loader = DataLoader(
        TextDataset(texts),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    logits = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(**batch).logits
            logits.append(out.float().cpu().numpy())
    return np.concatenate(logits, axis=0)


def macro_f1_for_bias(logits, y, bias):
    preds = np.argmax(logits + bias[None, :], axis=1)
    return f1_score(y, preds, labels=list(range(len(ACTION_CLASSES))), average="macro", zero_division=0)


def tune_bias(logits, y):
    bias = np.zeros(len(ACTION_CLASSES), dtype=np.float32)
    best = macro_f1_for_bias(logits, y, bias)
    print(f"base_macro_f1={best:.6f}")

    # Coarse-to-fine coordinate search over additive class logit biases.
    for step_size in [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01]:
        improved = True
        rounds = 0
        while improved and rounds < 8:
            improved = False
            rounds += 1
            for class_id in range(len(ACTION_CLASSES)):
                current = bias[class_id]
                candidates = [current - step_size, current, current + step_size]
                scores = []
                for value in candidates:
                    trial = bias.copy()
                    trial[class_id] = value
                    scores.append(macro_f1_for_bias(logits, y, trial))
                best_idx = int(np.argmax(scores))
                if scores[best_idx] > best + 1e-8:
                    bias[class_id] = candidates[best_idx]
                    best = scores[best_idx]
                    improved = True
        print(f"step={step_size:.3f} best_macro_f1={best:.6f}")
    return bias, best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--model-dir", default="./model/granite-311m-fold0")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-history-events", type=int, default=12)
    parser.add_argument("--open-files-mode", choices=["count", "basename", "basename_space", "ext", "path"], default="count")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    texts, y = build_validation_data(
        args.data_dir,
        args.fold,
        args.n_splits,
        args.max_history_events,
        args.open_files_mode,
    )
    print(f"validation_samples={len(texts)}")
    logits = predict_logits(model_dir, texts, args.max_length, args.batch_size)

    base_preds = np.argmax(logits, axis=1)
    print(classification_report(y, base_preds, target_names=ACTION_CLASSES, digits=4, zero_division=0))

    bias, tuned_f1 = tune_bias(logits, y)
    tuned_preds = np.argmax(logits + bias[None, :], axis=1)
    print(classification_report(y, tuned_preds, target_names=ACTION_CLASSES, digits=4, zero_division=0))

    payload = {
        "fold": args.fold,
        "n_splits": args.n_splits,
        "base_macro_f1": float(macro_f1_for_bias(logits, y, np.zeros(len(ACTION_CLASSES), dtype=np.float32))),
        "tuned_macro_f1": float(tuned_f1),
        "action_classes": ACTION_CLASSES,
        "open_files_mode": args.open_files_mode,
        "bias": {label: float(bias[i]) for i, label in enumerate(ACTION_CLASSES)},
    }
    out_path = model_dir / "logit_bias.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
