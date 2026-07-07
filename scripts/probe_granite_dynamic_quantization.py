import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
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


def build_validation_data(data_dir, fold, n_splits, max_history_events, limit):
    samples = load_jsonl(Path(data_dir) / "train.jsonl")
    labels = load_labels(Path(data_dir) / "train_labels.csv")
    texts = []
    y = []
    groups = []
    for sample in samples:
        sample_id = sample["id"]
        texts.append(render_granite_sample(sample, max_history_events=max_history_events))
        y.append(LABEL2ID[labels[sample_id]])
        groups.append(session_group(sample_id))

    texts = np.array(texts, dtype=object)
    y = np.array(y, dtype=np.int64)
    groups = np.array(groups, dtype=object)
    splits = list(GroupKFold(n_splits=n_splits).split(texts, y, groups))
    _, val_idx = splits[fold]
    if limit:
        val_idx = val_idx[:limit]
    return texts[val_idx].tolist(), y[val_idx]


def load_logit_bias(model_dir, id2label):
    path = Path(model_dir) / "logit_bias.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    bias_map = payload.get("bias", {})
    return np.array([float(bias_map.get(id2label[idx], 0.0)) for idx in range(len(id2label))], dtype=np.float32)


def predict(model, tokenizer, texts, max_length, batch_size, bias):
    import torch
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding

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

    preds = []
    start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            logits = model(**batch).logits.float().numpy()
            if bias is not None:
                logits = logits + bias[None, :]
            preds.extend(np.argmax(logits, axis=1).tolist())
    elapsed = time.perf_counter() - start
    return np.array(preds, dtype=np.int64), elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--model-dir", default="./model/granite-311m-fold0-h16-l512")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-history-events", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    texts, y = build_validation_data(
        args.data_dir,
        args.fold,
        args.n_splits,
        args.max_history_events,
        args.limit,
    )
    print(f"validation_samples={len(texts)} device=cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir, local_files_only=True, attn_implementation="eager"
    )  # ModernBERT: eager must match training or accuracy craters to ~0.14
    model.to("cpu")
    model.eval()
    bias = load_logit_bias(args.model_dir, model.config.id2label)

    fp_preds, fp_elapsed = predict(model, tokenizer, texts, args.max_length, args.batch_size, bias)
    fp_f1 = f1_score(y, fp_preds, labels=list(range(len(ACTION_CLASSES))), average="macro", zero_division=0)
    print(f"fp_model_macro_f1={fp_f1:.6f} elapsed_sec={fp_elapsed:.2f}")

    quantized = torch.ao.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
        inplace=False,
    )
    quantized.eval()
    q_preds, q_elapsed = predict(quantized, tokenizer, texts, args.max_length, args.batch_size, bias)
    q_f1 = f1_score(y, q_preds, labels=list(range(len(ACTION_CLASSES))), average="macro", zero_division=0)
    print(f"dynamic_int8_macro_f1={q_f1:.6f} elapsed_sec={q_elapsed:.2f}")
    print(f"delta_macro_f1={q_f1 - fp_f1:+.6f} speedup={fp_elapsed / q_elapsed:.3f}x")


if __name__ == "__main__":
    main()
