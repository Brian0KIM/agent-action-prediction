import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from action_router.features import render_granite_sample


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_sample_submission(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def save_submission(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_logit_bias(model_dir, id2label):
    path = Path(model_dir) / "logit_bias.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    bias_map = payload.get("bias", {})
    return np.array([float(bias_map.get(id2label[idx], 0.0)) for idx in range(len(id2label))], dtype=np.float32)


def predict_logits(model_dir, texts, max_length, batch_size):
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
            return self.tokenizer(
                self.items[idx],
                truncation=True,
                max_length=max_length,
                padding=False,
            )

    model_dir = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    bias = load_logit_bias(model_dir, model.config.id2label)

    loader = DataLoader(
        TextDataset(texts, tokenizer),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    chunks = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(**batch).logits.float().cpu().numpy()
            if bias is not None:
                logits = logits + bias[None, :]
            chunks.append(logits)

    return np.concatenate(chunks, axis=0), model.config.id2label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--model-dirs", nargs="+", required=True)
    parser.add_argument("--output-path", default="./output/submission_granite_ensemble.csv")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-history-events", type=int, default=16)
    args = parser.parse_args()

    samples = load_jsonl(Path(args.data_dir) / "test.jsonl")
    ids = [sample["id"] for sample in samples]
    texts = [render_granite_sample(sample, max_history_events=args.max_history_events) for sample in samples]

    ensemble_logits = None
    id2label = None
    for model_dir in args.model_dirs:
        logits, model_id2label = predict_logits(model_dir, texts, args.max_length, args.batch_size)
        id2label = model_id2label
        if ensemble_logits is None:
            ensemble_logits = logits
        else:
            ensemble_logits += logits
        print(f"added model={model_dir} rows={len(texts)}")

    pred_ids = np.argmax(ensemble_logits / len(args.model_dirs), axis=1).tolist()
    preds = [id2label[int(i)] for i in pred_ids]
    pred_map = dict(zip(ids, preds))

    fieldnames, rows = load_sample_submission(Path(args.data_dir) / "sample_submission.csv")
    for row in rows:
        row["action"] = pred_map[row["id"]]
    save_submission(args.output_path, fieldnames, rows)
    print(f"Saved {args.output_path} rows={len(rows)} models={len(args.model_dirs)}")


if __name__ == "__main__":
    main()
