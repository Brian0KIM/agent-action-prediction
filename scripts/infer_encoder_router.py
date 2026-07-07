import argparse
import csv
import json
import os
import sys
from pathlib import Path

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
    return [float(bias_map.get(id2label[idx], 0.0)) for idx in range(len(id2label))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--model-dir", default="./model/bge-m3-router-fold0")
    parser.add_argument("--output-path", default="./output/submission.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-history-events", type=int, default=12)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--use-slow-tokenizer", action="store_true")
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

    class TextDataset:
        def __init__(self, texts, tokenizer, max_length):
            self.texts = texts
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            return self.tokenizer(
                self.texts[idx],
                truncation=True,
                max_length=self.max_length,
                padding=False,
            )

    model_dir = Path(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        use_fast=not args.use_slow_tokenizer,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=args.trust_remote_code,
        attn_implementation="eager",  # ModernBERT needs eager to match training; safe for other encoders too
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    bias_values = load_logit_bias(model_dir, model.config.id2label)
    bias = torch.tensor(bias_values, dtype=torch.float32, device=device) if bias_values is not None else None

    samples = load_jsonl(Path(args.data_dir) / "test.jsonl")
    ids = [sample["id"] for sample in samples]
    texts = [render_granite_sample(sample, max_history_events=args.max_history_events) for sample in samples]

    loader = DataLoader(
        TextDataset(texts, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(**batch).logits
            if bias is not None:
                logits = logits.float() + bias
            pred_ids = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
            preds.extend([model.config.id2label[int(i)] for i in pred_ids])

    pred_map = dict(zip(ids, preds))
    fieldnames, rows = load_sample_submission(Path(args.data_dir) / "sample_submission.csv")
    for row in rows:
        row["action"] = pred_map[row["id"]]
    save_submission(args.output_path, fieldnames, rows)
    print(f"Saved {args.output_path} rows={len(rows)}")


if __name__ == "__main__":
    main()
