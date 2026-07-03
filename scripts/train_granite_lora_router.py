import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from action_router.constants import ACTION_CLASSES, ID2LABEL, LABEL2ID
from action_router.features import render_granite_sample, session_group


class ActionDataset:
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        encoded["labels"] = int(self.labels[idx])
        return encoded


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_labels(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {row["id"]: row["action"] for row in csv.DictReader(f)}


def build_data(data_dir, max_history_events):
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
    return np.array(texts, dtype=object), np.array(y, dtype=np.int64), np.array(groups, dtype=object)


def class_weights(y):
    counts = Counter(int(v) for v in y)
    weights = []
    total = len(y)
    n_classes = len(ACTION_CLASSES)
    for label_id in range(n_classes):
        weights.append(total / (n_classes * max(counts[label_id], 1)))
    weights = np.array(weights, dtype=np.float32)
    return weights / weights.mean()


def evaluate(model, loader, device, use_fp16):
    import torch

    model.eval()
    preds = []
    gold = []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").numpy().tolist()
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=use_fp16 and device.type == "cuda"):
                logits = model(**batch).logits
            preds.extend(torch.argmax(logits, dim=-1).cpu().numpy().tolist())
            gold.extend(labels)
    return f1_score(gold, preds, labels=list(range(len(ACTION_CLASSES))), average="macro", zero_division=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--model-name", default="ibm-granite/granite-embedding-311m-multilingual-r2")
    parser.add_argument("--output-dir", default="./model/granite-311m-lora-fold0")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-history-events", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-targets", default="Wqkv,Wo,Wi")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--save-merged", action="store_true")
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        get_linear_schedule_with_warmup,
        set_seed,
    )

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    texts, y, groups = build_data(args.data_dir, args.max_history_events)

    splitter = GroupKFold(n_splits=args.n_splits)
    splits = list(splitter.split(texts, y, groups))
    train_idx, val_idx = splits[args.fold]
    if args.max_train_samples:
        train_idx = train_idx[: args.max_train_samples]
    if args.max_val_samples:
        val_idx = val_idx[: args.max_val_samples]
    train_texts, val_texts = texts[train_idx].tolist(), texts[val_idx].tolist()
    y_train, y_val = y[train_idx], y[val_idx]
    print(f"train={len(train_texts)} val={len(val_texts)} fold={args.fold}/{args.n_splits} device={device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(ACTION_CLASSES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[item.strip() for item in args.lora_targets.split(",") if item.strip()],
        modules_to_save=["head", "classifier"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(
        ActionDataset(train_texts, y_train, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=2,
    )
    val_loader = DataLoader(
        ActionDataset(val_texts, y_val, tokenizer, args.max_length),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
    )

    weights = torch.tensor(class_weights(y_train), dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    update_steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = update_steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * args.warmup_ratio)),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16 and device.type == "cuda")

    best_f1 = -1.0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        update_step = 0
        for step, batch in enumerate(train_loader, start=1):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=args.fp16 and device.type == "cuda"):
                logits = model(**batch).logits
                loss = criterion(logits, labels) / args.grad_accum
            scaler.scale(loss).backward()
            running_loss += float(loss.item()) * args.grad_accum

            if step % args.grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update_step += 1
                if update_step % 50 == 0:
                    print(
                        f"epoch={epoch} update={update_step}/{update_steps_per_epoch} "
                        f"loss={running_loss / step:.4f}"
                    )

        macro_f1 = evaluate(model, val_loader, device, args.fp16)
        print(f"epoch={epoch} val_macro_f1={macro_f1:.5f}")
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            with open(Path(args.output_dir) / "training_meta.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "base_model": args.model_name,
                        "best_val_macro_f1": best_f1,
                        "fold": args.fold,
                        "n_splits": args.n_splits,
                        "max_length": args.max_length,
                        "max_history_events": args.max_history_events,
                        "lora_r": args.lora_r,
                        "lora_alpha": args.lora_alpha,
                        "lora_dropout": args.lora_dropout,
                        "lora_targets": args.lora_targets,
                        "action_classes": ACTION_CLASSES,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"saved best adapter model to {args.output_dir}")

    if args.save_merged:
        merged_dir = Path(args.output_dir) / "merged"
        print(f"saving merged full model to {merged_dir}")
        merged = model.merge_and_unload()
        merged.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)


if __name__ == "__main__":
    main()
