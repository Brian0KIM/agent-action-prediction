import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, f1_score

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from action_router.features import render_granite_sample, session_group
from action_router.split import split_train_val


os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

SEARCH_CLASSES = ["read_file", "grep_search", "list_directory", "glob_pattern"]
SEARCH_LABEL2ID = {label: idx for idx, label in enumerate(SEARCH_CLASSES)}
SEARCH_ID2LABEL = {idx: label for label, idx in SEARCH_LABEL2ID.items()}


class SearchDataset:
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
        action = labels[sample["id"]]
        if action not in SEARCH_LABEL2ID:
            continue
        texts.append(render_granite_sample(sample, max_history_events=max_history_events))
        y.append(SEARCH_LABEL2ID[action])
        groups.append(session_group(sample["id"]))
    return np.array(texts, dtype=object), np.array(y, dtype=np.int64), np.array(groups, dtype=object)


def class_weights(y):
    counts = Counter(int(v) for v in y)
    total = len(y)
    weights = []
    for label_id in range(len(SEARCH_CLASSES)):
        weights.append((total / max(counts[label_id], 1)) ** 0.5)
    weights = np.asarray(weights, dtype=np.float32)
    return weights / weights.mean()


def evaluate(model, loader, device, use_amp, amp_dtype):
    import torch

    model.eval()
    preds = []
    gold = []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").numpy().tolist()
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda", dtype=amp_dtype):
                logits = model(**batch).logits
            preds.extend(torch.argmax(logits, dim=-1).cpu().numpy().tolist())
            gold.extend(labels)
    macro_f1 = f1_score(gold, preds, labels=list(range(len(SEARCH_CLASSES))), average="macro", zero_division=0)
    print(classification_report(gold, preds, target_names=SEARCH_CLASSES, digits=4, zero_division=0))
    return macro_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--model-name", default="ibm-granite/granite-embedding-311m-multilingual-r2")
    parser.add_argument("--output-dir", default="./model/search-specialist-fold0")
    parser.add_argument("--split-mode", choices=["group", "stratified", "all"], default="group")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-history-events", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--attn-implementation", default="eager")
    args = parser.parse_args()

    import torch
    import torch._dynamo
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        get_linear_schedule_with_warmup,
        set_seed,
    )

    torch._dynamo.config.suppress_errors = True
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    texts, y, groups = build_data(args.data_dir, args.max_history_events)
    train_texts, val_texts, y_train, y_val, has_val, _, _ = split_train_val(
        texts, y, groups, args.split_mode, args.fold, args.n_splits, args.val_size, args.seed
    )
    if has_val:
        print(
            f"search_train={len(train_texts)} search_val={len(val_texts)} "
            f"split={args.split_mode} fold={args.fold}/{args.n_splits}"
        )
    else:
        print(f"search_train={len(train_texts)} search_val=0 split=all")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(SEARCH_CLASSES),
        id2label=SEARCH_ID2LABEL,
        label2id=SEARCH_LABEL2ID,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    )
    model.to(device)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(
        SearchDataset(train_texts, y_train, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=2,
    )
    val_loader = None
    if has_val:
        val_loader = DataLoader(
            SearchDataset(val_texts, y_val, tokenizer, args.max_length),
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
    use_amp = (args.fp16 or args.bf16) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16

    best_f1 = -1.0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        update_step = 0
        for step, batch in enumerate(train_loader, start=1):
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
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

        train_loss = running_loss / len(train_loader)
        should_save = False
        if has_val:
            macro_f1 = evaluate(model, val_loader, device, use_amp, amp_dtype)
            print(f"epoch={epoch} val_search_macro_f1={macro_f1:.5f}")
            if macro_f1 > best_f1:
                best_f1 = macro_f1
                should_save = True
        else:
            print(f"epoch={epoch} train_loss={train_loss:.4f}")
            should_save = True

        if should_save:
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            meta = {
                "base_model": args.model_name,
                "split_mode": args.split_mode,
                "max_length": args.max_length,
                "max_history_events": args.max_history_events,
                "search_classes": SEARCH_CLASSES,
            }
            if has_val:
                meta["best_val_search_macro_f1"] = best_f1
                meta["fold"] = args.fold
                meta["n_splits"] = args.n_splits
            else:
                meta["epochs"] = args.epochs
                meta["final_epoch"] = epoch
                meta["final_train_loss"] = train_loss
            with open(Path(args.output_dir) / "training_meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            print(f"saved search specialist to {args.output_dir}")


if __name__ == "__main__":
    main()
