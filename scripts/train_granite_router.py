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


def load_exclude_ids(path):
    with open(path, newline="", encoding="utf-8") as f:
        first_line = f.readline()
        f.seek(0)
        if first_line.strip().split(",")[0] == "id":
            return {row["id"] for row in csv.DictReader(f)}
        return {line.strip() for line in f if line.strip()}


def build_data(data_dir, max_history_events, include_open_file_names=False):
    samples = load_jsonl(Path(data_dir) / "train.jsonl")
    labels = load_labels(Path(data_dir) / "train_labels.csv")
    ids = []
    texts = []
    y = []
    groups = []
    for sample in samples:
        sample_id = sample["id"]
        ids.append(sample_id)
        texts.append(render_granite_sample(
            sample, max_history_events=max_history_events, include_open_file_names=include_open_file_names,
        ))
        y.append(LABEL2ID[labels[sample_id]])
        groups.append(session_group(sample_id))
    return (
        np.array(ids, dtype=object),
        np.array(texts, dtype=object),
        np.array(y, dtype=np.int64),
        np.array(groups, dtype=object),
    )


def dump_oof(oof_path, ids, y_true, val_logits, id2label):
    """Save OOF predictions as npz (ids, y_true, classes, logits, probs) with
    columns reordered to ACTION_CLASSES order. Consumed by verify_int8.py,
    blend_eval.py, eval_search_specialist.py."""
    label2id = {id2label[i]: i for i in range(val_logits.shape[1])}
    col_order = [label2id[name] for name in ACTION_CLASSES]
    ordered = val_logits[:, col_order].astype(np.float32)
    shifted = ordered - ordered.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)
    os.makedirs(Path(oof_path).parent, exist_ok=True)
    np.savez(
        oof_path,
        ids=np.array(ids, dtype=object),
        y_true=np.asarray(y_true, dtype=np.int64),
        classes=np.array(ACTION_CLASSES),
        logits=ordered,
        probs=probs,
    )
    print(f"saved OOF to {oof_path} shape={probs.shape}")


def class_weights(y):
    counts = Counter(int(v) for v in y)
    weights = []
    total = len(y)
    n_classes = len(ACTION_CLASSES)
    for label_id in range(n_classes):
        weights.append(total / (n_classes * max(counts[label_id], 1)))
    weights = np.array(weights, dtype=np.float32)
    return weights / weights.mean()


def evaluate(model, loader, device, use_fp16, return_logits=False):
    import torch

    model.eval()
    preds = []
    gold = []
    logit_chunks = []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").numpy().tolist()
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=use_fp16 and device.type == "cuda"):
                logits = model(**batch).logits
            logits = logits.float()
            if return_logits:
                logit_chunks.append(logits.cpu().numpy())
            preds.extend(torch.argmax(logits, dim=-1).cpu().numpy().tolist())
            gold.extend(labels)
    macro = f1_score(gold, preds, labels=list(range(len(ACTION_CLASSES))), average="macro", zero_division=0)
    if return_logits:
        return macro, np.concatenate(logit_chunks, axis=0), np.asarray(gold, dtype=np.int64)
    return macro


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--model-name", default="ibm-granite/granite-embedding-311m-multilingual-r2")
    parser.add_argument("--output-dir", default="./model/granite-311m-fold0")
    parser.add_argument("--split-mode", choices=["group", "all"], default="group",
                        help="'group' = GroupKFold fold (has val); 'all' = train on 100% of data (no val).")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--eval-only", action="store_true",
                        help="Load --output-dir, evaluate the fold val split, dump OOF, exit (no training).")
    parser.add_argument("--oof-path", default="",
                        help="If set, dump OOF npz (ids/y_true/classes/logits/probs) for the val split.")
    parser.add_argument("--exclude-ids-file", default="",
                        help="Drop these ids from the TRAIN split only (val is untouched, so this fold's "
                        "held-out set stays comparable to a run without this flag). Accepts a CSV with an "
                        "'id' column (e.g. find_label_issues.py --output) or a plain one-id-per-line file.")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-history-events", type=int, default=12)
    parser.add_argument("--include-open-file-names", action="store_true",
                        help="Append an 'openfiles=<basenames>' field to [META] on top of the existing "
                        "'open=<count>' field. This is the 'names' variant validated by the "
                        "submit_names_single submission (granite-311m-v2-fold1 + logit bias). Must match "
                        "at inference/tune_granite_bias time or predictions will be wrong.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    # NOTE: pure fp16 (.half()) save overflows some weights to inf on this model and
    # destroys the checkpoint. Default OFF; use --save-dtype bf16 for a safe compact save.
    parser.add_argument("--save-fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--attn-implementation", default="eager",
                        help="granite-embedding (ModernBert) needs 'eager'.")
    args = parser.parse_args()

    import torch
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
    ids, texts, y, groups = build_data(args.data_dir, args.max_history_events, args.include_open_file_names)

    # ---- split ----
    if args.split_mode == "all":
        has_val = False
        train_ids_list = ids.tolist()
        train_texts, y_train = texts.tolist(), y
        val_texts, y_val, val_ids = [], np.array([], dtype=np.int64), []
        print(f"train={len(train_texts)} (split-mode=all, no val)")
    else:
        splitter = GroupKFold(n_splits=args.n_splits)
        splits = list(splitter.split(texts, y, groups))
        train_idx, val_idx = splits[args.fold]
        train_ids_list = ids[train_idx].tolist()
        train_texts, val_texts = texts[train_idx].tolist(), texts[val_idx].tolist()
        y_train, y_val = y[train_idx], y[val_idx]
        val_ids = ids[val_idx].tolist()
        has_val = True
        print(f"train={len(train_texts)} val={len(val_texts)} fold={args.fold}/{args.n_splits}")

    if args.exclude_ids_file:
        exclude_ids = load_exclude_ids(args.exclude_ids_file)
        keep = [i for i, sample_id in enumerate(train_ids_list) if sample_id not in exclude_ids]
        dropped = len(train_texts) - len(keep)
        train_texts = [train_texts[i] for i in keep]
        y_train = y_train[keep]
        print(f"--exclude-ids-file {args.exclude_ids_file}: dropped {dropped} rows from TRAIN "
              f"({dropped / (dropped + len(keep)):.2%}), val split untouched")

    collator = None  # set after tokenizer

    # eval-only: load the saved model, evaluate val split, dump OOF, exit.
    if args.eval_only:
        if not has_val:
            raise SystemExit("--eval-only needs a val split (use --split-mode group)")
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir, use_fast=True, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.output_dir, local_files_only=True, attn_implementation=args.attn_implementation
        ).to(device).eval()
        collator = DataCollatorWithPadding(tokenizer=tokenizer)
        val_loader = DataLoader(
            ActionDataset(val_texts, y_val, tokenizer, args.max_length),
            batch_size=args.eval_batch_size, shuffle=False, collate_fn=collator, num_workers=2,
        )
        macro, val_logits, val_gold = evaluate(model, val_loader, device, args.fp16, return_logits=True)
        print(f"[eval-only] val_macro_f1={macro:.5f}")
        if args.oof_path:
            dump_oof(args.oof_path, val_ids, val_gold, val_logits, model.config.id2label)
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(ACTION_CLASSES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        attn_implementation=args.attn_implementation,
    )
    model.to(device)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(
        ActionDataset(train_texts, y_train, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=2,
    )
    val_loader = None
    if has_val:
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

        train_loss = running_loss / len(train_loader)
        should_save = False
        val_logits = val_gold = None
        if has_val:
            macro_f1, val_logits, val_gold = evaluate(model, val_loader, device, args.fp16, return_logits=True)
            print(f"epoch={epoch} val_macro_f1={macro_f1:.5f}")
            if macro_f1 > best_f1:
                best_f1 = macro_f1
                should_save = True
        else:
            print(f"epoch={epoch} train_loss={train_loss:.4f}")
            should_save = True  # no val -> keep latest each epoch

        if should_save:
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            meta = {
                "base_model": args.model_name,
                "split_mode": args.split_mode,
                "max_length": args.max_length,
                "max_history_events": args.max_history_events,
                "include_open_file_names": args.include_open_file_names,
                "action_classes": ACTION_CLASSES,
            }
            if has_val:
                meta["best_val_macro_f1"] = best_f1
                meta["fold"] = args.fold
                meta["n_splits"] = args.n_splits
            else:
                meta["epochs"] = args.epochs
                meta["final_epoch"] = epoch
            with open(Path(args.output_dir) / "training_meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            print(f"saved model to {args.output_dir}")
            if has_val and args.oof_path and val_logits is not None:
                dump_oof(args.oof_path, val_ids, val_gold, val_logits, model.config.id2label)

    save_dtype = args.save_dtype
    if args.save_fp16:
        save_dtype = "fp16"  # backward-compat with the old flag
    if save_dtype != "fp32" and device.type == "cuda":
        print(f"re-saving best model as {save_dtype}")
        best_model = AutoModelForSequenceClassification.from_pretrained(args.output_dir)
        # bf16 keeps fp32's exponent range so weights never overflow to inf,
        # unlike fp16 (.half()) which craters this model's accuracy.
        best_model.to(torch.bfloat16 if save_dtype == "bf16" else torch.float16)
        best_model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()

