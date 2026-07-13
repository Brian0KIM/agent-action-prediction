"""Train BAAI/bge-m3 as a 14-class action classifier.

Same recipe/conventions as train_gte_router.py (full fine-tune, default
linear head, --epochs 3 / --max-length 512 defaults, full untruncated
history via qwen_serialize's richargs/richmeta, --split-mode {group,all} /
--oof-path / --exclude-ids-file). One difference from GTE: bge-m3 is a
standard XLM-RoBERTa architecture, natively supported by transformers --
trust_remote_code defaults to False here (harmless to pass True, but not
required, unlike gte-multilingual-base's custom modeling code).

Bias tuning is a separate step: see tune_bge_bias.py.

Example:
    python scripts/train_bge_router.py \
      --data-dir ../data --output-dir ./model/bge-full-richargs-ls \
      --split-mode group --fold 0 --n-splits 5 \
      --oof-path output/oof/bge-fold0.npz

    python scripts/train_bge_router.py \
      --data-dir ../data --output-dir ./model/bge-full-richargs-ls-all \
      --split-mode all
"""
import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from action_router.constants import ACTION_CLASSES, ID2LABEL, LABEL2ID
from action_router.features import session_group
from action_router.qwen_serialize import SERIALIZE_VARIANTS, serialize


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


def build_data(data_dir, variant):
    samples = load_jsonl(Path(data_dir) / "train.jsonl")
    labels = load_labels(Path(data_dir) / "train_labels.csv")
    var_kw = SERIALIZE_VARIANTS[variant]
    ids, texts, y, groups = [], [], [], []
    for sample in samples:
        sample_id = sample["id"]
        ids.append(sample_id)
        texts.append(serialize(sample, max_hist=None, **var_kw))
        y.append(LABEL2ID[labels[sample_id]])
        groups.append(session_group(sample_id))
    return (
        np.array(ids, dtype=object),
        np.array(texts, dtype=object),
        np.array(y, dtype=np.int64),
        np.array(groups, dtype=object),
    )


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
            self.texts[idx], truncation=True, max_length=self.max_length, padding=False,
        )
        encoded["labels"] = int(self.labels[idx])
        return encoded


def dump_oof(oof_path, ids, y_true, val_logits, id2label):
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


def evaluate(model, loader, device, use_amp, amp_dtype):
    import torch
    from sklearn.metrics import f1_score

    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").numpy().tolist()
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda", dtype=amp_dtype):
                logits = model(**batch).logits
            all_logits.append(logits.float().cpu().numpy())
            all_labels.extend(labels)
    logits = np.concatenate(all_logits, axis=0)
    preds = logits.argmax(axis=1)
    macro = f1_score(all_labels, preds, labels=list(range(len(ACTION_CLASSES))), average="macro", zero_division=0)
    return macro, logits, np.array(all_labels, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--model-name", default="BAAI/bge-m3")
    parser.add_argument("--output-dir", default="./model/bge-full-richargs-ls")
    parser.add_argument("--split-mode", choices=["group", "all"], default="group",
                        help="'group' = GroupKFold fold (has val, needed for tune_bge_bias.py "
                        "afterwards); 'all' = train on 100% of data (no val, no calibration possible).")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--variant", choices=list(SERIALIZE_VARIANTS), default="richargs")
    parser.add_argument("--oof-path", default="",
                        help="If set, dump OOF npz (ids/y_true/classes/logits/probs) for the val split. "
                        "Requires --split-mode group.")
    parser.add_argument("--exclude-ids-file", default="",
                        help="Drop these ids from TRAIN only (val untouched). CSV with an 'id' "
                        "column, or one id per line.")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Not required for bge-m3 (standard XLM-RoBERTa arch), unlike "
                        "gte-multilingual-base. Off by default.")
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
    ids, texts, y, groups = build_data(args.data_dir, args.variant)

    if args.split_mode == "all":
        if args.oof_path:
            raise SystemExit("--oof-path needs a held-out val split; use --split-mode group.")
        has_val = False
        train_ids_list = ids.tolist()
        train_texts, y_train = texts.tolist(), y
        val_texts, y_val, val_ids = [], np.array([], dtype=np.int64), []
        print(f"train={len(train_texts)} (split-mode=all, no val -- can't compute local macro-F1)")
    else:
        splitter = GroupKFold(n_splits=args.n_splits)
        splits = list(splitter.split(texts, y, groups))
        train_idx, val_idx = splits[args.fold]
        train_ids_list = ids[train_idx].tolist()
        train_texts, val_texts = texts[train_idx].tolist(), texts[val_idx].tolist()
        y_train, y_val = y[train_idx], y[val_idx]
        val_ids = ids[val_idx].tolist()
        has_val = True
        print(f"train={len(train_texts)} val={len(val_texts)} fold={args.fold}/{args.n_splits} variant={args.variant}")

    if args.exclude_ids_file:
        exclude_ids = load_exclude_ids(args.exclude_ids_file)
        keep = [i for i, sample_id in enumerate(train_ids_list) if sample_id not in exclude_ids]
        dropped = len(train_texts) - len(keep)
        train_texts = [train_texts[i] for i in keep]
        y_train = y_train[keep]
        print(f"--exclude-ids-file {args.exclude_ids_file}: dropped {dropped} rows from TRAIN "
              f"({dropped / (dropped + len(keep)):.2%}), val split untouched")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=args.trust_remote_code)
    tokenizer.truncation_side = "right"

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(ACTION_CLASSES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
        trust_remote_code=args.trust_remote_code,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.to(device)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(
        ActionDataset(train_texts, y_train, tokenizer, args.max_length),
        batch_size=args.batch_size, shuffle=True, collate_fn=collator, num_workers=2,
    )
    val_loader = None
    if has_val:
        val_loader = DataLoader(
            ActionDataset(val_texts, y_val, tokenizer, args.max_length),
            batch_size=args.eval_batch_size, shuffle=False, collate_fn=collator, num_workers=2,
        )

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    update_steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = update_steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * args.warmup_ratio)),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16 and device.type == "cuda")
    use_amp = args.fp16 and device.type == "cuda"
    amp_dtype = torch.float16

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
                    print(f"epoch={epoch} update={update_step}/{update_steps_per_epoch} "
                          f"loss={running_loss / step:.4f}")

        train_loss = running_loss / len(train_loader)
        should_save = False
        val_logits, val_gold = None, None
        if has_val:
            macro_f1, val_logits, val_gold = evaluate(model, val_loader, device, use_amp, amp_dtype)
            print(f"epoch={epoch} val_macro_f1={macro_f1:.5f}")
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
                "variant": args.variant,
                "max_length": args.max_length,
                "label_smoothing": args.label_smoothing,
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

    if args.save_dtype != "fp32" and device.type == "cuda":
        print(f"re-saving best model as {args.save_dtype}")
        best_model = AutoModelForSequenceClassification.from_pretrained(
            args.output_dir, trust_remote_code=args.trust_remote_code,
        )
        best_model.to(torch.bfloat16 if args.save_dtype == "bf16" else torch.float16)
        best_model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
