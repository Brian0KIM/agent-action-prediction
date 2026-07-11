"""Verify int8 quantization on a 4-class search specialist (read_file /
grep_search / list_directory / glob_pattern).

Adapted from verify_int8.py for the specialist's 4-class problem: same idea
(measure macro-F1 retention fp vs int8, and real on-disk size), but rebuilds
its OWN fold validation set via train_search_specialist.py's SEARCH_CLASSES /
build_data / split_train_val instead of assuming a 14-class baseline npz.

Uses bitsandbytes 8-bit (--method bnb) or torch dynamic int8 (--method
dynamic), same as verify_int8.py. This is NOT byte-identical to whatever
custom int8 scheme submit_ens4_706 used (per-row absmax -> int8 + fp16 scale,
manually dequantized) -- treat the size number here as a same-ballpark
estimate, not the exact number that scheme would produce. If you get the
actual ens4 quantization script, re-run with that for a precise figure.

Example
-------
    python scripts/verify_specialist_int8.py \
        --model-dir ./model/search-specialist-gte-fold0 \
        --data-dir ../data --fold 0 --n-splits 5 \
        --method bnb --dtype bf16
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from action_router.features import render_granite_sample, session_group  # noqa: E402
from action_router.split import split_train_val  # noqa: E402

SEARCH_CLASSES = ["read_file", "grep_search", "list_directory", "glob_pattern"]
SEARCH_LABEL2ID = {label: idx for idx, label in enumerate(SEARCH_CLASSES)}
DTYPE_MAP = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_labels(path):
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        return {row["id"]: row["action"] for row in csv.DictReader(f)}


def build_val_set(data_dir, fold, n_splits, max_history_events):
    """Same filtering as train_search_specialist.py's build_data: keep only
    rows whose true label is one of the 4 search classes, fold-split by
    session group."""
    samples = load_jsonl(Path(data_dir) / "train.jsonl")
    labels = load_labels(Path(data_dir) / "train_labels.csv")
    ids, texts, y, groups = [], [], [], []
    for sample in samples:
        action = labels[sample["id"]]
        if action not in SEARCH_LABEL2ID:
            continue
        ids.append(sample["id"])
        texts.append(render_granite_sample(sample, max_history_events=max_history_events))
        y.append(SEARCH_LABEL2ID[action])
        groups.append(session_group(sample["id"]))
    ids = np.array(ids, dtype=object)
    texts = np.array(texts, dtype=object)
    y = np.array(y, dtype=np.int64)
    groups = np.array(groups, dtype=object)
    _, val_texts, _, y_val, has_val, _, val_idx = split_train_val(
        texts, y, groups, "group", fold, n_splits, 0.2, 42,
    )
    if not has_val:
        raise SystemExit("no val split produced; check --fold/--n-splits")
    val_ids = ids[val_idx].tolist()
    return val_ids, val_texts, y_val


def macro_f1(y_true, y_pred, n_classes=len(SEARCH_CLASSES)):
    scores = []
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        scores.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(scores))


def dir_size_mb(path):
    total = sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())
    return total / 1e6


def run_inference(model, tokenizer, texts, device, max_length, batch_size, id2label):
    import torch
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding

    class DS:
        def __len__(self):
            return len(texts)

        def __getitem__(self, i):
            return tokenizer(texts[i], truncation=True, max_length=max_length, padding=False)

    loader = DataLoader(DS(), batch_size=batch_size, shuffle=False,
                         collate_fn=DataCollatorWithPadding(tokenizer=tokenizer))
    all_logits = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits.float()
            all_logits.append(logits.cpu().numpy())
    out = np.concatenate(all_logits, axis=0)

    label2id = {id2label[i]: i for i in range(out.shape[1])}
    col_order = [label2id[name] for name in SEARCH_CLASSES]
    if col_order != list(range(out.shape[1])):
        print(f"note: reordering logit columns to SEARCH_CLASSES order: {col_order}")
    return out[:, col_order]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--max-history-events", type=int, default=12)
    parser.add_argument("--method", choices=["bnb", "dynamic"], default="bnb")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dtype", choices=list(DTYPE_MAP), default="bf16")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    args = parser.parse_args()

    print("=== verify_specialist_int8 ===")
    import torch
    try:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
    except Exception:  # noqa: BLE001
        pass
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch_dtype = getattr(torch, DTYPE_MAP[args.dtype])
    ids, texts, y_true = build_val_set(args.data_dir, args.fold, args.n_splits, args.max_history_events)
    print(f"val set: {len(ids)} samples (fold {args.fold}/{args.n_splits}, 4 search classes only)")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True,
                                               trust_remote_code=args.trust_remote_code)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    orig_mb = dir_size_mb(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir, local_files_only=True, torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    fp_logits = run_inference(model, tokenizer, texts, device, args.max_length, args.batch_size, model.config.id2label)
    fp_macro = macro_f1(y_true, fp_logits.argmax(1))
    print(f"[{args.dtype} baseline] macro-F1 = {fp_macro:.4f}  size = {orig_mb:.1f} MB")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if args.method == "bnb":
        from transformers import BitsAndBytesConfig
        qconf = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=["classifier", "score", "pre_classifier"])
        q_model = AutoModelForSequenceClassification.from_pretrained(
            args.model_dir, local_files_only=True, quantization_config=qconf,
            torch_dtype=torch_dtype, device_map="auto", trust_remote_code=args.trust_remote_code,
        )
        q_device = next(q_model.parameters()).device
        int8_logits = run_inference(q_model, tokenizer, texts, q_device, args.max_length, args.batch_size, q_model.config.id2label)
        int8_dir = Path(args.model_dir).parent / (Path(args.model_dir).name + "-int8")
        try:
            q_model.save_pretrained(int8_dir)
            tokenizer.save_pretrained(int8_dir)
            int8_mb = dir_size_mb(int8_dir)
        except Exception as e:  # noqa: BLE001
            int8_mb = float("nan")
            print(f"warning: could not serialize int8 model for size measurement: {e}")
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_dir, local_files_only=True, trust_remote_code=args.trust_remote_code,
        ).to("cpu").eval()
        q_model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        int8_logits = run_inference(q_model, tokenizer, texts, torch.device("cpu"), args.max_length, args.batch_size, model.config.id2label)
        int8_dir = Path(args.model_dir).parent / (Path(args.model_dir).name + "-int8-dynamic")
        int8_dir.mkdir(parents=True, exist_ok=True)
        torch.save(q_model.state_dict(), int8_dir / "int8_state.pt")
        int8_mb = dir_size_mb(int8_dir)

    int8_macro = macro_f1(y_true, int8_logits.argmax(1))

    print("\n================ specialist int8 verification ================")
    print(f"model            : {args.model_dir}")
    print(f"method           : {args.method}")
    print(f"macro-F1  {args.dtype:<6}  : {fp_macro:.4f}")
    print(f"macro-F1  int8    : {int8_macro:.4f}")
    print(f"macro-F1  delta   : {int8_macro - fp_macro:+.4f}")
    print(f"size  {args.dtype:<6} (dir) : {orig_mb:8.1f} MB")
    print(f"size  int8 (dir)  : {int8_mb:8.1f} MB")
    print("================================================================")


if __name__ == "__main__":
    main()
