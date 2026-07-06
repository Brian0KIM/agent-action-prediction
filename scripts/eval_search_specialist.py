"""Single-fold OOF eval of a search (exploration) specialist cascade.

Idea
----
The base router (granite v2) is weakest on the 4 exploration actions
(read_file / grep_search / list_directory / glob_pattern). This script tests a
cascade: wherever the base predicts one of those 4 (optionally also where the
base is uncertain), hand the sample to a 4-way specialist and let it re-decide.

Base predictions come straight from an existing OOF npz (e.g.
granite-fold0.npz) -- we do NOT recompute base logits, so there is no renderer
mismatch and no GPU needed for the base. Only the specialist runs on GPU, and
only on the gated subset.

Reports base macro-F1 vs cascade macro-F1 on the same fold0-val set, plus the
per-class F1 change on the 4 exploration classes.

Example
-------
    python scripts/eval_search_specialist.py \
        --base-npz output/oof/granite-fold0.npz \
        --specialist-dir ./model/search-specialist-granite-fold0 \
        --data-dir ../data --dtype bf16
    # compare candidates by swapping --specialist-dir (e5-small / e5-base / granite)
    # average several specialists: --specialist-dir a,b,c
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from action_router.constants import ACTION_CLASSES  # noqa: E402
from action_router.features import render_granite_sample  # noqa: E402

SEARCH_CLASSES = ["read_file", "grep_search", "list_directory", "glob_pattern"]
SEARCH_SET = set(SEARCH_CLASSES)
SEARCH_TO_GLOBAL = np.asarray([ACTION_CLASSES.index(c) for c in SEARCH_CLASSES], dtype=np.int64)
DTYPE_MAP = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}


def macro_f1(y_true, y_pred, n_classes=len(ACTION_CLASSES)):
    fs = []
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        fs.append(2 * p * r / (p + r) if (p + r) else 0.0)
    return float(np.mean(fs)), fs


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_texts(data_dir, ids):
    by_id = {str(s["id"]): s for s in load_jsonl(Path(data_dir) / "train.jsonl")}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise SystemExit(f"{len(missing)} ids not in train.jsonl (e.g. {missing[:3]}). Check --data-dir.")
    return [render_granite_sample(by_id[i], max_history_events=12) for i in ids]


def modernbert_kwargs(model_dir, attn):
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    archs = getattr(cfg, "architectures", None) or []
    is_mb = "modernbert" in (getattr(cfg, "model_type", "") or "").lower() or any(
        "modernbert" in a.lower() for a in archs
    )
    return ({"reference_compile": False, "attn_implementation": attn} if is_mb else {}), is_mb


def predict_search(model_dir, texts, args, torch_dtype):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

    extra, is_mb = modernbert_kwargs(model_dir, args.attn_implementation)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True, torch_dtype=torch_dtype, **extra
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # column reorder: model order -> SEARCH_CLASSES order (defensive, like train script)
    id2label = model.config.id2label
    label2id = {id2label[i]: i for i in range(len(id2label))}
    col_order = [label2id[c] for c in SEARCH_CLASSES]

    class DS:
        def __len__(self):
            return len(texts)

        def __getitem__(self, i):
            return tokenizer(texts[i], truncation=True, max_length=args.max_length, padding=False)

    loader = DataLoader(DS(), batch_size=args.batch_size, shuffle=False,
                        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer))
    out = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out.append(model(**batch).logits.float().cpu().numpy())
    logits = np.concatenate(out, axis=0)[:, col_order]
    print(f"  specialist {Path(model_dir).name}: modernbert={is_mb} col_order={col_order}")
    return logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-npz", required=True, help="base OOF dump (ids,y_true,probs/logits in ACTION_CLASSES order)")
    parser.add_argument("--specialist-dir", required=True, help="one dir, or comma-separated to average")
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--dtype", choices=list(DTYPE_MAP), default="bf16")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--margin-threshold", type=float, default=0.0,
                        help="also send base samples whose top-2 logit margin <= this to the specialist")
    parser.add_argument("--blend-weights", default="0,0.25,0.5,0.75,1.0",
                        help="comma list of blend weights on specialist. 0=base, 1=hard override.")
    args = parser.parse_args()

    print("=== eval_search_specialist v1 ===")
    import torch
    try:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
    except Exception:  # noqa: BLE001
        pass
    torch_dtype = getattr(torch, DTYPE_MAP[args.dtype])

    base = np.load(args.base_npz, allow_pickle=True)
    ids = [str(i) for i in base["ids"]]
    y_true = base["y_true"].astype(np.int64)
    base_logits = base["logits"].astype(np.float32) if "logits" in base else np.log(base["probs"] + 1e-9)
    base_pred = base_logits.argmax(1)
    base_macro, base_fs = macro_f1(y_true, base_pred)
    print(f"base: n={len(ids)}  macro-F1={base_macro:.4f}")

    # ---- gate ----
    base_labels = np.asarray(ACTION_CLASSES, dtype=object)[base_pred]
    selected = np.isin(base_labels, list(SEARCH_SET))
    if args.margin_threshold > 0:
        order = np.argsort(-base_logits, axis=1)
        top1 = base_logits[np.arange(len(base_logits)), order[:, 0]]
        top2 = base_logits[np.arange(len(base_logits)), order[:, 1]]
        selected = selected | ((top1 - top2) <= args.margin_threshold)
    sel_idx = np.flatnonzero(selected)
    print(f"gated (base predicts explore{' or low-margin' if args.margin_threshold > 0 else ''}): "
          f"{len(sel_idx)} / {len(ids)}")
    if len(sel_idx) == 0:
        raise SystemExit("nothing selected; check base npz / margin.")

    # ---- specialist(s) on selected only ----
    texts_all = build_texts(args.data_dir, ids)
    sel_texts = [texts_all[i] for i in sel_idx]
    spec_dirs = [d.strip() for d in args.specialist_dir.split(",") if d.strip()]
    spec_logits = None
    for d in spec_dirs:
        lg = predict_search(d, sel_texts, args, torch_dtype)
        spec_logits = lg if spec_logits is None else spec_logits + lg
    spec_logits /= len(spec_dirs)

    def softmax(x):
        x = x - x.max(axis=1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=1, keepdims=True)

    # probs over the 4 exploration classes (SEARCH_CLASSES order) for selected rows
    base_expl = softmax(base_logits[sel_idx][:, SEARCH_TO_GLOBAL])
    spec_expl = softmax(spec_logits)

    def eval_weight(w):
        """w=0 -> base ranking (no change); w=1 -> hard override; between -> soft blend."""
        blended = (1 - w) * base_expl + w * spec_expl
        choice = SEARCH_TO_GLOBAL[blended.argmax(1)]
        fp = base_pred.copy()
        fp[sel_idx] = choice
        m, fs = macro_f1(y_true, fp)
        fixed = int(np.sum((fp[sel_idx] == y_true[sel_idx]) & (base_pred[sel_idx] != y_true[sel_idx])))
        broke = int(np.sum((fp[sel_idx] != y_true[sel_idx]) & (base_pred[sel_idx] == y_true[sel_idx])))
        return m, fs, fixed, broke

    weights = [float(x) for x in args.blend_weights.split(",")]
    print("\n================ search specialist: prob-blend sweep ================")
    print(f"specialist(s) : {', '.join(Path(d).name for d in spec_dirs)}")
    print(f"base macro-F1 : {base_macro:.4f}   (gated {len(sel_idx)}/{len(ids)})")
    best = (-1.0, None, None)
    for w in weights:
        m, fs, fixed, broke = eval_weight(w)
        tag = "  <- hard override" if w == 1.0 else ("  (base)" if w == 0.0 else "")
        print(f"  w={w:.2f}  macro={m:.4f}  ({m - base_macro:+.4f})  fixed={fixed} broke={broke} net={fixed - broke:+d}{tag}")
        if m > best[0]:
            best = (m, w, fs)
    best_m, best_w, best_fs = best
    print(f"\nbest: w={best_w:.2f}  macro={best_m:.4f}  (delta {best_m - base_macro:+.4f})")
    print("exploration per-class F1 (base -> best blend):")
    for c in SEARCH_CLASSES:
        i = ACTION_CLASSES.index(c)
        print(f"  {c:16s} {base_fs[i]:.3f} -> {best_fs[i]:.3f}  ({best_fs[i] - base_fs[i]:+.3f})")


if __name__ == "__main__":
    main()
