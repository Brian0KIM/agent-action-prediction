"""Verify int8 quantization on a fine-tuned sequence-classification router.

Purpose
-------
De-risk the 2-way (granite + bge) ensemble under the 1GB packaging limit by
measuring, for ONE model, both:

  1. accuracy retention: macro-F1 fp16 vs int8 on the exact same eval set, and
  2. on-disk size: original model dir vs saved int8 model dir.

The eval set is reconstructed from an existing baseline `.npz` dump (e.g.
`granite-fold0.npz`) by matching its `ids` back to the training JSONL, so the
int8 numbers are directly comparable to the fp baseline already stored in the
npz (same samples, same order, same y_true).

The int8 probabilities are re-dumped to `--out-npz` with the same schema
(ids, y_true, classes, logits, probs), so `blend_eval.py` can consume them and
you can measure the *blend* macro under int8 directly.

Quantization methods
--------------------
  --method bnb      bitsandbytes 8-bit (GPU; closest to T4 deploy). Default.
  --method dynamic  torch dynamic int8 on Linear layers (CPU-only, no extra deps).

Granite loads as ModernBertForSequenceClassification and needs
transformers==4.48.3; bge-m3 loads as XLMRobertaForSequenceClassification and
works on the default stack. Run each model in its matching environment.

Example
-------
    # granite (transformers==4.48.3 env)
    python scripts/verify_int8.py \
        --model-dir ./model/granite-311m-fold0 \
        --data-dir ./data \
        --baseline-npz ./granite-fold0.npz \
        --renderer granite \
        --method bnb \
        --out-npz ./granite-fold0-int8.npz

    # bge-m3
    python scripts/verify_int8.py \
        --model-dir ./model/bge-m3-fold0 \
        --data-dir ./data \
        --baseline-npz ./bge-m3-fold0.npz \
        --renderer granite \
        --method bnb \
        --out-npz ./bge-m3-fold0-int8.npz
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from action_router.constants import ACTION_CLASSES  # noqa: E402
from action_router.features import render_granite_text, render_sample  # noqa: E402


def resolve_feature_mode(model_dir, override):
    """feature_mode decides the input text format (granite / granite_v2 / v3).
    Mismatch => the model sees the wrong text => near-random macro.
    Priority: explicit override > training_meta.json > 'granite'."""
    if override:
        return override
    meta = Path(model_dir) / "training_meta.json"
    if meta.exists():
        with open(meta, encoding="utf-8") as f:
            fm = json.load(f).get("feature_mode")
            if fm:
                return fm
    return "granite"


def make_renderer(kind, feature_mode):
    if kind == "e5":
        return lambda s: render_sample(s, max_history=8)
    return lambda s: render_granite_text(s, max_history_events=12, feature_mode=feature_mode)


def macro_f1(y_true, y_pred, n_classes=len(ACTION_CLASSES)):
    """Unweighted mean per-class F1. Kept dependency-free (no sklearn)."""
    scores = []
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        scores.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(scores))


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dir_size_mb(path):
    total = sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())
    return total / 1e6


def build_eval_set(data_dir, baseline_npz, renderer):
    """Reconstruct texts for exactly the ids stored in the baseline npz."""
    base = np.load(baseline_npz, allow_pickle=True)
    ids = [str(i) for i in base["ids"]]
    y_true = base["y_true"].astype(np.int64)
    base_probs = base["probs"]

    samples = load_jsonl(Path(data_dir) / "train.jsonl")
    by_id = {str(s["id"]): s for s in samples}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise SystemExit(
            f"{len(missing)} ids from {baseline_npz} not found in train.jsonl "
            f"(first few: {missing[:3]}). Check --data-dir."
        )
    texts = [renderer(by_id[i]) for i in ids]
    return ids, texts, y_true, base_probs


def load_logit_bias(model_dir, id2label):
    path = Path(model_dir) / "logit_bias.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    bias_map = payload.get("bias", {})
    return [float(bias_map.get(id2label[idx], 0.0)) for idx in range(len(id2label))]


DTYPE_MAP = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}


def run_inference(model, tokenizer, texts, device, max_length, batch_size, bias, id2label, autocast=False):
    """Run in the model's own dtype (no autocast) and return logits in
    ACTION_CLASSES column order.

    Two things this guards against:
    - autocast forces fp16, which makes ModernBert (granite-embedding) overflow
      to NaN. Loading in bf16/fp32 and running WITHOUT autocast avoids it.
    - a model's config.id2label order may NOT equal ACTION_CLASSES. The training
      script reorders OOF columns defensively; we must match, or argmax indices
      won't line up with the npz y_true (-> near-random macro).
    """
    import torch
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding

    class DS:
        def __len__(self):
            return len(texts)

        def __getitem__(self, i):
            return tokenizer(texts[i], truncation=True, max_length=max_length, padding=False)

    loader = DataLoader(
        DS(),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    all_logits = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=autocast and device.type == "cuda"):
                logits = model(**batch).logits
            logits = logits.float()
            if bias is not None:  # bias is in model (id2label) order, applied pre-reorder
                logits = logits + bias
            all_logits.append(logits.cpu().numpy())
    out = np.concatenate(all_logits, axis=0)

    # Reorder columns model-order -> ACTION_CLASSES order.
    label2id = {id2label[i]: i for i in range(out.shape[1])}
    col_order = [label2id[name] for name in ACTION_CLASSES]
    if col_order != list(range(out.shape[1])):
        print(f"note: reordering logit columns to ACTION_CLASSES (model order != canonical): {col_order}")
    return out[:, col_order]


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--baseline-npz", required=True,
                        help="Existing fp dump (ids/y_true/probs) to compare against and match eval set.")
    parser.add_argument("--renderer", choices=["granite", "e5"], default="granite")
    parser.add_argument("--feature-mode", default=None,
                        help="granite/granite_v2/granite_v3. Default: read from model's training_meta.json.")
    parser.add_argument("--method", choices=["bnb", "dynamic"], default="bnb")
    parser.add_argument("--out-npz", default=None)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dtype", choices=list(DTYPE_MAP), default="bf16",
                        help="Inference dtype for non-int8 parts. Use bf16 for granite/ModernBert (fp16 -> NaN).")
    parser.add_argument("--keep-head-fp", action="store_true", default=True,
                        help="Keep the classification head out of int8 (bnb). On by default; encoder heads collapse if quantized.")
    parser.add_argument("--quantize-head", dest="keep_head_fp", action="store_false",
                        help="Force-quantize the head too (to reproduce the collapse).")
    parser.add_argument("--attn-implementation", default="eager",
                        help="granite/ModernBert needs 'eager' (matches make_teacher_logits.py).")
    parser.add_argument("--autocast", action="store_true",
                        help="Wrap fp recompute in fp16 autocast (matches the fast submit script). "
                             "Forces fp32 load. Use to measure the fast submission's real accuracy.")
    parser.add_argument("--skip-int8", action="store_true", help="Only do the fp recompute; skip the int8 stage.")
    parser.add_argument("--skip-fp", action="store_true",
                        help="Trust baseline npz as the fp reference instead of recomputing it.")
    args = parser.parse_args()

    print("=== verify_int8 v4 (feature_mode + reorder + eager + diagnostics) ===")

    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    # ModernBert (granite) triggers torch.compile internally; dynamo can't trace
    # bitsandbytes' int8 autograd fn and errors. Fall back to eager.
    try:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
    except Exception:  # noqa: BLE001
        pass

    # reference_compile / eager are ModernBert-only kwargs; XLM-R (bge) rejects
    # reference_compile with a TypeError. Only pass them for ModernBert.
    _cfg = AutoConfig.from_pretrained(args.model_dir, local_files_only=True)
    _archs = getattr(_cfg, "architectures", None) or []
    is_modernbert = "modernbert" in (getattr(_cfg, "model_type", "") or "").lower() or any(
        "modernbert" in a.lower() for a in _archs
    )
    extra_kwargs = {"reference_compile": False, "attn_implementation": args.attn_implementation} if is_modernbert else {}
    print(f"model_type={getattr(_cfg, 'model_type', None)} is_modernbert={is_modernbert} extra_kwargs={list(extra_kwargs)}")

    torch_dtype = getattr(torch, DTYPE_MAP[args.dtype])
    # Module-name fragments bnb should NOT quantize (classification head + norms).
    skip_modules = ["classifier", "score", "pre_classifier", "head"] if args.keep_head_fp else None

    feature_mode = resolve_feature_mode(args.model_dir, args.feature_mode)
    renderer = make_renderer(args.renderer, feature_mode)
    print(f"renderer={args.renderer}  feature_mode={feature_mode}")
    ids, texts, y_true, base_probs = build_eval_set(args.data_dir, args.baseline_npz, renderer)
    print(f"eval set: {len(ids)} samples reconstructed from {args.baseline_npz}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- baseline (fp) reference ----
    base_macro = macro_f1(y_true, base_probs.argmax(1))
    print(f"[baseline npz] macro-F1 = {base_macro:.4f}")

    fp_macro = base_macro
    if not args.skip_fp:
        # --autocast reproduces the fast submit (fp32 master + fp16 autocast).
        fp_load_dtype = torch.float32 if args.autocast else torch_dtype
        label = f"fp32+autocast(fp16)" if args.autocast else args.dtype
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_dir, local_files_only=True, torch_dtype=fp_load_dtype, **extra_kwargs
        )
        model.to(device)
        print(f"config.id2label = {dict(model.config.id2label)}")
        bias_vals = load_logit_bias(args.model_dir, model.config.id2label)
        bias = torch.tensor(bias_vals, device=device) if bias_vals else None
        fp_logits = run_inference(model, tokenizer, texts, device, args.max_length, args.batch_size, bias, model.config.id2label, autocast=args.autocast)
        fp_macro = macro_f1(y_true, fp_logits.argmax(1))
        pred = fp_logits.argmax(1)
        dist = {ACTION_CLASSES[c]: int((pred == c).sum()) for c in range(len(ACTION_CLASSES))}
        print(f"pred distribution: {dist}")
        print(f"[{label} recompute] macro-F1 = {fp_macro:.4f}  (drift vs npz: {fp_macro - base_macro:+.4f})")
        del model
        if args.skip_int8:
            print("--skip-int8 set; done.")
            return
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- int8 ----
    orig_mb = dir_size_mb(args.model_dir)
    if args.method == "bnb":
        try:
            from transformers import BitsAndBytesConfig
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"bitsandbytes/transformers BitsAndBytesConfig unavailable: {e}. Try --method dynamic.")
        qconf = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=skip_modules)
        print(f"bnb 8bit  dtype={args.dtype}  skip_modules={skip_modules}")
        q_model = AutoModelForSequenceClassification.from_pretrained(
            args.model_dir, local_files_only=True, quantization_config=qconf,
            torch_dtype=torch_dtype, device_map="auto", **extra_kwargs,
        )
        q_device = next(q_model.parameters()).device
        bias_vals = load_logit_bias(args.model_dir, q_model.config.id2label)
        bias = torch.tensor(bias_vals, device=q_device) if bias_vals else None
        int8_logits = run_inference(q_model, tokenizer, texts, q_device, args.max_length, args.batch_size, bias, q_model.config.id2label)
        int8_dir = Path(args.model_dir).parent / (Path(args.model_dir).name + "-int8")
        try:
            q_model.save_pretrained(int8_dir)
            tokenizer.save_pretrained(int8_dir)
            int8_mb = dir_size_mb(int8_dir)
        except Exception as e:  # noqa: BLE001
            int8_mb = float("nan")
            print(f"warning: could not serialize int8 model for size measurement: {e}")
    else:  # dynamic (CPU)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_dir, local_files_only=True, **extra_kwargs
        )
        model.to("cpu").eval()
        q_model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        bias_vals = load_logit_bias(args.model_dir, model.config.id2label)
        bias = torch.tensor(bias_vals) if bias_vals else None
        int8_logits = run_inference(q_model, tokenizer, texts, torch.device("cpu"), args.max_length, args.batch_size, bias, model.config.id2label)
        int8_dir = Path(args.model_dir).parent / (Path(args.model_dir).name + "-int8-dynamic")
        int8_dir.mkdir(parents=True, exist_ok=True)
        torch.save(q_model.state_dict(), int8_dir / "int8_state.pt")
        int8_mb = dir_size_mb(int8_dir)

    int8_macro = macro_f1(y_true, int8_logits.argmax(1))
    int8_probs = softmax(int8_logits)

    print("\n================ int8 verification ================")
    print(f"model            : {args.model_dir}")
    print(f"method           : {args.method}")
    print(f"macro-F1  fp      : {fp_macro:.4f}")
    print(f"macro-F1  int8    : {int8_macro:.4f}")
    print(f"macro-F1  delta   : {int8_macro - fp_macro:+.4f}")
    print(f"size  fp   (dir)  : {orig_mb:8.1f} MB")
    print(f"size  int8 (dir)  : {int8_mb:8.1f} MB")
    print("===================================================")

    if args.out_npz:
        np.savez(
            args.out_npz,
            ids=np.array(ids, dtype=object),
            y_true=y_true,
            classes=np.array(ACTION_CLASSES),
            logits=int8_logits.astype(np.float32),
            probs=int8_probs.astype(np.float32),
        )
        print(f"saved int8 dump -> {args.out_npz}")


if __name__ == "__main__":
    main()
