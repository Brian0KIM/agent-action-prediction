"""Quantize a trained checkpoint into the exact int8 format submit_ens4_706's
script.py `load_member()` expects, so the output folder can be dropped
straight into an ens4/ens5-style zip.

Format (per load_member): `model_int8.safetensors` holds, for each quantized
2D weight tensor `k`, an int8 tensor `k` (same shape) plus a companion fp16
1D tensor `k__scale` of shape (rows,) -- one scale per OUTPUT row (per-channel
absmax quantization). Dequant: `v.float() * scale.unsqueeze(-1) -> fp16`.
Non-2D tensors (biases, norms, embeddings if --keep-embeddings-fp) are stored
as-is in fp16, unquantized.

This does NOT do vocab pruning / remap.npy generation -- that's a separate,
higher-risk step because ens4's 3 granite members currently SHARE one
tokenizer + remap.npy, and this member's own token-usage-driven pruning would
differ from theirs. Ship this member with its full embedding table (bigger,
but safe) unless/until a per-member remap is set up.

Verifies round-trip: dequantized model's predictions vs the original fp
model's predictions on a small held-out sample, reporting max/mean logit
drift and argmax agreement, so you're not shipping a silent regression.

Example
-------
    python scripts/quantize_int8_member.py \
        --model-dir ./model/granite-311m-all-names-lr5e5 \
        --output-dir ./model/member_4_all_lr5e5_int8 \
        --variant richargs \
        --data-dir ../data --verify-samples 500
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from action_router.features import render_granite_sample  # noqa: E402


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dir_size_mb(path):
    total = sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())
    return total / 1e6


def quantize_tensor(t):
    """Per-row (per-output-channel) symmetric absmax int8 quantization."""
    import torch
    scale = t.abs().amax(dim=1).clamp(min=1e-8) / 127.0
    q = torch.round(t / scale.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    return q, scale.to(torch.float16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variant", required=True, choices=["richargs", "richmeta"],
                        help="Written to serialize_variant.json, read by ens4/ens5 script.py.")
    parser.add_argument("--min-quantize-numel", type=int, default=1_000_000,
                        help="Only quantize 2D tensors with at least this many elements "
                        "(skips tiny 2D tensors where int8 error isn't worth the size win).")
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--verify-samples", type=int, default=500,
                        help="0 to skip the round-trip accuracy/drift check.")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--include-open-file-names", action="store_true")
    parser.add_argument("--max-history-events", type=int, default=12)
    args = parser.parse_args()

    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = Path(args.model_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True, torch_dtype=torch.float32)
    model.eval()
    sd = model.state_dict()

    out_sd = {}
    n_quantized, n_kept_fp = 0, 0
    for k, v in sd.items():
        if v.dim() == 2 and v.numel() >= args.min_quantize_numel:
            q, scale = quantize_tensor(v)
            out_sd[k] = q
            out_sd[k + "__scale"] = scale
            n_quantized += 1
        else:
            out_sd[k] = v.to(torch.float16) if v.is_floating_point() else v
            n_kept_fp += 1
    print(f"quantized {n_quantized} tensors, kept {n_kept_fp} in fp16 (biases/norms/small tensors)")

    save_file(out_sd, str(out_dir / "model_int8.safetensors"))
    shutil.copy(model_dir / "config.json", out_dir / "config.json")
    with open(out_dir / "serialize_variant.json", "w", encoding="utf-8") as f:
        json.dump({"variant": args.variant}, f)

    fp_mb = dir_size_mb(model_dir)
    int8_mb = dir_size_mb(out_dir)
    print(f"size fp   (dir): {fp_mb:8.1f} MB  ({model_dir})")
    print(f"size int8 (dir): {int8_mb:8.1f} MB  ({out_dir})")
    print(f"note: tokenizer NOT copied -- this member is meant to share ens4/ens5's model/tokenizer/. "
          f"If it needs its OWN tokenizer/vocab, copy tokenizer files into {out_dir} separately.")

    if args.verify_samples <= 0:
        return

    # ---- round-trip check: dequantized model vs original fp model ----
    from safetensors.torch import load_file

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

    dq_model = AutoModelForSequenceClassification.from_config(model.config).half().to(device)
    sd_q = load_file(str(out_dir / "model_int8.safetensors"))
    dq_sd = {}
    for k, v in sd_q.items():
        if k.endswith("__scale"):
            continue
        if v.dtype == torch.int8:
            scale = sd_q[k + "__scale"].float()
            dq_sd[k] = (v.float() * scale.unsqueeze(-1)).to(torch.float16)
        else:
            dq_sd[k] = v.to(torch.float16) if v.is_floating_point() else v
    missing, unexpected = dq_model.load_state_dict(dq_sd, strict=False)
    bad = [k for k in missing if "inv_freq" not in k and "position_ids" not in k]
    assert not bad, f"missing weights after dequant load: {bad[:5]}"
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    dq_model.eval()

    fp_model = model.to(device).half().eval()

    samples = load_jsonl(Path(args.data_dir) / "train.jsonl")[: args.verify_samples]
    texts = [render_granite_sample(s, max_history_events=args.max_history_events,
                                    include_open_file_names=args.include_open_file_names)
             for s in samples]

    fp_logits, dq_logits = [], []
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, truncation=True, max_length=args.max_length, return_tensors="pt").to(device)
            fp_logits.append(fp_model(**enc).logits.float().cpu().numpy()[0])
            dq_logits.append(dq_model(**enc).logits.float().cpu().numpy()[0])
    fp_logits = np.stack(fp_logits)
    dq_logits = np.stack(dq_logits)

    agree = (fp_logits.argmax(1) == dq_logits.argmax(1)).mean()
    drift = np.abs(fp_logits - dq_logits)
    print("\n================ round-trip verification ================")
    print(f"samples               : {len(texts)}")
    print(f"argmax agreement      : {agree:.4f}")
    print(f"logit drift max/mean  : {drift.max():.4f} / {drift.mean():.4f}")
    print("============================================================")


if __name__ == "__main__":
    main()
