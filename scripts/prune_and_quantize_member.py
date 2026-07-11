"""Vocab-prune + int8-quantize a granite-family checkpoint into a new
ens4/ens5 member with its OWN remap.npy (does not touch the other members'
shared tokenizer/remap.npy).

Two things this member folder ends up with, matching ens4/ens5's
`load_member()` + `resolve_remap()` contract:
  - model_int8.safetensors: same per-row-scale int8 format as
    quantize_int8_member.py, but the token-embedding weight is first sliced
    down to only the KEPT rows (pruned vocab), then quantized.
  - remap.npy: int64 array of length == original vocab size. remap[i] = the
    pruned row index for original token id i if i is kept, else the fallback
    row's index. ens5_submit_script.py applies this to input_ids before the
    forward pass, so the pruned embedding lines up.

"Kept" = every token id that appears when train.jsonl + test.jsonl are
rendered with this member's own variant, plus tokenizer special tokens,
plus (optionally, --margin > 0) the first --margin ids by raw vocab index --
low ids are OFTEN, but not guaranteed, the more frequent subword pieces for a
BPE tokenizer; this is a heuristic safety margin against a hidden-test token
never seen in train/test, not a verified frequency ranking. Set --margin 0
to rely purely on train+test coverage.

Verifies round-trip like quantize_int8_member.py (argmax agreement + logit
drift vs the original fp model), on a sample of train.jsonl.

Example
-------
    python scripts/prune_and_quantize_member.py \
        --model-dir ./model/granite-311m-all-names-lr5e5 \
        --output-dir ./model/member_4_all_lr5e5_pruned_int8 \
        --variant richargs --data-dir ../data \
        --include-open-file-names --margin 4096 --verify-samples 500
"""
import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from action_router.features import render_granite_sample  # noqa: E402

VARIANT_STRIP = {"richargs": True, "richmeta": False}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dir_size_mb(path):
    total = sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())
    return total / 1e6


def render_for_pruning(sample, variant, max_history_events, include_open_file_names):
    # This member's own render is render_granite_sample (richargs vs richmeta
    # only affects the standalone qwen3/ens4 serialize_rich(), which this
    # granite member does NOT use -- it uses render_granite_sample same as
    # every other script in this repo). Kept as a parameter for clarity/
    # forward-compat if this member ever switches renderers.
    return render_granite_sample(sample, max_history_events=max_history_events,
                                  include_open_file_names=include_open_file_names)


def collect_used_token_ids(tokenizer, data_dir, variant, max_history_events, include_open_file_names, max_length):
    used = Counter()
    for fname in ["train.jsonl", "test.jsonl"]:
        path = Path(data_dir) / fname
        if not path.exists():
            print(f"note: {path} not found, skipping")
            continue
        samples = load_jsonl(path)
        texts = [render_for_pruning(s, variant, max_history_events, include_open_file_names) for s in samples]
        for start in range(0, len(texts), 1024):
            enc = tokenizer(texts[start:start + 1024], truncation=True, max_length=max_length, padding=False)
            for ids in enc["input_ids"]:
                used.update(ids)
        print(f"{fname}: {len(samples)} samples tokenized, running unique token count = {len(used)}")
    return used


def build_remap(vocab_size, used_counter, special_ids, margin, fallback_id):
    kept = set(used_counter) | set(special_ids)
    if margin > 0:
        kept |= set(range(min(margin, vocab_size)))
    kept = sorted(kept)
    if fallback_id not in kept:
        kept = sorted(set(kept) | {fallback_id})
    old_to_new = {old: new for new, old in enumerate(kept)}
    fallback_new = old_to_new[fallback_id]
    remap = np.full(vocab_size, fallback_new, dtype=np.int64)
    for old, new in old_to_new.items():
        remap[old] = new
    return remap, kept


def quantize_tensor(t):
    import torch
    scale = t.abs().amax(dim=1).clamp(min=1e-8) / 127.0
    q = torch.round(t / scale.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    return q, scale.to(torch.float16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variant", required=True, choices=["richargs", "richmeta"])
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--include-open-file-names", action="store_true")
    parser.add_argument("--max-history-events", type=int, default=12)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--margin", type=int, default=4096,
                         help="Extra low-vocab-index ids to keep as a hidden-test safety margin. 0 to disable.")
    parser.add_argument("--min-quantize-numel", type=int, default=100_000,
                         help="Lowered from quantize_int8_member.py's 1e6 default -- catches the ~885k-element "
                              "Wo layers that default missed.")
    parser.add_argument("--embedding-key", default="model.embeddings.tok_embeddings.weight",
                         help="State dict key of the token embedding to prune. Check with the "
                              "safetensors-header inspection snippet if this model's key differs.")
    parser.add_argument("--verify-samples", type=int, default=500)
    args = parser.parse_args()

    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = Path(args.model_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True, torch_dtype=torch.float32)
    model.eval()
    sd = model.state_dict()

    if args.embedding_key not in sd:
        raise SystemExit(f"'{args.embedding_key}' not in state dict. Keys look like:\n  " +
                          "\n  ".join(k for k in sd if "embed" in k.lower()))
    emb = sd[args.embedding_key]
    vocab_size = emb.shape[0]
    print(f"embedding {args.embedding_key}: shape={tuple(emb.shape)}")

    special_ids = [i for i in tokenizer.all_special_ids if i is not None]
    fallback_id = tokenizer.unk_token_id
    if fallback_id is None:
        fallback_id = special_ids[0] if special_ids else 0
        print(f"note: tokenizer has no unk_token_id, using {fallback_id} as fallback")

    used = collect_used_token_ids(tokenizer, args.data_dir, args.variant, args.max_history_events,
                                   args.include_open_file_names, args.max_length)
    remap, kept = build_remap(vocab_size, used, special_ids, args.margin, fallback_id)
    print(f"vocab {vocab_size} -> kept {len(kept)} rows "
          f"({100 * len(kept) / vocab_size:.1f}%), fallback_id={fallback_id}")
    np.save(out_dir / "remap.npy", remap)

    kept_idx = torch.tensor(kept, dtype=torch.long)
    out_sd = {}
    n_quantized, n_kept_fp = 0, 0
    for k, v in sd.items():
        if k == args.embedding_key:
            v = v.index_select(0, kept_idx)  # prune rows BEFORE quantizing
        if v.dim() == 2 and v.numel() >= args.min_quantize_numel:
            q, scale = quantize_tensor(v)
            out_sd[k] = q
            out_sd[k + "__scale"] = scale
            n_quantized += 1
        else:
            out_sd[k] = v.to(torch.float16) if v.is_floating_point() else v
            n_kept_fp += 1
    print(f"quantized {n_quantized} tensors, kept {n_kept_fp} in fp16")

    save_file(out_sd, str(out_dir / "model_int8.safetensors"))
    shutil.copy(model_dir / "config.json", out_dir / "config.json")
    with open(out_dir / "serialize_variant.json", "w", encoding="utf-8") as f:
        json.dump({"variant": args.variant}, f)

    fp_mb = dir_size_mb(model_dir)
    out_mb = dir_size_mb(out_dir)
    print(f"size fp          (dir): {fp_mb:8.1f} MB  ({model_dir})")
    print(f"size pruned+int8 (dir): {out_mb:8.1f} MB  ({out_dir})")

    if args.verify_samples <= 0:
        return

    # ---- round-trip check against a NEW model instance loading the pruned+quantized weights ----
    from safetensors.torch import load_file

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # pruned config: same architecture, but embedding row count changes, so
    # build the dequantized module directly rather than from_config (which
    # would allocate the ORIGINAL vocab_size embedding).
    dq_sd = {}
    sd_q = load_file(str(out_dir / "model_int8.safetensors"))
    for k, v in sd_q.items():
        if k.endswith("__scale"):
            continue
        if v.dtype == torch.int8:
            scale = sd_q[k + "__scale"].float()
            dq_sd[k] = (v.float() * scale.unsqueeze(-1)).to(torch.float16)
        else:
            dq_sd[k] = v.to(torch.float16) if v.is_floating_point() else v

    dq_model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True, torch_dtype=torch.float16)
    # resize embedding to the pruned row count, then load
    dq_model.resize_token_embeddings(len(kept))
    missing, unexpected = dq_model.load_state_dict(dq_sd, strict=False)
    bad = [k for k in missing if "inv_freq" not in k and "position_ids" not in k]
    assert not bad, f"missing weights after dequant load: {bad[:5]}"
    dq_model.to(device).eval()

    fp_model = model.to(device).half().eval()
    remap_t = torch.from_numpy(remap).long().to(device)

    samples = load_jsonl(Path(args.data_dir) / "train.jsonl")[: args.verify_samples]
    texts = [render_for_pruning(s, args.variant, args.max_history_events, args.include_open_file_names) for s in samples]

    fp_logits, dq_logits = [], []
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, truncation=True, max_length=args.max_length, return_tensors="pt").to(device)
            fp_logits.append(fp_model(**enc).logits.float().cpu().numpy()[0])
            enc_pruned = {**enc, "input_ids": remap_t[enc["input_ids"]]}
            dq_logits.append(dq_model(**enc_pruned).logits.float().cpu().numpy()[0])
    fp_logits = np.stack(fp_logits)
    dq_logits = np.stack(dq_logits)

    agree = (fp_logits.argmax(1) == dq_logits.argmax(1)).mean()
    drift = np.abs(fp_logits - dq_logits)
    print("\n================ round-trip verification (pruned+int8) ================")
    print(f"samples               : {len(texts)}")
    print(f"argmax agreement      : {agree:.4f}")
    print(f"logit drift max/mean  : {drift.max():.4f} / {drift.mean():.4f}")
    print("==========================================================================")


if __name__ == "__main__":
    main()
