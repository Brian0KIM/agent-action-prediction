"""DACON action-decision inference — ens5: submit_ens4_706's pipeline extended
to support an arbitrary Nth granite-family member that does NOT share the
existing vocab-pruning/remap.npy.

Difference from submit_ens4_706/script.py: `run_encoder_members` now checks
for a PER-MEMBER `<mdir>/remap.npy` first, falling back to the shared
`model/remap.npy` only if the member doesn't have its own, and skipping
remap entirely (raw token ids) if NEITHER exists. This matters because a
remap is only valid if the member's own embedding matrix was pruned to
exactly that row subset -- applying the shared 3-granite remap to a member
with a full, unpruned embedding table would silently corrupt its inputs.

New members ship with just: model_int8.safetensors, config.json,
serialize_variant.json (from quantize_int8_member.py) -- and, if they use
their own vocab pruning, their own remap.npy.

Everything else (int8 dequant, qwen provably-safe margin gate, uniform mean,
raw argmax) is byte-for-byte submit_ens4_706/script.py.
"""
import csv
import json
import os
import time
from pathlib import Path

ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]
MAX_LENGTH = 512
BATCH_SIZE = int(os.environ.get("ENS_BS", "256"))
QWEN_BATCH_SIZE = int(os.environ.get("ENS_QWEN_BS", "64"))


def _budget_bucket(tokens):
    if tokens < 2_000:
        return "very_low"
    if tokens < 10_000:
        return "low"
    if tokens < 50_000:
        return "medium"
    return "high"


def _elapsed_bucket(seconds):
    if seconds < 120:
        return "early"
    if seconds < 900:
        return "mid"
    return "late"


def _strip_dirs(v):
    if "/" not in v:
        return v
    tail = "/" if v.endswith("/") else ""
    return v.rstrip("/").rsplit("/", 1)[-1] + tail


def serialize_rich(r, strip_args):
    sm = r["session_meta"]
    ws = sm["workspace"]
    parts = []
    mix = ws.get("language_mix") or {}
    codelang = max(mix.items(), key=lambda kv: kv[1])[0] if mix else "-"
    names = [f.rsplit("/", 1)[-1] for f in ws["open_files"][:6]]
    parts.append(
        f"[tier={sm['user_tier']} lang={sm['language_pref']} turn={sm['turn_index']} "
        f"budget={_budget_bucket(sm['budget_tokens_remaining'])} "
        f"elapsed={_elapsed_bucket(sm['elapsed_session_sec'])} "
        f"codelang={codelang} loc={ws.get('loc', '-')} ci={ws['last_ci_status']} "
        f"dirty={ws['git_dirty']} open={len(ws['open_files'])} "
        f"files={','.join(names) or '-'}]"
    )
    for t in r["history"]:
        if t.get("role") == "user":
            parts.append(f"USER: {t['content']}")
        else:
            args = t.get("args", {}) or {}
            if strip_args:
                args = {k: _strip_dirs(v) if isinstance(v, str) else v
                        for k, v in args.items()}
            parts.append(f"ACTION {t['name']}({args}) -> {t.get('result_summary', '')}")
    parts.append(f"PROMPT: {r['current_prompt']}")
    return "\n".join(parts)


VARIANT_STRIP = {"richargs": True, "richmeta": False}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_member(mdir, device):
    """Load fp16 model; if model_int8.safetensors exists, dequantize int8->fp16."""
    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForSequenceClassification

    qpath = Path(mdir) / "model_int8.safetensors"
    if not qpath.exists():
        return AutoModelForSequenceClassification.from_pretrained(
            mdir, local_files_only=True, torch_dtype=torch.float16
        ).to(device).eval()

    cfg = AutoConfig.from_pretrained(mdir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_config(cfg).half().to(device)
    sd_q = load_file(str(qpath), device=str(device))
    sd = {}
    for k, v in sd_q.items():
        if k.endswith("__scale"):
            continue
        if v.dtype == torch.int8:
            scale = sd_q[k + "__scale"].float()
            sd[k] = (v.float() * scale.unsqueeze(-1)).to(torch.float16)
        else:
            sd[k] = v.to(torch.float16) if v.is_floating_point() else v
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad = [k for k in missing if "inv_freq" not in k and "position_ids" not in k]
    assert not bad, f"missing weights after dequant load: {bad[:5]}"
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    return model.eval()


def resolve_remap(mdir, model_root, device):
    """Per-member remap if present, else the shared one IF this member's dir
    doesn't opt out, else None (no remap -- raw token ids)."""
    import numpy as np
    import torch

    own = Path(mdir) / "remap.npy"
    if own.exists():
        print(f"  {Path(mdir).name}: using own remap.npy")
        return torch.from_numpy(np.load(own)).long().to(device)
    shared = Path(model_root) / "remap.npy"
    opt_out = Path(mdir) / "no_remap"
    if shared.exists() and not opt_out.exists():
        print(f"  {Path(mdir).name}: using shared model/remap.npy")
        return torch.from_numpy(np.load(shared)).long().to(device)
    print(f"  {Path(mdir).name}: no remap (raw token ids, full/unpruned embedding)")
    return None


def run_encoder_members(member_dirs, variants, samples, model_root, device, mean_probs, row_indices=None):
    """Granite-family members: shared tokenizer, CLS-pooled encoder. Remap is
    resolved per-member (see resolve_remap) instead of assumed shared.

    row_indices: if given, only these sample indices are inferred (and only
    mean_probs[row_indices] is updated) -- used for cascaded margin-gating of
    non-core members. None = run on everything (core members)."""
    import numpy as np
    import torch
    from transformers import AutoTokenizer, DataCollatorWithPadding

    rows = list(range(len(samples))) if row_indices is None else row_indices
    if not rows:
        print(f"member(s) skipped entirely (0 rows remain undecided): {[m.name for m in member_dirs]}")
        return
    row_samples = [samples[i] for i in rows]

    tokenizer = AutoTokenizer.from_pretrained(model_root / "tokenizer", local_files_only=True)
    tokenizer.truncation_side = "right"
    collator = DataCollatorWithPadding(tokenizer)

    enc_by_variant, order_by_variant = {}, {}
    for v in {variants[m] for m in member_dirs}:
        texts = [serialize_rich(s, VARIANT_STRIP[v]) for s in row_samples]
        e = tokenizer(texts, truncation=True, max_length=MAX_LENGTH)
        e = [{"input_ids": e["input_ids"][i], "attention_mask": e["attention_mask"][i]}
             for i in range(len(texts))]
        enc_by_variant[v] = e
        order_by_variant[v] = sorted(range(len(e)), key=lambda i: len(e[i]["input_ids"]))

    for mdir in member_dirs:
        v = variants[mdir]
        enc, order = enc_by_variant[v], order_by_variant[v]
        remap = resolve_remap(mdir, model_root, device)
        model = load_member(mdir, device)
        with torch.no_grad():
            for start in range(0, len(order), BATCH_SIZE):
                loc = order[start:start + BATCH_SIZE]
                batch = {k: t.to(device) for k, t in collator([enc[i] for i in loc]).items()}
                if remap is not None:
                    batch["input_ids"] = remap[batch["input_ids"]]
                logits = model(**batch).logits.float()
                probs = torch.softmax(logits, -1).cpu().numpy()
                global_idx = [rows[i] for i in loc]
                mean_probs[global_idx] += probs
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"member done: {mdir.name} ({v}) on {len(rows)}/{len(samples)} rows")


def run_qwen_member(mdir, variant, samples, device, mean_probs, run_idx):
    import numpy as np
    import torch
    from transformers import AutoTokenizer, DataCollatorWithPadding

    if not run_idx:
        print(f"member skipped entirely (0 rows remain undecided): {Path(mdir).name}")
        return
    tokenizer = AutoTokenizer.from_pretrained(mdir, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "right"
    remap = torch.from_numpy(np.load(Path(mdir) / "remap.npy")).long().to(device)
    collator = DataCollatorWithPadding(tokenizer)

    texts = [serialize_rich(samples[i], VARIANT_STRIP[variant]) for i in run_idx]
    enc = []
    for start in range(0, len(texts), 1024):
        e = tokenizer(texts[start:start + 1024], truncation=True,
                      max_length=MAX_LENGTH, padding=False)
        enc.extend({"input_ids": e["input_ids"][i], "attention_mask": e["attention_mask"][i]}
                   for i in range(len(e["input_ids"])))
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]["input_ids"]))

    model = load_member(mdir, device)
    with torch.no_grad():
        for start in range(0, len(order), QWEN_BATCH_SIZE):
            loc = order[start:start + QWEN_BATCH_SIZE]
            batch = {k: t.to(device) for k, t in collator([enc[i] for i in loc]).items()}
            batch["input_ids"] = remap[batch["input_ids"]]
            logits = model(**batch).logits.float()
            p = torch.softmax(logits, -1).cpu().numpy()
            mean_probs[[run_idx[i] for i in loc]] += p
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"member done: {Path(mdir).name} ({variant}, qwen) "
          f"on {len(run_idx)}/{len(samples)} gated samples")


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import numpy as np
    import torch
    from transformers import AutoConfig

    data_dir = Path("./data")
    model_root = Path("./model")
    output_path = Path("./output/submission.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    member_dirs = sorted(d for d in model_root.glob("member_*") if d.is_dir())
    assert member_dirs, "no model/member_* dirs in the zip"
    variants, families = {}, {}
    for mdir in member_dirs:
        v = json.load(open(mdir / "serialize_variant.json"))["variant"]
        assert v in VARIANT_STRIP, f"unknown variant {v}"
        variants[mdir] = v
        families[mdir] = AutoConfig.from_pretrained(mdir, local_files_only=True).model_type

    samples = load_jsonl(data_dir / "test.jsonl")
    ids = [s["id"] for s in samples]
    mean_probs = np.zeros((len(samples), len(ALL_CLASSES)), dtype=np.float64)

    t0 = time.time()
    granite_members = [m for m in member_dirs if families[m] != "qwen3"]
    qwen_members = [m for m in member_dirs if families[m] == "qwen3"]
    # "core" granite members (no `gated` marker file) always run on every row,
    # exactly like submit_ens4_706 -- this keeps the proven 3-granite base
    # untouched. Any OTHER granite member carrying a `gated` marker file joins
    # qwen in the cascaded margin-gate below instead of running unconditionally.
    core_granite = [m for m in granite_members if not (m / "gated").exists()]
    gated_granite = [m for m in granite_members if (m / "gated").exists()]

    if core_granite:
        run_encoder_members(core_granite, variants, samples, model_root, device, mean_probs)
    print(f"[t] core granite done at {time.time()-t0:.0f}s")

    # Cascaded margin gate: with `remaining_count` probability-vector members
    # left to add (each entry in [0,1], row-sums to 1), a row's argmax is
    # PROVABLY unaffected by any of them once the current margin (top1-top2 of
    # the sum-so-far) is >= remaining_count -- worst case each remaining
    # member shifts the gap by at most 1 (moving all its mass from the
    # current top class to the runner-up). This generalizes submit_ens4_706's
    # single-qwen gate (remaining_count == 1, threshold 1.0) to any number of
    # remaining members, processed one at a time so the pool of "still
    # undecided" rows only shrinks.
    remaining_pool = gated_granite + qwen_members
    undecided = list(range(len(samples)))
    for i, mdir in enumerate(remaining_pool):
        remaining_count = len(remaining_pool) - i
        if undecided:
            srt = np.sort(mean_probs[undecided], axis=1)
            margin = srt[:, -1] - srt[:, -2]
            undecided = [undecided[j] for j in range(len(undecided)) if margin[j] < remaining_count]
        print(f"[gate] before {mdir.name}: {len(undecided)}/{len(samples)} rows undecided "
              f"(margin < {remaining_count}, {remaining_count} member(s) incl. this one remaining)")
        if families[mdir] == "qwen3":
            run_qwen_member(mdir, variants[mdir], samples, device, mean_probs, undecided)
        else:
            run_encoder_members([mdir], variants, samples, model_root, device, mean_probs, row_indices=undecided)
    print(f"[t] all members done at {time.time()-t0:.0f}s")

    mean_probs /= len(member_dirs)

    preds = [ALL_CLASSES[int(i)] for i in mean_probs.argmax(1)]
    pred_map = dict(zip(ids, preds))
    with open(data_dir / "sample_submission.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames, rows = reader.fieldnames, list(reader)
    for row in rows:
        row["action"] = pred_map[row["id"]]
    os.makedirs(output_path.parent, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {output_path} rows={len(rows)} | members={[d.name for d in member_dirs]}")


if __name__ == "__main__":
    main()
