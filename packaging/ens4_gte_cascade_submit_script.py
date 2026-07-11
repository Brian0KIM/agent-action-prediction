"""DACON action-decision inference — ens4 (3 granite + 1 qwen3, int8, gated)
PLUS a GTE search-specialist cascade on top.

This is `submit_ens4_706/script.py`'s exact ensemble logic (member loading,
serialization, int8 dequant, the provably-safe qwen margin gate) UNCHANGED,
with one addition at the very end: wherever the 4-member mean lands on one of
the 4 exploration classes (read_file/grep_search/list_directory/glob_pattern
-- ALL_CLASSES indices 0-3), a GTE specialist re-scores just those 4 classes
and gets blended in at --blend-weight (default 0.5, the best weight found in
eval_search_specialist.py's sweep on the granite-only base).

IMPORTANT -- this has NOT been locally validated end-to-end: the ens4 members
were trained on 100% of train.jsonl (no held-out fold), so there is no
train.jsonl subset that isn't leaked for them. Any macro-F1 computed against
train.jsonl using this pipeline is meaningless. The only real signal is an
actual DACON submission's public LB score.

Zip layout expected (mirrors submit_ens4_706, plus one addition):
  model/tokenizer/, model/remap.npy, model/member_*/         <- from ens4, unchanged
  model/specialist_gte/                                       <- GTE search-specialist
    checkpoint files (config.json, model.safetensors, tokenizer files),
    trust_remote_code-loadable, id2label covering the 4 SEARCH_CLASSES.

Env vars (same as ens4): ENS_BS (granite batch size, default 256),
ENS_QWEN_BS (qwen batch size, default 64). New: ENS_SPEC_BS (specialist batch
size, default 64), ENS_BLEND_W (blend weight, default 0.5).
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
SEARCH_CLASSES = ["read_file", "grep_search", "list_directory", "glob_pattern"]
SEARCH_IDX = [ALL_CLASSES.index(c) for c in SEARCH_CLASSES]  # == [0, 1, 2, 3]
MAX_LENGTH = 512
BATCH_SIZE = int(os.environ.get("ENS_BS", "256"))
QWEN_BATCH_SIZE = int(os.environ.get("ENS_QWEN_BS", "64"))
SPEC_BATCH_SIZE = int(os.environ.get("ENS_SPEC_BS", "64"))
BLEND_W = float(os.environ.get("ENS_BLEND_W", "0.5"))


# ------------------------------------------------------------ ens4 (unchanged)
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


def run_encoder_members(member_dirs, variants, samples, model_root, device, mean_probs):
    import numpy as np
    import torch
    from transformers import AutoTokenizer, DataCollatorWithPadding

    tokenizer = AutoTokenizer.from_pretrained(model_root / "tokenizer", local_files_only=True)
    tokenizer.truncation_side = "right"
    remap = torch.from_numpy(np.load(model_root / "remap.npy")).long().to(device)
    collator = DataCollatorWithPadding(tokenizer)

    enc_by_variant, order_by_variant = {}, {}
    for v in {variants[m] for m in member_dirs}:
        texts = [serialize_rich(s, VARIANT_STRIP[v]) for s in samples]
        e = tokenizer(texts, truncation=True, max_length=MAX_LENGTH)
        e = [{"input_ids": e["input_ids"][i], "attention_mask": e["attention_mask"][i]}
             for i in range(len(texts))]
        enc_by_variant[v] = e
        order_by_variant[v] = sorted(range(len(e)), key=lambda i: len(e[i]["input_ids"]))

    for mdir in member_dirs:
        v = variants[mdir]
        enc, order = enc_by_variant[v], order_by_variant[v]
        model = load_member(mdir, device)
        with torch.no_grad():
            for start in range(0, len(order), BATCH_SIZE):
                idx = order[start:start + BATCH_SIZE]
                batch = {k: t.to(device) for k, t in collator([enc[i] for i in idx]).items()}
                batch["input_ids"] = remap[batch["input_ids"]]
                logits = model(**batch).logits.float()
                mean_probs[idx] += torch.softmax(logits, -1).cpu().numpy()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"member done: {mdir.name} ({v})")


def run_qwen_member(mdir, variant, samples, device, mean_probs, run_idx):
    import numpy as np
    import torch
    from transformers import AutoTokenizer, DataCollatorWithPadding

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


# ------------------------------------------------------ GTE specialist cascade
def _safe_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _compact_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def render_granite_sample(sample, max_history_events=12):
    """Byte-for-byte copy of action_router.features.render_granite_sample with
    include_open_file_names=False -- the format train_search_specialist.py
    trains specialists on. Do NOT change independently of that script."""
    meta = sample.get("session_meta") or {}
    workspace = meta.get("workspace") or {}
    history = sample.get("history") or []
    recent_history = history[-max_history_events:]

    open_files = workspace.get("open_files") or []
    language_mix = workspace.get("language_mix") or {}
    main_lang = ""
    if language_mix:
        main_lang = max(language_mix.items(), key=lambda item: item[1])[0]

    meta_text = " ".join(
        [
            f"tier={_safe_text(meta.get('user_tier'))}",
            f"pref={_safe_text(meta.get('language_pref'))}",
            f"turn={_safe_text(meta.get('turn_index'))}",
            f"budget={_budget_bucket(meta.get('budget_tokens_remaining'))}",
            f"elapsed={_elapsed_bucket(meta.get('elapsed_session_sec'))}",
            f"lang={main_lang}",
            f"ci={_safe_text(workspace.get('last_ci_status'))}",
            f"git={'dirty' if workspace.get('git_dirty') else 'clean'}",
            f"open={len(open_files)}",
            f"loc={_safe_text(workspace.get('loc'))}",
        ]
    )

    hist_parts = []
    for item in recent_history:
        role = item.get("role", "")
        if role == "user":
            hist_parts.append(f"U: {_safe_text(item.get('content'))}")
        elif role == "assistant_action":
            name = _safe_text(item.get("name"))
            args = _compact_json(item.get("args") or {})
            result = _safe_text(item.get("result_summary"))
            hist_parts.append(f"A[{name}] {args} -> {result}")

    return " ".join(
        [
            "[META]", meta_text, "[HIST]", " | ".join(hist_parts),
            "[CUR]", _safe_text(sample.get("current_prompt")),
        ]
    )


def run_specialist_cascade(spec_dir, samples, device, mean_probs, blend_w):
    """Gate rows whose 4-member mean argmax is a SEARCH_CLASSES member to the
    GTE specialist; blend probabilities within that 4-class subset only."""
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

    top1 = mean_probs.argmax(axis=1)
    gated_idx = np.nonzero(np.isin(top1, SEARCH_IDX))[0].tolist()
    print(f"[specialist] gated {len(gated_idx)}/{len(samples)} "
          f"({100 * len(gated_idx) / len(samples):.1f}%) rows to GTE")
    if not gated_idx:
        return mean_probs

    tokenizer = AutoTokenizer.from_pretrained(spec_dir, local_files_only=True, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        spec_dir, local_files_only=True, torch_dtype=torch.float16, trust_remote_code=True,
    ).to(device).eval()
    id2label = model.config.id2label
    label2id = {id2label[i]: i for i in range(len(id2label))}
    col_order = [label2id[c] for c in SEARCH_CLASSES]  # specialist's own label order -> SEARCH_CLASSES order

    texts = [render_granite_sample(samples[i], max_history_events=12) for i in gated_idx]
    enc = [tokenizer(t, truncation=True, max_length=MAX_LENGTH, padding=False) for t in texts]
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]["input_ids"]))
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    spec_probs = np.zeros((len(gated_idx), len(SEARCH_CLASSES)), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(order), SPEC_BATCH_SIZE):
            loc = order[start:start + SPEC_BATCH_SIZE]
            batch = {k: v.to(device) for k, v in collator([enc[i] for i in loc]).items()}
            logits = model(**batch).logits.float()[:, col_order]
            p = torch.softmax(logits, -1).cpu().numpy()
            for j, i in enumerate(loc):
                spec_probs[i] = p[j]
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    base_sub = mean_probs[np.ix_(gated_idx, SEARCH_IDX)]
    base_sub = base_sub / base_sub.sum(axis=1, keepdims=True).clip(min=1e-9)
    blended = (1 - blend_w) * base_sub + blend_w * spec_probs
    choice_local = blended.argmax(axis=1)  # 0..3 within SEARCH_CLASSES

    # overwrite mean_probs for gated rows: one-hot the chosen search class so
    # the final global argmax (over all 14 classes) picks it deterministically
    for row, local_c in zip(gated_idx, choice_local):
        mean_probs[row, :] = 0.0
        mean_probs[row, SEARCH_IDX[local_c]] = 1.0
    print(f"[specialist] done, blend_w={blend_w}")
    return mean_probs


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
    if granite_members:
        run_encoder_members(granite_members, variants, samples, model_root, device, mean_probs)
    print(f"[t] granite done at {time.time()-t0:.0f}s")

    if qwen_members:
        srt = np.sort(mean_probs, axis=1)
        flippable = (srt[:, -1] - srt[:, -2]) < 1.0
        run_idx = np.nonzero(flippable)[0].tolist()
        print(f"[gate] qwen runs on {len(run_idx)}/{len(samples)} "
              f"({100*len(run_idx)/len(samples):.1f}%) — rest provably unaffected")
        for mdir in qwen_members:
            run_qwen_member(mdir, variants[mdir], samples, device, mean_probs, run_idx)
    print(f"[t] all members done at {time.time()-t0:.0f}s")

    mean_probs /= len(member_dirs)

    spec_dir = model_root / "specialist_gte"
    if spec_dir.exists():
        mean_probs = run_specialist_cascade(spec_dir, samples, device, mean_probs, BLEND_W)
    else:
        print(f"[specialist] {spec_dir} not found, skipping cascade")
    print(f"[t] all done at {time.time()-t0:.0f}s")

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
    print(f"Saved {output_path} rows={len(rows)} | members={[d.name for d in member_dirs]} "
          f"| specialist={'gte' if spec_dir.exists() else 'none'}")


if __name__ == "__main__":
    main()
