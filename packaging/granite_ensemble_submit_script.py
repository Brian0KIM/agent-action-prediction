import csv
import json
import os
from pathlib import Path

import numpy as np


def _safe_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _compact_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _budget_bucket(tokens):
    try:
        tokens = int(tokens)
    except (TypeError, ValueError):
        return "unknown"
    if tokens < 2_000:
        return "very_low"
    if tokens < 10_000:
        return "low"
    if tokens < 50_000:
        return "medium"
    return "high"


def _elapsed_bucket(seconds):
    try:
        seconds = int(seconds)
    except (TypeError, ValueError):
        return "unknown"
    if seconds < 120:
        return "early"
    if seconds < 900:
        return "mid"
    return "late"


def render_sample(sample, max_history_events=16):
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

    return " ".join(["[META]", meta_text, "[HIST]", " | ".join(hist_parts), "[CUR]", _safe_text(sample.get("current_prompt"))])


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_sample_submission(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def save_submission(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_logit_bias(model_dir, id2label):
    path = Path(model_dir) / "logit_bias.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    bias_map = payload.get("bias", {})
    return np.array([float(bias_map.get(id2label[idx], 0.0)) for idx in range(len(id2label))], dtype=np.float32)


def predict_logits(model_dir, texts, max_length, batch_size):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

    class TextDataset:
        def __init__(self, items, tokenizer):
            self.items = items
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            return self.tokenizer(self.items[idx], truncation=True, max_length=max_length, padding=False)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    bias = load_logit_bias(model_dir, model.config.id2label)

    loader = DataLoader(
        TextDataset(texts, tokenizer),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    logits_chunks = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(**batch).logits.float().cpu().numpy()
            if bias is not None:
                logits = logits + bias[None, :]
            logits_chunks.append(logits)

    return np.concatenate(logits_chunks, axis=0), model.config.id2label


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    data_dir = Path("./data")
    model_dirs = [
        Path("./model/granite-311m-fold0-h16-l512"),
        Path("./model/granite-311m-fold1-h16-l512"),
    ]
    output_path = Path("./output/submission.csv")
    max_length = 512
    max_history_events = 16
    batch_size = 64

    samples = load_jsonl(data_dir / "test.jsonl")
    ids = [sample["id"] for sample in samples]
    texts = [render_sample(sample, max_history_events=max_history_events) for sample in samples]

    ensemble_logits = None
    id2label = None
    for model_dir in model_dirs:
        logits, model_id2label = predict_logits(model_dir, texts, max_length, batch_size)
        id2label = model_id2label
        ensemble_logits = logits if ensemble_logits is None else ensemble_logits + logits

    pred_ids = np.argmax(ensemble_logits / len(model_dirs), axis=1).tolist()
    preds = [id2label[int(i)] for i in pred_ids]
    pred_map = dict(zip(ids, preds))

    fieldnames, rows = load_sample_submission(data_dir / "sample_submission.csv")
    for row in rows:
        row["action"] = pred_map[row["id"]]
    save_submission(output_path, fieldnames, rows)
    print(f"Saved {output_path} rows={len(rows)} models={len(model_dirs)}")


if __name__ == "__main__":
    main()
