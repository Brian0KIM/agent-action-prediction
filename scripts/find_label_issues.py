"""Confident Learning pass over train labels using OOF predictions.

Reference: Northcutt, Jiang, Chuang. "Confident Learning: Estimating
Uncertainty in Dataset Labels." JAIR 2021. https://arxiv.org/abs/1911.00068
Implementation: https://github.com/cleanlab/cleanlab

Takes one or more `.npz` OOF prob dumps with the shared schema (ids, y_true,
classes, logits, probs) -- the same schema `train_granite_router.py --oof-path`
writes and `blend_eval.py` / `eval_search_specialist.py` consume -- and reports
which train.jsonl rows cleanlab flags as likely label errors, i.e. rows where
the out-of-fold prediction disagrees with the given label with high, consistent
confidence.

This does NOT tell you the row is wrong -- only that it's a candidate. Two
things produce this symptom and they need different fixes:

  - genuine annotation/simulation error -> drop or relabel the row
  - a row that is legitimately hard/ambiguous but the given label is correct
    -> keep it; this is a target for the search-specialist cascade instead

Read the flagged `current_prompt` / history yourself before acting on either.

Multiple npz inputs are merged by id: same id in >1 file -> probs averaged
(ensembling models on the same fold, like eval_search_specialist.py); disjoint
ids across files -> concatenated (pooling OOF across folds for full coverage).
Mixing both patterns in one run is fine.

Example
-------
    pip install cleanlab
    python scripts/find_label_issues.py \
        --oof-npz output/oof/granite-fold0.npz,output/oof/granite-fold1.npz \
        --data-dir ../data \
        --output output/label_issues.csv \
        --top-k 50

    # restrict the printed/CSV report to rows whose GIVEN label is in the
    # exploration cluster (the specific confusion this was written for)
    python scripts/find_label_issues.py \
        --oof-npz output/oof/granite-fold0.npz \
        --data-dir ../data \
        --focus-classes read_file,grep_search,list_directory,glob_pattern
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from action_router.features import session_group  # noqa: E402


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def last_action_name(history):
    for item in reversed(history or []):
        if item.get("role") == "assistant_action":
            return item.get("name")
    return None


def merge_npz(paths):
    """Merge OOF dumps by id: average probs where ids repeat, concatenate where
    they don't. Returns (ids, y_true, classes, probs) aligned on ACTION_CLASSES
    column order (guaranteed by the writers of this npz schema)."""
    classes = None
    sums = {}
    counts = {}
    y_true = {}
    for path in paths:
        dump = np.load(path, allow_pickle=True)
        these_classes = list(dump["classes"])
        if classes is None:
            classes = these_classes
        elif classes != these_classes:
            raise SystemExit(
                f"{path}: class order {these_classes} != {classes} from an earlier "
                "npz. All dumps must share ACTION_CLASSES column order."
            )
        ids = [str(i) for i in dump["ids"]]
        probs = dump["probs"].astype(np.float64)
        y = dump["y_true"].astype(np.int64)
        for i, row_id in enumerate(ids):
            if row_id in y_true and y_true[row_id] != int(y[i]):
                raise SystemExit(
                    f"{path}: y_true for {row_id} disagrees with an earlier npz "
                    "-- these dumps are not from the same labeling."
                )
            y_true[row_id] = int(y[i])
            if row_id in sums:
                sums[row_id] += probs[i]
                counts[row_id] += 1
            else:
                sums[row_id] = probs[i].copy()
                counts[row_id] = 1

    ordered_ids = sorted(sums)
    probs_out = np.stack([sums[i] / counts[i] for i in ordered_ids]).astype(np.float32)
    y_out = np.array([y_true[i] for i in ordered_ids], dtype=np.int64)
    n_from_ensemble = sum(1 for i in ordered_ids if counts[i] > 1)
    if n_from_ensemble:
        print(f"note: {n_from_ensemble} ids appeared in >1 npz and were prob-averaged")
    return ordered_ids, y_out, classes, probs_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof-npz", required=True,
                         help="comma-separated list of OOF npz dumps (ids,y_true,classes,logits,probs)")
    parser.add_argument("--data-dir", default="../data", help="dir with train.jsonl / train_labels.csv")
    parser.add_argument("--output", default="output/label_issues.csv")
    parser.add_argument("--top-k", type=int, default=50, help="rows to print to stdout")
    parser.add_argument("--focus-classes", default="",
                         help="comma list; if set, only report rows whose GIVEN label is in this set "
                              "(confident learning still runs on the full label set for a correct joint estimate)")
    parser.add_argument("--filter-by", choices=["prune_by_noise_rate", "prune_by_class", "both", "confident_learning", "predicted_neq_given"],
                         default="prune_by_noise_rate", help="cleanlab find_label_issues filter_by strategy")
    args = parser.parse_args()

    try:
        from cleanlab.filter import find_label_issues
        from cleanlab.rank import get_label_quality_scores
    except ImportError:
        raise SystemExit("cleanlab not installed. `pip install cleanlab` and re-run.")

    npz_paths = [p.strip() for p in args.oof_npz.split(",") if p.strip()]
    ids, y_true, classes, probs = merge_npz(npz_paths)
    print(f"merged OOF: n={len(ids)} classes={len(classes)}")

    issue_idx = find_label_issues(
        labels=y_true,
        pred_probs=probs,
        filter_by=args.filter_by,
        return_indices_ranked_by="self_confidence",
    )
    quality = get_label_quality_scores(y_true, probs)
    print(f"cleanlab flagged {len(issue_idx)} / {len(ids)} rows ({len(issue_idx) / len(ids):.2%}) "
          f"[filter_by={args.filter_by}]")

    samples_by_id = {str(s["id"]): s for s in load_jsonl(Path(args.data_dir) / "train.jsonl")}
    missing = [ids[i] for i in issue_idx if ids[i] not in samples_by_id]
    if missing:
        print(f"warning: {len(missing)} flagged ids not found in {args.data_dir}/train.jsonl "
              f"(e.g. {missing[:3]}); check --data-dir")

    focus = {c.strip() for c in args.focus_classes.split(",") if c.strip()} or None
    if focus:
        unknown = focus - set(classes)
        if unknown:
            raise SystemExit(f"--focus-classes has unknown class(es): {unknown}")

    rows = []
    for i in issue_idx:
        row_id = ids[i]
        given = classes[y_true[i]]
        if focus and given not in focus:
            continue
        pred_idx = int(probs[i].argmax())
        rows.append({
            "id": row_id,
            "given_label": given,
            "suggested_label": classes[pred_idx],
            "given_label_prob": float(probs[i][y_true[i]]),
            "suggested_label_prob": float(probs[i][pred_idx]),
            "label_quality_score": float(quality[i]),
            "session": session_group(row_id),
        })
    rows.sort(key=lambda r: r["label_quality_score"])

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "given_label", "suggested_label", "given_label_prob",
                           "suggested_label_prob", "label_quality_score", "session",
                           "prev_action", "current_prompt"],
        )
        writer.writeheader()
        for row in rows:
            sample = samples_by_id.get(row["id"], {})
            writer.writerow({
                **row,
                "prev_action": last_action_name(sample.get("history")),
                "current_prompt": (sample.get("current_prompt") or "")[:300],
            })
    print(f"wrote {len(rows)} rows to {args.output}"
          + (f" (filtered to given_label in {sorted(focus)})" if focus else ""))

    pair_counts = Counter((r["given_label"], r["suggested_label"]) for r in rows)
    print("\ntop given->suggested pairs among flagged rows:")
    for (given, suggested), count in pair_counts.most_common(15):
        print(f"  {given:18s} -> {suggested:18s} {count:5d}")

    print(f"\ntop {min(args.top_k, len(rows))} lowest label-quality rows:")
    for row in rows[: args.top_k]:
        sample = samples_by_id.get(row["id"], {})
        prompt = (sample.get("current_prompt") or "")[:100]
        print(
            f"  q={row['label_quality_score']:.4f}  {row['given_label']:16s} -> "
            f"{row['suggested_label']:16s}  prev={str(last_action_name(sample.get('history'))):16s}  "
            f"{row['id']}  {prompt!r}"
        )


if __name__ == "__main__":
    main()
