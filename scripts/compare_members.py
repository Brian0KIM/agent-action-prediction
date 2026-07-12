"""Characterize what's actually DIFFERENT between two members' predictions --
not just the blend score (blend_eval.py does that), but WHERE they disagree
and WHO tends to be right when they do. This is the diagnostic behind
"disagreement is where ensemble gains live" (digital-competition's E26 notes).

Takes two `.npz` OOF dumps with the shared schema (ids, y_true, classes,
logits/probs) and reports:
  - each model's solo macro-F1 (on the shared id intersection)
  - overall prediction agreement rate
  - per-class disagreement confusion (A's pred vs B's pred, rows where they differ)
  - of the disagreement rows: how often A alone is right, B alone is right,
    both right (can't happen by definition here), both wrong
  - which TRUE classes concentrate the disagreement (row's y_true, not pred)

Example
-------
    python scripts/compare_members.py \
        --npz-a output/oof/granite-aum06-fold0.npz --name-a granite_aum06 \
        --npz-b output/oof/gte-full-fold0.npz --name-b gte
"""
import argparse
from collections import Counter

import numpy as np


def macro_f1(y_true, y_pred, n_classes):
    scores = []
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        scores.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(scores)), scores


def align(a, b):
    ids_a = [str(i) for i in a["ids"]]
    ids_b = [str(i) for i in b["ids"]]
    idx_b = {i: k for k, i in enumerate(ids_b)}
    keep = [(k, idx_b[i]) for k, i in enumerate(ids_a) if i in idx_b]
    if len(keep) != len(ids_a) or len(keep) != len(ids_b):
        print(f"note: aligned on {len(keep)} shared ids (a={len(ids_a)}, b={len(ids_b)})")
    ka = [k for k, _ in keep]
    kb = [k for _, k in keep]
    ya, yb = a["y_true"][ka], b["y_true"][kb]
    if not np.array_equal(ya, yb):
        raise SystemExit("y_true mismatch on shared ids; are these dumps from the same labeling?")
    logits_a = a["logits"][ka] if "logits" in a else np.log(a["probs"][ka] + 1e-9)
    logits_b = b["logits"][kb] if "logits" in b else np.log(b["probs"][kb] + 1e-9)
    return logits_a, logits_b, ya


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-a", required=True)
    ap.add_argument("--npz-b", required=True)
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    ap.add_argument("--top-k-pairs", type=int, default=15)
    args = ap.parse_args()

    a = np.load(args.npz_a, allow_pickle=True)
    b = np.load(args.npz_b, allow_pickle=True)
    classes = list(a["classes"])
    nc = len(classes)

    la, lb, y = align(a, b)
    pa, pb = la.argmax(1), lb.argmax(1)

    fa, fs_a = macro_f1(y, pa, nc)
    fb, fs_b = macro_f1(y, pb, nc)
    print(f"{args.name_a:>14s} solo macro-F1 = {fa:.4f}")
    print(f"{args.name_b:>14s} solo macro-F1 = {fb:.4f}")

    disagree = pa != pb
    n = len(y)
    print(f"\noverall agreement: {(~disagree).mean():.4f}  ({n - disagree.sum()}/{n} rows)")
    print(f"disagreement rate: {disagree.mean():.4f}  ({disagree.sum()}/{n} rows)")

    a_right = disagree & (pa == y)
    b_right = disagree & (pb == y)
    both_wrong = disagree & (pa != y) & (pb != y)
    print(f"\nof disagreement rows:")
    print(f"  {args.name_a} right, {args.name_b} wrong : {a_right.sum():5d}  ({a_right.sum()/max(1,disagree.sum()):.2%})")
    print(f"  {args.name_b} right, {args.name_a} wrong : {b_right.sum():5d}  ({b_right.sum()/max(1,disagree.sum()):.2%})")
    print(f"  both wrong (different mistakes)      : {both_wrong.sum():5d}  ({both_wrong.sum()/max(1,disagree.sum()):.2%})")
    print(f"  -> if you always deferred to {args.name_b} on disagreement: "
          f"fixes {b_right.sum()} of {args.name_a}'s misses, breaks {a_right.sum()} of {args.name_a}'s hits "
          f"(net {b_right.sum() - a_right.sum():+d})")

    print(f"\nper-class F1 ({args.name_a} vs {args.name_b}):")
    for i, cls in enumerate(classes):
        print(f"  {cls:18s} {fs_a[i]:.3f}  vs  {fs_b[i]:.3f}   (Δ {fs_b[i]-fs_a[i]:+.3f})")

    print(f"\ntop true-class where disagreement concentrates:")
    true_counts = Counter(classes[c] for c in y[disagree])
    for cls, count in true_counts.most_common(args.top_k_pairs):
        print(f"  y_true={cls:18s} {count:5d} disagreement rows ({count/max(1,disagree.sum()):.2%} of all disagreement)")

    print(f"\ntop ({args.name_a}_pred, {args.name_b}_pred) pairs among disagreement rows:")
    pair_counts = Counter((classes[pa[i]], classes[pb[i]]) for i in np.nonzero(disagree)[0])
    for (ca, cb), count in pair_counts.most_common(args.top_k_pairs):
        print(f"  {args.name_a}={ca:18s} {args.name_b}={cb:18s} {count:5d}")


if __name__ == "__main__":
    main()
