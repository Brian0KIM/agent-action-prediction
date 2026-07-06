"""Evaluate a 2-way probability-averaging blend of two routers.

Takes two `.npz` prob dumps with the shared schema (ids, y_true, classes,
logits, probs) and reports, on the intersection of their ids:

  - each model's solo macro-F1,
  - a weight sweep for  w*A + (1-w)*B  with the best macro,
  - per-class F1 at the best weight (highlighting the exploration cluster),
  - decorrelation stats: Jaccard of correct-sets, and how many samples B
    fixes / breaks relative to A.

Works identically on fp dumps and int8 dumps, so you can compare:

    # fp blend
    python scripts/blend_eval.py --npz-a granite-fold0.npz --npz-b bge-m3-fold0.npz

    # int8 blend (after verify_int8.py)
    python scripts/blend_eval.py --npz-a granite-fold0-int8.npz --npz-b bge-m3-fold0-int8.npz

The difference in "best blend macro" between the two runs is exactly the int8
penalty on the ensemble.

IMPORTANT: for the blend to be valid, both dumps must come from the SAME fold
split (identical ids). If the printed "aligned on N shared ids" warning shows
fewer ids than expected, the two models were dumped on different splits and the
blend numbers are meaningless.

Note: this is single-fold. A single sweep will slightly overfit the weight to
this fold, so treat the printed best-w as indicative. For a defensible weight,
pool OOF across folds 1-4 and re-run with --weight fixed to the pooled optimum.
"""

import argparse
import numpy as np

# Exploration cluster from the analysis: where multilingual diversity helps.
EXPLORE = ["read_file", "grep_search", "list_directory", "glob_pattern"]


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
    """Return probs_a, probs_b, y_true aligned on the intersection of ids (a's order)."""
    ids_a = [str(i) for i in a["ids"]]
    ids_b = [str(i) for i in b["ids"]]
    idx_b = {i: k for k, i in enumerate(ids_b)}
    keep = [(k, idx_b[i]) for k, i in enumerate(ids_a) if i in idx_b]
    if len(keep) != len(ids_a) or len(keep) != len(ids_b):
        print(f"note: aligned on {len(keep)} shared ids (a={len(ids_a)}, b={len(ids_b)})")
    ka = [k for k, _ in keep]
    kb = [k for _, k in keep]
    ya = a["y_true"][ka]
    yb = b["y_true"][kb]
    if not np.array_equal(ya, yb):
        raise SystemExit("y_true mismatch on shared ids; are these the same fold?")
    return a["probs"][ka], b["probs"][kb], ya


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz-a", required=True, help="e.g. granite dump")
    parser.add_argument("--npz-b", required=True, help="e.g. bge-m3 dump")
    parser.add_argument("--name-a", default="A")
    parser.add_argument("--name-b", default="B")
    parser.add_argument("--weight", type=float, default=None,
                        help="Fixed weight on A (skip sweep). Blend = w*A + (1-w)*B.")
    parser.add_argument("--step", type=float, default=0.05)
    args = parser.parse_args()

    a = np.load(args.npz_a, allow_pickle=True)
    b = np.load(args.npz_b, allow_pickle=True)
    classes = list(a["classes"])
    nc = len(classes)

    pa, pb, y = align(a, b)

    ma, _ = macro_f1(y, pa.argmax(1), nc)
    mb, _ = macro_f1(y, pb.argmax(1), nc)
    print(f"{args.name_a:>10s} solo macro-F1 = {ma:.4f}")
    print(f"{args.name_b:>10s} solo macro-F1 = {mb:.4f}")

    # ---- weight sweep (or fixed) ----
    if args.weight is not None:
        weights = [args.weight]
    else:
        weights = [round(w, 2) for w in np.arange(0.0, 1.0 + 1e-9, args.step)]

    print("\nweight sweep (w on A):")
    best = (-1.0, None, None)
    for w in weights:
        blend = w * pa + (1 - w) * pb
        m, _ = macro_f1(y, blend.argmax(1), nc)
        if m > best[0]:
            best = (m, w, blend)
        print(f"  w={w:.2f}  macro={m:.4f}")
    best_m, best_w, best_blend = best
    print(f"\nbest: w={best_w:.2f} (A) / {1 - best_w:.2f} (B)  ->  macro={best_m:.4f}")
    print(f"gain vs {args.name_a}: {best_m - ma:+.4f}   vs {args.name_b}: {best_m - mb:+.4f}")

    # ---- per-class at best weight, exploration cluster highlighted ----
    _, fs_a = macro_f1(y, pa.argmax(1), nc)
    _, fs_bl = macro_f1(y, best_blend.argmax(1), nc)
    print(f"\nper-class F1  ({args.name_a} -> blend@w={best_w:.2f}):")
    for i, cls in enumerate(classes):
        tag = "  <- explore" if cls in EXPLORE else ""
        print(f"  {cls:18s} {fs_a[i]:.3f} -> {fs_bl[i]:.3f}  ({fs_bl[i] - fs_a[i]:+.3f}){tag}")

    # ---- decorrelation ----
    ca = pa.argmax(1) == y
    cb = pb.argmax(1) == y
    inter = np.sum(ca & cb)
    union = np.sum(ca | cb)
    jac = inter / union if union else 0.0
    fixes = int(np.sum(~ca & cb))   # B right where A wrong
    breaks = int(np.sum(ca & ~cb))  # B wrong where A right
    print("\ndecorrelation:")
    print(f"  Jaccard(correct_A, correct_B) = {jac:.3f}")
    print(f"  {args.name_b} fixes {fixes} of {args.name_a}'s errors, breaks {breaks} of its correct")


if __name__ == "__main__":
    main()
