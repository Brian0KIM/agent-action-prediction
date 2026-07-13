"""Evaluate a margin-gated pairwise hardcode: for rows where the model's
top1-top2 probability margin is below --margin-threshold (default 0.2),
override the prediction using an empirically-learned (top1_class, top2_class)
-> which-is-usually-right prior, instead of trusting the model's own (weak,
low-confidence) ranking.

This is E27's "flip rule" idea (digital-competition SUBMISSIONS.md row 19:
local slice +0.0012, but LB -0.00011 -- NEUTRAL, did NOT transfer), narrowed
to a single, simple margin threshold instead of an (r1,r2,decile) cell grid.

CRITICAL: the pair-prior is fit on one CV fold and evaluated on a DIFFERENT,
disjoint fold (session-grouped, no leakage) -- fitting and testing on the
same rows is exactly the mistake that made E27's local number look better
than it really was. This script reports the CROSS-VALIDATED gain, which is
the closest local proxy to "will this survive contact with hidden test" that
you can get without an actual LB submission -- still not a guarantee (E27
proves local-clean-CV can still fail to transfer), but a necessary filter
before spending a submission slot on it.

Takes an OOF npz (ids, y_true, classes, logits/probs) from a model with a
REAL held-out fold (not --split-mode all). Splits those OOF rows into
--cv-folds groups (by session, reusing session_group) for the fit/test split.

Example
-------
    python scripts/margin_hardcode_eval.py \
        --npz output/oof/granite-fold0.npz \
        --margin-threshold 0.2 --cv-folds 5 \
        --focus-classes read_file,grep_search,list_directory,glob_pattern
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from action_router.features import session_group  # noqa: E402


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def macro_f1(y_true, y_pred, n_classes=None, class_list=None):
    """Average F1 over class_list if given, else over range(n_classes).
    NOTE: when scoring a SUBSET of rows restricted to a few classes (e.g.
    --focus-classes), you MUST pass class_list=those classes -- averaging
    over all 14 with n_classes=14 silently zeroes out the 10 absent classes
    (0 recall by construction) and craters the reported macro-F1."""
    classes = class_list if class_list is not None else range(n_classes)
    scores = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        scores.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(scores))


def fit_pair_prior(top1, top2, y_true, min_support):
    """For each (top1_class, top2_class) pair, whether top1 or top2 wins more
    often empirically. min_support: pairs with fewer than this many fit-fold
    rows fall back to 'trust top1' (not enough data to override confidently)."""
    pair_counts = defaultdict(lambda: Counter())
    for t1, t2, y in zip(top1, top2, y_true):
        pair_counts[(t1, t2)][y] += 1
    prior = {}
    for pair, counts in pair_counts.items():
        n = sum(counts.values())
        t1, t2 = pair
        if n < min_support:
            prior[pair] = t1  # not enough data -- default to trusting top1
            continue
        prior[pair] = t1 if counts[t1] >= counts[t2] else t2
    return prior, pair_counts


def apply_hardcode(top1, top2, prior, default_class):
    out = np.empty(len(top1), dtype=np.int64)
    for i, (t1, t2) in enumerate(zip(top1, top2)):
        out[i] = prior.get((t1, t2), default_class[i])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="OOF dump (ids,y_true,classes,logits/probs)")
    ap.add_argument("--margin-threshold", type=float, default=0.2)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--min-support", type=int, default=30,
                     help="min fit-fold rows for a (top1,top2) pair before the prior overrides it")
    ap.add_argument("--focus-classes", default="",
                     help="comma list; if set, only rows whose GIVEN (y_true) label is in this "
                          "set count toward the reported macro-F1 delta (still fits priors on all rows)")
    ap.add_argument("--force-class", default="",
                     help="if set, SKIP the pair-prior fit entirely and just force every gated row's "
                          "prediction to this one class (e.g. list_directory). No fitting means no "
                          "leakage risk -- applied over the whole sample directly, not per-CV-fold, "
                          "since there's nothing learned from the data to overfit.")
    args = ap.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    ids = [str(i) for i in data["ids"]]
    y = data["y_true"].astype(np.int64)
    classes = list(data["classes"])
    nc = len(classes)
    logits = data["logits"].astype(np.float32) if "logits" in data else np.log(data["probs"] + 1e-9)
    probs = softmax(logits)

    order = np.argsort(-probs, axis=1)
    top1 = order[:, 0]
    top2 = order[:, 1]
    p1 = probs[np.arange(len(probs)), top1]
    p2 = probs[np.arange(len(probs)), top2]
    margin = p1 - p2
    gated = margin < args.margin_threshold
    print(f"n={len(ids)}  gated (margin < {args.margin_threshold}): {gated.sum()} "
          f"({gated.mean():.2%})")

    groups = np.array([session_group(i) for i in ids], dtype=object)
    unique_groups = np.array(sorted(set(groups)))
    rng = np.random.default_rng(42)
    rng.shuffle(unique_groups)
    fold_of_group = {g: i % args.cv_folds for i, g in enumerate(unique_groups)}
    fold_id = np.array([fold_of_group[g] for g in groups])

    focus = {c.strip() for c in args.focus_classes.split(",") if c.strip()} or None
    if focus:
        focus_idx = {classes.index(c) for c in focus}

    baseline_preds = top1.copy()
    hardcoded_preds = top1.copy()

    if args.force_class:
        force_idx = classes.index(args.force_class)
        gated_idx = np.nonzero(gated)[0]
        hardcoded_preds[gated_idx] = force_idx
        print(f"forced {len(gated_idx)} gated rows to '{args.force_class}' (no fitting -- "
              f"nothing learned from data, so no CV needed for this mode)")
    else:
        for k in range(args.cv_folds):
            test_mask = fold_id == k
            fit_mask = ~test_mask
            fit_gated = fit_mask & gated
            prior, pair_counts = fit_pair_prior(top1[fit_gated], top2[fit_gated], y[fit_gated], args.min_support)

            test_gated_idx = np.nonzero(test_mask & gated)[0]
            if len(test_gated_idx) == 0:
                continue
            overridden = apply_hardcode(top1[test_gated_idx], top2[test_gated_idx], prior, top1[test_gated_idx])
            hardcoded_preds[test_gated_idx] = overridden

    def report(preds, label):
        if focus:
            keep = np.isin(y, list(focus_idx))
            f1 = macro_f1(y[keep], preds[keep], class_list=sorted(focus_idx))
            acc = (preds[keep] == y[keep]).mean()
            print(f"{label:>12s}  focus-class macro-F1={f1:.4f}  acc={acc:.4f}  n={keep.sum()}")
        f1_all = macro_f1(y, preds, n_classes=nc)
        acc_all = (preds == y).mean()
        print(f"{label:>12s}  overall     macro-F1={f1_all:.4f}  acc={acc_all:.4f}  n={len(y)}")
        return f1_all

    mode_label = "forced-class (no fitting, whole sample)" if args.force_class else "cross-validated (fit fold != test fold, no leakage)"
    print(f"\n=== {mode_label} ===")
    f1_base = report(baseline_preds, "baseline")
    f1_hard = report(hardcoded_preds, "hardcoded")
    print(f"\ndelta macro-F1 (overall): {f1_hard - f1_base:+.4f}")

    changed = baseline_preds != hardcoded_preds
    fixed = changed & (hardcoded_preds == y) & (baseline_preds != y)
    broke = changed & (hardcoded_preds != y) & (baseline_preds == y)
    print(f"rows changed by hardcode: {changed.sum()}  (fixed {fixed.sum()}, broke {broke.sum()}, "
          f"net {fixed.sum() - broke.sum():+d})")


if __name__ == "__main__":
    main()
