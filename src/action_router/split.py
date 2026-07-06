import numpy as np
from sklearn.model_selection import GroupKFold, train_test_split


def split_train_val(texts, y, groups, split_mode, fold=0, n_splits=5, val_size=0.2, seed=42):
    if split_mode == "all":
        idx = np.arange(len(texts))
        return texts.tolist(), [], y, np.array([], dtype=y.dtype), False, idx, np.array([], dtype=np.int64)

    if split_mode == "group":
        splits = list(GroupKFold(n_splits=n_splits).split(texts, y, groups))
        train_idx, val_idx = splits[fold]
        return (
            texts[train_idx].tolist(),
            texts[val_idx].tolist(),
            y[train_idx],
            y[val_idx],
            True,
            train_idx,
            val_idx,
        )

    if split_mode == "stratified":
        train_idx, val_idx = train_test_split(
            np.arange(len(texts)),
            test_size=val_size,
            stratify=y,
            random_state=seed,
        )
        return (
            texts[train_idx].tolist(),
            texts[val_idx].tolist(),
            y[train_idx],
            y[val_idx],
            True,
            train_idx,
            val_idx,
        )

    raise ValueError(f"unknown split_mode: {split_mode}")
