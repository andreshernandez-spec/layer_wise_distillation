"""A real-data budget for the harvest (docs/08).

Under the data-limited protocol everything real that any arm touches has to come from the
same D tokens: the interface statistics (so also the whitening contract and the noise
moments), the anchors, the heal's top-k store, and the validation split used for stopping
and selection. Budgets are nested prefixes of the declared slice, so a smaller budget is a
subset of a larger one and what changes between them is only the amount of data.
"""
from __future__ import annotations

import hashlib

import numpy as np


def split_budget(rows: np.ndarray, budget_rows: int, val_frac: float = 0.1, min_val: int = 4,
                 seq_len: int | None = None):
    """First `budget_rows` rows; the last `val_frac` of them (at least `min_val`) are the
    validation split, the rest are training rows. Returns (train, val, meta)."""
    assert 0 < budget_rows <= len(rows), (budget_rows, len(rows))
    n_val = max(min_val, int(round(val_frac * budget_rows)))
    assert n_val < budget_rows, f"budget of {budget_rows} rows leaves nothing to train on"
    sub = rows[:budget_rows]
    train, val = sub[: budget_rows - n_val], sub[budget_rows - n_val:]
    # tokens actually read: the harvest truncates rows to seq_len, and a budget stated in
    # tokens has to count what was touched, not the width of the file
    L = seq_len if seq_len is not None else rows.shape[1] - 1
    sha = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
    meta = {"budget_rows": int(budget_rows), "train_rows": int(len(train)), "val_rows": int(len(val)),
            "seq_len": int(L), "budget_tokens": int(len(sub) * L), "train_tokens": int(len(train) * L),
            "sha256_budget": sha(sub), "sha256_train": sha(train), "sha256_val": sha(val)}
    return train, val, meta
