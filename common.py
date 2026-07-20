"""Shared helpers used by the methods, the QC metrics, and the harness.

Nothing here is a correction method or a metric on its own. It is the plumbing
they all need: loading the two-file schema, per-sample median normalization,
bridge-replicate detection, and small design-matrix / OLS utilities.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# IO and preprocessing
# --------------------------------------------------------------------------

def load_dataset(abundance_path, samples_path):
    """Read the abundance matrix and sample table into aligned frames.

    abundance.csv: leading ``Protein`` column, one column per sample, log2
    abundance values, blanks for not-detected. A literal 0 is also treated as
    not-detected (converted to NaN) because some export tools write 0 there.

    samples.csv: ``Sample`` plus ``Batch``, ``Group``, ``Subject``, ``Stratum``.

    Returns (X, meta) where X is proteins x samples (float, NaN = missing) and
    meta is indexed by sample. Only samples present in both files are kept, in
    the sample-table order.
    """
    X = pd.read_csv(abundance_path, index_col=0)
    X.index.name = "Protein"

    meta = pd.read_csv(samples_path, index_col="Sample")
    meta["Batch"] = meta["Batch"].astype(int)
    for col in ("Group", "Subject", "Stratum"):
        meta[col] = meta[col].astype(str)

    common = [s for s in meta.index if s in X.columns]
    if not common:
        raise ValueError("No samples are shared between abundance.csv and samples.csv")
    X = X[common].astype(float)
    meta = meta.loc[common]

    # not-detected convention: 0 -> NaN
    X = X.replace(0, np.nan)
    return X, meta


def median_normalize(X):
    """Center each sample so its median matches the global median.

    Global median is the median of the per-sample medians (robust to a few
    off samples). The NaN pattern is preserved.
    """
    arr = X.to_numpy(dtype=float, copy=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sample_med = np.nanmedian(arr, axis=0)          # one median per sample
        global_med = np.nanmedian(sample_med)
    shift = np.nan_to_num(sample_med - global_med)
    arr = arr - shift[None, :]
    return pd.DataFrame(arr, index=X.index, columns=X.columns)


# --------------------------------------------------------------------------
# Bridge-replicate detection
# --------------------------------------------------------------------------

def detect_bridges(meta, ref_batch=None):
    """Find bridge subjects and the reference batch within one stratum's meta.

    A bridge subject is one that appears in more than one batch. Its appearance
    in the reference batch is the biology instance; its other appearances are
    bridge replicates (same biology, different batch).

    ``ref_batch`` defaults to the batch holding the most distinct bridge
    subjects, falling back to the smallest batch id (also used when there are
    no bridges at all).

    Returns a namespace with ``by_subject`` ({subject: {batch: [sample, ...]}}),
    ``bridge_subjects`` (sorted list), and ``ref_batch`` (int).
    """
    by_subject = {}
    for sample, row in meta.iterrows():
        subj = str(row["Subject"])
        batch = int(row["Batch"])
        by_subject.setdefault(subj, {}).setdefault(batch, []).append(sample)

    bridge_subjects = sorted(s for s, bmap in by_subject.items() if len(bmap) > 1)

    if ref_batch is None:
        ref_batch = _default_ref_batch(meta, bridge_subjects, by_subject)

    return SimpleNamespace(
        by_subject=by_subject,
        bridge_subjects=bridge_subjects,
        ref_batch=int(ref_batch),
    )


def _default_ref_batch(meta, bridge_subjects, by_subject):
    batches = sorted(int(b) for b in pd.unique(meta["Batch"]))
    if not bridge_subjects:
        return batches[0]
    counts = {b: 0 for b in batches}
    for subj in bridge_subjects:
        for batch in by_subject[subj]:
            counts[batch] += 1
    top = max(counts.values())
    return sorted(b for b, c in counts.items() if c == top)[0]


def bridge_replicate_samples(meta, ref_batch=None):
    """Flat list of every sample belonging to a bridge subject in this stratum."""
    br = detect_bridges(meta, ref_batch=ref_batch)
    out = []
    for subj in br.bridge_subjects:
        for _batch, samples in br.by_subject[subj].items():
            out.extend(samples)
    return out


# --------------------------------------------------------------------------
# Design matrices and OLS
# --------------------------------------------------------------------------

def dummies(values, levels=None, drop="first"):
    """Indicator columns for a categorical vector.

    ``drop`` = "first" drops the first level (treatment coding), ``None`` keeps
    every level (full dummies), or a specific level value drops that level. If
    the requested drop level is absent, the first level is dropped instead.

    Returns (matrix float[n, k], kept_levels).
    """
    values = np.asarray(values)
    if levels is None:
        levels = sorted(pd.unique(values).tolist())
    else:
        levels = list(levels)

    if drop is None:
        kept = levels
    else:
        drop_level = levels[0] if drop == "first" or drop not in levels else drop
        kept = [lv for lv in levels if lv != drop_level]

    M = np.zeros((len(values), len(kept)), dtype=float)
    for j, lv in enumerate(kept):
        M[:, j] = (values == lv).astype(float)
    return M, kept


def biological_design(meta, samples, drop="first"):
    """Treatment-coded design for the biological covariates to protect.

    Uses ``Group`` and, if the column exists, ``Sex``. Returns (matrix, labels)
    with no intercept column (callers add their own).
    """
    parts, labels = [], []
    for col in ("Group", "Sex"):
        if col in meta.columns and meta.loc[samples, col].nunique() > 1:
            M, kept = dummies(meta.loc[samples, col].to_numpy(), drop=drop)
            parts.append(M)
            labels.extend(f"{col}:{lv}" for lv in kept)
    if not parts:
        return np.zeros((len(samples), 0)), labels
    return np.hstack(parts), labels


def ols_beta(y, A):
    """Least-squares coefficients for y ~ A via numpy (no statsmodels)."""
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta


def squared_distance_matrix(M):
    """Pairwise squared Euclidean distances between the rows of M."""
    gram = M @ M.T
    sq = np.diag(gram)
    d2 = sq[:, None] + sq[None, :] - 2.0 * gram
    return np.maximum(d2, 0.0)
