"""The six QC metrics that score a correction against the bridge ground truth.

Bridge replicates are the same biological sample measured in different batches,
so any spread between them is batch effect, not biology. The metrics use that
to tell a good correction (bridge replicates converge, batch structure drops)
from a destructive one (biological signal erased, missingness broken).

  Tier 1  bridge_mad          lower  is better
          pca_batch_r2        lower  is better
          bio_preservation    higher is better
  Tier 2  group_separation    higher is better
          missingness_preserved   must stay True (else disqualified)
  Tier 3  bridge_coverage     diagnostic

Each function takes a single stratum's matrix and meta.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from common import (
    biological_design,
    bridge_replicate_samples,
    detect_bridges,
    dummies,
    ols_beta,
    squared_distance_matrix,
)


def bridge_mad(X, meta):
    """Median across bridge replicate pairs of the mean absolute difference.

    For each bridge subject with >= 2 replicates, every pair of its replicates
    contributes the mean |difference| over the proteins both detect. Good
    correction pulls replicates together, so this drops.
    """
    br = detect_bridges(meta)
    pair_scores = []
    for subj in br.bridge_subjects:
        samples = [s for samples in br.by_subject[subj].values() for s in samples]
        if len(samples) < 2:
            continue
        for a, b in itertools.combinations(samples, 2):
            va = X[a].to_numpy(dtype=float)
            vb = X[b].to_numpy(dtype=float)
            both = ~np.isnan(va) & ~np.isnan(vb)
            if not both.any():
                continue
            pair_scores.append(float(np.mean(np.abs(va[both] - vb[both]))))
    if not pair_scores:
        return np.nan
    return float(np.median(pair_scores))


def pca_batch_r2(X, meta):
    """PERMANOVA R^2 of batch on complete-case Euclidean distances.

    Uses the standard partition of squared distances: SS_total = sum of squared
    pairwise distances / n, SS_within = sum over batches of the within-batch
    squared distances / n_batch, R^2 = 1 - SS_within / SS_total. Higher R^2
    means more of the sample-to-sample variation lines up with batch, so lower
    is better after correction.
    """
    samples = list(X.columns)
    arr = X.to_numpy(dtype=float)
    complete = ~np.isnan(arr).any(axis=1)
    M = arr[complete, :].T                      # samples x complete-case proteins
    n = M.shape[0]
    if M.shape[1] < 1 or n < 3:
        return np.nan

    d2 = squared_distance_matrix(M)
    ss_total = np.triu(d2, 1).sum() / n
    if ss_total <= 0:
        return np.nan

    batch_vals = meta.loc[samples, "Batch"].to_numpy()
    ss_within = 0.0
    for b in np.unique(batch_vals):
        idx = np.where(batch_vals == b)[0]
        if len(idx) < 2:
            continue
        sub = d2[np.ix_(idx, idx)]
        ss_within += np.triu(sub, 1).sum() / len(idx)

    return float(1.0 - ss_within / ss_total)


def bio_preservation(X_corrected, X_uncorrected, meta, ref_batch):
    """Pearson correlation of per-protein group effect sizes, corrected vs not.

    On uncorrected data the group effect is estimated with batch as a covariate
    (value ~ group + batch); on corrected data it is estimated without batch
    (value ~ group). If the correction protected biology the two agree, so the
    correlation stays high.
    """
    samples = list(X_corrected.columns)
    unc = _group_effect(X_uncorrected, meta, samples, include_batch=True, ref_batch=ref_batch)
    cor = _group_effect(X_corrected, meta, samples, include_batch=False, ref_batch=ref_batch)
    both = ~np.isnan(unc) & ~np.isnan(cor)
    if both.sum() < 3:
        return np.nan
    if np.std(unc[both]) == 0 or np.std(cor[both]) == 0:
        return np.nan
    return float(np.corrcoef(unc[both], cor[both])[0, 1])


def _group_effect(X, meta, samples, include_batch, ref_batch):
    """Per-protein coefficient on the first non-reference group level."""
    groups = meta.loc[samples, "Group"].to_numpy()
    group_levels = sorted(pd.unique(groups).tolist())
    if len(group_levels) < 2:
        return np.full(X.shape[0], np.nan)

    group_dum, _ = dummies(groups, levels=group_levels, drop="first")
    intercept = np.ones((len(samples), 1))
    if include_batch:
        batch_dum, _ = dummies(meta.loc[samples, "Batch"].to_numpy(), drop=ref_batch)
    else:
        batch_dum = np.zeros((len(samples), 0))

    arr = X[samples].to_numpy(dtype=float)
    coef = np.full(arr.shape[0], np.nan)
    for i in range(arr.shape[0]):
        y = arr[i, :]
        obs = ~np.isnan(y)
        # need at least two observations per group to estimate the effect
        if any(np.sum(obs & (groups == lv)) < 2 for lv in group_levels):
            continue
        A = np.hstack([intercept[obs], group_dum[obs, :], batch_dum[obs, :]])
        if obs.sum() <= A.shape[1]:
            continue
        beta = ols_beta(y[obs], A)
        coef[i] = beta[1]                        # first group dummy (col 0 is intercept)
    return coef


def group_separation(X, meta):
    """Mean silhouette width of the group labels in complete-case space."""
    samples = list(X.columns)
    arr = X.to_numpy(dtype=float)
    complete = ~np.isnan(arr).any(axis=1)
    M = arr[complete, :].T
    groups = meta.loc[samples, "Group"].to_numpy()
    labels = np.unique(groups)
    if M.shape[1] < 1 or len(labels) < 2 or M.shape[0] <= len(labels):
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return float(silhouette_score(M, groups, metric="euclidean"))
        except ValueError:
            return np.nan


def missingness_preserved(X_corrected, X_uncorrected):
    """True iff the corrected matrix has exactly the uncorrected NaN pattern."""
    a = np.isnan(X_corrected.to_numpy(dtype=float))
    b = np.isnan(X_uncorrected.to_numpy(dtype=float))
    return bool(a.shape == b.shape and np.array_equal(a, b))


def bridge_coverage(X, meta):
    """Fraction of proteins detected in every bridge replicate (diagnostic)."""
    samples = bridge_replicate_samples(meta)
    if not samples:
        return np.nan
    sub = X[samples].to_numpy(dtype=float)
    detected_everywhere = ~np.isnan(sub).any(axis=1)
    return float(np.mean(detected_everywhere))


METRIC_DIRECTION = {
    "bridge_mad": "lower",
    "pca_batch_r2": "lower",
    "bio_preservation": "higher",
    "group_separation": "higher",
    "missingness_preserved": "must-hold",
    "bridge_coverage": "diagnostic",
}


def compute_all(X_corrected, X_uncorrected, meta, ref_batch):
    """All six metrics for one corrected stratum as a plain dict."""
    return {
        "bridge_mad": bridge_mad(X_corrected, meta),
        "pca_batch_r2": pca_batch_r2(X_corrected, meta),
        "bio_preservation": bio_preservation(X_corrected, X_uncorrected, meta, ref_batch),
        "group_separation": group_separation(X_corrected, meta),
        "missingness_preserved": missingness_preserved(X_corrected, X_uncorrected),
        "bridge_coverage": bridge_coverage(X_corrected, meta),
    }
