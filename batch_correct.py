"""The five batch-correction methods, behind one uniform interface.

Every method has the signature ``method(X, meta, ref_batch) -> X_corrected``
where X is proteins x samples for a single stratum, meta is that stratum's
sample table, and ref_batch is the batch the correction anchors to. Output has
the same shape, index, columns, and (unless noted) the same NaN pattern.

Implemented from the source logic, no external batch-correction libraries:

  bridge_median   global additive offset from bridge replicates
  limma_rbe       per-protein design-aware batch removal (removeBatchEffect)
  combat_param    parametric empirical-Bayes ComBat
  combat_bridge   ComBat, mean-only (location) variant
  median_polish   Tukey median polish of per-batch protein medians
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from common import (
    biological_design,
    detect_bridges,
    dummies,
    ols_beta,
)

DEFAULT_METHOD = "limma_rbe"


# --------------------------------------------------------------------------
# 1. bridge_median
# --------------------------------------------------------------------------

def bridge_median(X, meta, ref_batch, min_deltas=3, min_shared=100):
    """Global additive correction anchored on bridge replicates.

    For each non-reference batch, the offset is the median (over bridge
    subjects) of each subject's median protein-wise shift relative to its
    reference-batch replicate. The offset is subtracted from every sample in
    that batch. A single scalar per batch, so it cannot fix protein-specific
    batch effects; it is the baseline the per-protein methods should beat.
    """
    br = detect_bridges(meta, ref_batch=ref_batch)
    ref_batch = br.ref_batch
    arr = X.to_numpy(dtype=float, copy=True)
    col_index = {s: i for i, s in enumerate(X.columns)}
    batches = sorted(int(b) for b in pd.unique(meta["Batch"]))

    for batch in batches:
        if batch == ref_batch:
            continue
        deltas = []
        for subj in br.bridge_subjects:
            locs = br.by_subject[subj]
            if ref_batch not in locs or batch not in locs:
                continue
            v_ref = arr[:, col_index[locs[ref_batch][0]]]
            v_bat = arr[:, col_index[locs[batch][0]]]
            shared = ~np.isnan(v_ref) & ~np.isnan(v_bat)
            n_shared = int(shared.sum())
            if n_shared == 0:
                continue
            if n_shared < min_shared:
                warnings.warn(
                    f"bridge_median: only {n_shared} shared proteins for subject "
                    f"{subj} between batch {batch} and reference {ref_batch}",
                    stacklevel=2,
                )
            deltas.append(np.median(v_bat[shared] - v_ref[shared]))

        if not deltas:
            warnings.warn(
                f"bridge_median: no usable bridge deltas for batch {batch}; "
                "left uncorrected",
                stacklevel=2,
            )
            continue
        if len(deltas) < min_deltas:
            warnings.warn(
                f"bridge_median: only {len(deltas)} bridge delta(s) for batch "
                f"{batch}; offset is unstable",
                stacklevel=2,
            )
        offset = float(np.median(deltas))
        cols = [col_index[s] for s in meta.index[meta["Batch"] == batch]]
        arr[:, cols] -= offset

    return pd.DataFrame(arr, index=X.index, columns=X.columns)


# --------------------------------------------------------------------------
# 2. limma_rbe  (limma::removeBatchEffect)
# --------------------------------------------------------------------------

def limma_rbe(X, meta, ref_batch):
    """Per-protein batch removal that protects the biological design.

    For each protein, fit value ~ [intercept, biology design, batch dummies]
    on the observed samples and subtract only the fitted batch part. Batch
    dummies drop the reference level, so the correction shifts non-reference
    batches onto the reference. The biology design (Group, and Sex if present)
    stays in the model, so group differences are not regressed out. NaNs are
    preserved. This is the default method.
    """
    samples = list(X.columns)
    n = len(samples)

    design_bio, _ = biological_design(meta, samples, drop="first")
    batch_vals = meta.loc[samples, "Batch"].to_numpy()
    batch_dum, _ = dummies(batch_vals, drop=ref_batch)
    n_batch_cols = batch_dum.shape[1]

    intercept = np.ones((n, 1))
    full = np.hstack([intercept, design_bio, batch_dum])
    n_params = full.shape[1]

    arr = X.to_numpy(dtype=float, copy=True)
    if n_batch_cols == 0:
        return pd.DataFrame(arr, index=X.index, columns=X.columns)

    for i in range(arr.shape[0]):
        y = arr[i, :]
        obs = ~np.isnan(y)
        if obs.sum() <= n_params:
            continue  # too few observations to fit; leave protein untouched
        beta = ols_beta(y[obs], full[obs, :])
        batch_beta = beta[-n_batch_cols:]
        arr[i, obs] = y[obs] - batch_dum[obs, :] @ batch_beta

    return pd.DataFrame(arr, index=X.index, columns=X.columns)


# --------------------------------------------------------------------------
# 3 & 4. ComBat (parametric empirical Bayes; mean-only variant)
# --------------------------------------------------------------------------

def combat_param(X, meta, ref_batch):
    """Parametric ComBat (Johnson, Li & Rabinovic 2007), location and scale."""
    return _combat(X, meta, ref_batch, mean_only=False)


def combat_bridge(X, meta, ref_batch):
    """ComBat mean-only variant: correct the location shift, not the scale."""
    return _combat(X, meta, ref_batch, mean_only=True)


def _combat(X, meta, ref_batch, mean_only=False):
    samples = list(X.columns)
    arr = X.to_numpy(dtype=float, copy=True)
    nan_mask = np.isnan(arr)

    _warn_confounding(meta, samples)

    # impute each protein's missing entries with that protein's row-min
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        row_min = np.nanmin(np.where(nan_mask, np.nan, arr), axis=1)
    imp = arr.copy()
    miss_rows, miss_cols = np.where(nan_mask)
    imp[miss_rows, miss_cols] = row_min[miss_rows]

    # drop zero-variance / all-missing proteins: pass them through uncorrected
    finite_row = np.all(np.isfinite(imp), axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        row_var = np.nanvar(imp, axis=1)
    keep = finite_row & (row_var > 1e-12)

    out = arr.copy()
    if keep.any():
        batch_vals = meta.loc[samples, "Batch"].to_numpy()
        batch_levels = sorted(int(b) for b in pd.unique(batch_vals))
        n_batch = len(batch_levels)
        batchmod, _ = dummies(batch_vals, levels=batch_levels, drop=None)
        mod, _ = biological_design(meta, samples, drop="first")
        design = np.hstack([batchmod, mod])
        batch_idx = [np.where(batch_vals == b)[0] for b in batch_levels]
        ref_idx = batch_levels.index(int(ref_batch)) if int(ref_batch) in batch_levels else None

        corrected = _combat_core(
            imp[keep, :], design, n_batch, batch_idx, ref_idx, mean_only
        )
        out[keep, :] = corrected

    out[nan_mask] = np.nan  # restore original missingness
    return pd.DataFrame(out, index=X.index, columns=X.columns)


def _combat_core(dat, design, n_batch, batch_idx, ref_idx, mean_only):
    """Standardize, empirical-Bayes shrink batch parameters, adjust, unstandardize."""
    n_samples = dat.shape[1]
    batch_sizes = np.array([len(ix) for ix in batch_idx], dtype=float)

    # standardize across features
    B_hat = np.linalg.pinv(design.T @ design) @ design.T @ dat.T   # params x genes
    if ref_idx is not None:
        grand_mean = B_hat[ref_idx, :]
    else:
        grand_mean = (batch_sizes / n_samples) @ B_hat[:n_batch, :]

    if ref_idx is not None:
        ref_ix = batch_idx[ref_idx]
        fitted = (design[ref_ix, :] @ B_hat).T
        resid = dat[:, ref_ix] - fitted
        var_pooled = np.sum(resid ** 2, axis=1) / len(ref_ix)
    else:
        fitted = (design @ B_hat).T
        resid = dat - fitted
        var_pooled = np.sum(resid ** 2, axis=1) / n_samples
    var_pooled = np.where(var_pooled <= 0, np.nanmean(var_pooled[var_pooled > 0]) if np.any(var_pooled > 0) else 1.0, var_pooled)
    sqrt_var = np.sqrt(var_pooled)[:, None]

    # stand.mean carries grand mean + biological covariate effects, no batch
    tmp = design.copy()
    tmp[:, :n_batch] = 0.0
    stand_mean = grand_mean[:, None] + (tmp @ B_hat).T
    s_data = (dat - stand_mean) / sqrt_var

    # batch effect estimates on standardized data
    batch_design = design[:, :n_batch]
    gamma_hat = np.linalg.pinv(batch_design.T @ batch_design) @ batch_design.T @ s_data.T
    delta_hat = np.ones((n_batch, dat.shape[0]))
    if not mean_only:
        for i, ix in enumerate(batch_idx):
            delta_hat[i, :] = np.var(s_data[:, ix], axis=1, ddof=1)

    gamma_bar = np.mean(gamma_hat, axis=1)
    t2 = np.var(gamma_hat, axis=1, ddof=1)

    gamma_star = np.zeros_like(gamma_hat)
    delta_star = np.ones_like(delta_hat)
    for i, ix in enumerate(batch_idx):
        if mean_only:
            gamma_star[i, :] = _postmean(gamma_hat[i, :], gamma_bar[i], 1.0, 1.0, t2[i])
            delta_star[i, :] = 1.0
        else:
            a = _aprior(delta_hat[i, :])
            b = _bprior(delta_hat[i, :])
            g, d = _it_sol(s_data[:, ix], gamma_hat[i, :], delta_hat[i, :],
                           gamma_bar[i], t2[i], a, b)
            gamma_star[i, :] = g
            delta_star[i, :] = d

    if ref_idx is not None:
        gamma_star[ref_idx, :] = 0.0
        delta_star[ref_idx, :] = 1.0

    # adjust the data and reverse the standardization
    bayes = s_data.copy()
    for i, ix in enumerate(batch_idx):
        bayes[:, ix] = (bayes[:, ix] - gamma_star[i, :][:, None]) / np.sqrt(delta_star[i, :])[:, None]
    return bayes * sqrt_var + stand_mean


def _aprior(delta_hat):
    m = np.mean(delta_hat)
    s2 = np.var(delta_hat, ddof=1)
    s2 = s2 if s2 > 0 else 1e-12
    return (2.0 * s2 + m ** 2) / s2


def _bprior(delta_hat):
    m = np.mean(delta_hat)
    s2 = np.var(delta_hat, ddof=1)
    s2 = s2 if s2 > 0 else 1e-12
    return (m * s2 + m ** 3) / s2


def _postmean(g_hat, g_bar, n, d_star, t2):
    return (t2 * n * g_hat + d_star * g_bar) / (t2 * n + d_star)


def _postvar(sum2, n, a, b):
    return (0.5 * sum2 + b) / (n / 2.0 + a - 1.0)


def _it_sol(sdat, g_hat, d_hat, g_bar, t2, a, b, tol=1e-4, max_iter=100):
    """Fixed-point solve for the EB posterior gamma* and delta* (one batch)."""
    n = sdat.shape[1]
    g_old = g_hat.copy()
    d_old = d_hat.copy()
    change = np.inf
    it = 0
    while change > tol and it < max_iter:
        g_new = _postmean(g_hat, g_bar, n, d_old, t2)
        sum2 = np.sum((sdat - g_new[:, None]) ** 2, axis=1)
        d_new = _postvar(sum2, n, a, b)
        change = max(
            np.max(np.abs(g_new - g_old) / (np.abs(g_old) + 1e-9)),
            np.max(np.abs(d_new - d_old) / (np.abs(d_old) + 1e-9)),
        )
        g_old, d_old = g_new, d_new
        it += 1
    return g_old, d_old


def _warn_confounding(meta, samples):
    """Warn if any group sits in only one batch (EB may absorb that biology)."""
    sub = meta.loc[samples]
    for group, grp in sub.groupby("Group"):
        if grp["Batch"].nunique() <= 1 and sub["Batch"].nunique() > 1:
            warnings.warn(
                f"ComBat: group '{group}' appears in only one batch; empirical-Bayes "
                "shrinkage may absorb confounded biology",
                stacklevel=3,
            )


# --------------------------------------------------------------------------
# 5. median_polish
# --------------------------------------------------------------------------

def median_polish(X, meta, ref_batch, max_iter=10, tol=1e-4):
    """Subtract per-batch effects from a Tukey median polish.

    Build a protein x batch table of per-protein per-batch medians, median-
    polish it into overall + protein effects + batch effects + residual, then
    subtract each batch's fitted effect from that batch's samples. Like
    bridge_median this is a per-batch scalar shift, not a per-protein fix.
    """
    samples = list(X.columns)
    batch_vals = meta.loc[samples, "Batch"].to_numpy()
    batch_levels = sorted(int(b) for b in pd.unique(batch_vals))

    arr = X.to_numpy(dtype=float, copy=True)
    table = np.full((arr.shape[0], len(batch_levels)), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for j, b in enumerate(batch_levels):
            cols = np.where(batch_vals == b)[0]
            table[:, j] = np.nanmedian(arr[:, cols], axis=1)

    _overall, _row_eff, col_eff, _resid = _median_polish(table, max_iter, tol)

    for j, b in enumerate(batch_levels):
        cols = np.where(batch_vals == b)[0]
        arr[:, cols] -= col_eff[j]

    return pd.DataFrame(arr, index=X.index, columns=X.columns)


def _median_polish(M, max_iter=10, tol=1e-4):
    resid = np.array(M, dtype=float, copy=True)
    n_row, n_col = resid.shape
    overall = 0.0
    row = np.zeros(n_row)
    col = np.zeros(n_col)
    prev = np.inf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for _ in range(max_iter):
            rm = np.nan_to_num(np.nanmedian(resid, axis=1))
            resid = resid - rm[:, None]
            row = row + rm
            delta = np.nan_to_num(np.nanmedian(row))
            row -= delta
            overall += delta

            cm = np.nan_to_num(np.nanmedian(resid, axis=0))
            resid = resid - cm[None, :]
            col = col + cm
            delta = np.nan_to_num(np.nanmedian(col))
            col -= delta
            overall += delta

            cur = np.nansum(np.abs(resid))
            if abs(prev - cur) <= tol * (abs(prev) + 1e-9):
                break
            prev = cur

    return overall, row, col, resid


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

METHOD_REGISTRY = {
    "bridge_median": bridge_median,
    "limma_rbe": limma_rbe,
    "combat_param": combat_param,
    "combat_bridge": combat_bridge,
    "median_polish": median_polish,
}

METHOD_LABELS = {
    "bridge_median": "Bridge-anchored global median offset",
    "limma_rbe": "Per-protein design-aware batch removal (removeBatchEffect)",
    "combat_param": "Parametric ComBat (empirical Bayes)",
    "combat_bridge": "ComBat, mean-only location variant",
    "median_polish": "Tukey median-polish batch effects",
}
