"""Synthetic multi-batch proteomics data with planted, known truth.

The generator plants exactly the structure the harness is meant to recover, so
the demo can check correctness rather than just run:

  - a per-protein additive batch effect plus a per-protein per-batch scale
    wobble, so a single global offset cannot fix it and the per-protein methods
    should win;
  - a biological group effect on a slice of proteins, the real signal a
    correction must keep;
  - bridge subjects replicated across batches with identical underlying
    biology, so their replicate spread is pure batch effect (the ground truth
    for bridge_mad);
  - left-censored plus random missingness.

Everything is driven by an explicit seed argument. Nothing is drawn at import.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate(
    seed,
    n_proteins=1200,
    n_batches=4,
    groups=("A", "B"),
    n_strata=2,
    n_bridge=4,
    n_regular_per_batch=10,
    batch_shift_sd=0.5,
    batch_scale_sd=0.05,
    group_effect_sd=1.2,
    signal_fraction=0.12,
    subject_sd=0.3,
    noise_sd=0.15,
    confounding=0.15,
    censor_threshold=12.5,
    censor_scale=1.0,
    random_dropout=0.01,
):
    """Build (abundance, samples) frames with a reproducible planted truth.

    abundance is proteins x samples (log2, NaN = not detected); samples has
    Sample, Batch, Group, Subject, Stratum. Reference batch per stratum is
    batch 1: its additive batch effect is zero and its scale wobble is one, so a
    perfect correction recovers batch-1 biology.
    """
    rng = np.random.default_rng(seed)
    groups = list(groups)

    proteins = [f"PROT{p:05d}" for p in range(n_proteins)]
    base = rng.normal(16.0, 2.5, n_proteins)          # per-protein baseline log2

    # biological group effect on a slice of proteins (applied to non-reference group)
    n_signal = int(round(signal_fraction * n_proteins))
    signal_idx = rng.choice(n_proteins, size=n_signal, replace=False)
    group_effect = np.zeros(n_proteins)
    group_effect[signal_idx] = rng.normal(0.0, group_effect_sd, n_signal)

    # per-protein batch effects; batch index 0 (batch id 1) is the anchor
    batch_shift = np.zeros((n_proteins, n_batches))
    batch_scale = np.ones((n_proteins, n_batches))
    for b in range(1, n_batches):
        batch_shift[:, b] = rng.normal(0.0, batch_shift_sd, n_proteins)
        batch_scale[:, b] = np.clip(rng.normal(1.0, batch_scale_sd, n_proteins), 0.7, 1.3)

    ref_group = groups[0]
    columns = {}
    meta_rows = []
    sample_counter = 0

    for s in range(n_strata):
        stratum = f"region{s + 1}"

        # bridge subjects: present in every batch, identical biology across them
        for k in range(n_bridge):
            subj = f"S{s + 1}B{k:02d}"
            group = groups[k % len(groups)]
            subject_bio = rng.normal(0.0, subject_sd, n_proteins)
            for b in range(n_batches):
                sample_counter += 1
                name = f"Sample_{sample_counter:04d}"
                columns[name] = _measure(
                    rng, base, group_effect, subject_bio, group, ref_group,
                    batch_shift[:, b], batch_scale[:, b], noise_sd,
                    censor_threshold, censor_scale, random_dropout,
                )
                meta_rows.append((name, b + 1, group, subj, stratum))

        # regular subjects: one appearance each, in a single batch
        for b in range(n_batches):
            labels = _assign_groups(rng, n_regular_per_batch, b, groups, confounding)
            for j, group in enumerate(labels):
                subj = f"S{s + 1}R{b:02d}{j:02d}"
                subject_bio = rng.normal(0.0, subject_sd, n_proteins)
                sample_counter += 1
                name = f"Sample_{sample_counter:04d}"
                columns[name] = _measure(
                    rng, base, group_effect, subject_bio, group, ref_group,
                    batch_shift[:, b], batch_scale[:, b], noise_sd,
                    censor_threshold, censor_scale, random_dropout,
                )
                meta_rows.append((name, b + 1, group, subj, stratum))

    abundance = pd.DataFrame(columns, index=pd.Index(proteins, name="Protein"))
    samples = pd.DataFrame(
        meta_rows, columns=["Sample", "Batch", "Group", "Subject", "Stratum"]
    ).set_index("Sample")
    return abundance, samples


def _measure(rng, base, group_effect, subject_bio, group, ref_group,
             shift, scale, noise_sd, censor_threshold, censor_scale, random_dropout):
    """One measured sample column: biology, then batch distortion, then dropout."""
    bio = base + subject_bio
    if group != ref_group:
        bio = bio + group_effect
    # deviations from baseline get scaled and shifted by the batch, then noise
    observed = base + shift + scale * (bio - base)
    observed = observed + rng.normal(0.0, noise_sd, base.shape[0])

    # left-censored detection: low abundance drops out more; plus a little random
    p_detect = _sigmoid((observed - censor_threshold) / censor_scale)
    detected = (rng.random(base.shape[0]) < p_detect) & (rng.random(base.shape[0]) > random_dropout)
    observed = np.where(detected, observed, np.nan)
    return observed


def _assign_groups(rng, n, batch_index, groups, confounding):
    """Group labels for a batch's regular subjects, with tunable confounding.

    confounding = 0 draws balanced labels; 1 pushes each batch toward one group.
    A guard keeps at least two of each group per batch so effects stay
    estimable.
    """
    if len(groups) != 2:
        return list(rng.choice(groups, size=n))
    lean = 1.0 if batch_index % 2 == 0 else -1.0
    p_first = float(np.clip(0.5 + 0.4 * confounding * lean, 0.02, 0.98))
    draws = rng.random(n)
    labels = [groups[0] if d < p_first else groups[1] for d in draws]

    for g_idx, other in ((0, 1), (1, 0)):
        while labels.count(groups[g_idx]) < 2:
            for i, lab in enumerate(labels):
                if lab == groups[other] and labels.count(groups[other]) > 2:
                    labels[i] = groups[g_idx]
                    break
            else:
                break
    return labels


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def write(outdir, seed, **kwargs):
    """Generate and write abundance.csv + samples.csv into ``outdir``."""
    import os

    os.makedirs(outdir, exist_ok=True)
    abundance, samples = generate(seed, **kwargs)
    abundance_path = os.path.join(outdir, "abundance.csv")
    samples_path = os.path.join(outdir, "samples.csv")
    abundance.to_csv(abundance_path)
    samples.to_csv(samples_path)
    return abundance_path, samples_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Write a synthetic dataset.")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    a, s = write(args.outdir, args.seed)
    print(f"wrote {a}\nwrote {s}")
