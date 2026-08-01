"""
privacy_utility_sweep.py
========================
Privacy-Utility trade-off sweep under constraint strictness.

Produces the dense version of the paper's Table 4 / Figure 1: for each
(architecture x constraint-strictness-level x seed) it trains a synthesizer,
optionally applies CAG post-hoc rejection, and computes the four pillars
(privacy, utility, fidelity, integrity), then aggregates across seeds with
95% confidence intervals and runs paired significance tests between adjacent
strictness levels.

Wired against the existing pipeline package:
  - train_sdv_models(...)         -> {arch: synth_df}   (auto-mocks if SDV absent)
  - trtr_baseline / tstr_evaluate -> utility (BUS, F1 gap, L2 recall)
  - compute_dcr_nnaa              -> privacy (DCR mean/p5, NNAA)
  - compute_fidelity              -> fidelity (KS, JS)
  - integrity_check_synthetic     -> integrity (CVR %)
  - confidence_interval           -> 95% CIs

USAGE
  Smoke test (no SDV / no real data needed -- proves the flow runs):
      python privacy_utility_sweep.py --smoke

  Real run on your corpus:
      python privacy_utility_sweep.py --real path/to/data.csv \
          --seeds 5 --n-synth 10000 --epochs 100 \
          --architectures TVAE CTGAN CopulaGAN \
          --out outputs/pu_sweep

NOTE ON SEEDS: train_sdv_models does not take a seed argument, so this harness
sets the global RNG state (numpy/random/torch) before each training call. With
real SDV installed this varies generation across seeds; the bundled MOCK
synthesizer is deterministic, so under --smoke the per-seed rows are identical
by design (the smoke test validates the pipeline, not the statistics).
"""
from __future__ import annotations
import argparse
import json
import os
import random
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

# --- interfaces from the existing codebase (facade imports cleanly w/o SDV) ---
from pipeline import (
    train_sdv_models,
    trtr_baseline,
    tstr_evaluate,
    compute_dcr_nnaa,
    compute_fidelity,
    integrity_check_synthetic,
    generate_proxy_dataset,
    FEATURE_COLS,
)
from pipeline.data import confidence_interval  # not re-exported by the facade
from pipeline.metrics import PRIVACY_FIDELITY_NUM_COLS
from pipeline.constraints import cag_rejection_filter

# ----------------------------------------------------------------------------- #
# Strictness ladder. Each level maps to (SDV constraint_level, CAG rejection?).
# "none" -> unconstrained; "moderate"/"strict" -> SDV native positivity rules;
# the optional "+reject" variant applies post-hoc CAG rejection on top.
# This mirrors the paper's none / temporal-only / full ladder and is extensible.
# ----------------------------------------------------------------------------- #
STRICTNESS_LADDER = {
    "none":          {"constraint_level": "none",     "reject": False},
    "temporal":      {"constraint_level": "temporal", "reject": False},
    "moderate":      {"constraint_level": "moderate", "reject": False},
    "strict":        {"constraint_level": "strict",   "reject": False},
    "strict+reject": {"constraint_level": "strict",   "reject": True},
}
DEFAULT_LEVELS = ["none", "temporal", "moderate", "strict"]

# Metrics pulled into the sweep table (display name -> meaning)
SWEEP_METRICS = [
    "dcr_mean", "dcr_p5", "nnaa",          # privacy
    "business_utility", "f1_gap", "l2_recall", "bus_ratio",  # utility
    "ks_mean", "js_mean",                  # fidelity
    "cvr",                                 # integrity
]


def set_global_seed(seed: int) -> None:
    """Set every RNG that SDV/torch/numpy may consult before a training call."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Membership-Inference Audit (upgrades privacy from screening proxy -> audit)
#
# A proper MIA cannot be computed on a generator trained on ALL real data
# (there are no non-members left). It requires the protocol below, so it runs
# as a separate pass that re-trains on a member split.
#
#   1. split real -> members / non_members (stratified by label)
#   2. train generator on MEMBERS only -> synth
#   3. score each query record by -distance_to_nearest_synthetic
#      (closer to synthetic => more likely a training member)
#   4. AUC and TPR@low-FPR of (score, is_member)
#
# AUC ~ 0.50 => adversary cannot distinguish members => no membership leakage.
# AUC -> 1.0 => strong leakage. This is the standard shadow-free black-box
# attack used in the DCR-MIA / TAPAS line of work; a DOMIAS-style density-ratio
# variant can be slotted into membership_inference_auc() later.
# --------------------------------------------------------------------------- #
def _member_split(df, member_frac=0.5, seed=42):
    """Stratified split into (members, non_members)."""
    if "delay_label" in df.columns:
        members = df.groupby("delay_label", group_keys=False).apply(
            lambda g: g.sample(frac=member_frac, random_state=seed))
    else:
        members = df.sample(frac=member_frac, random_state=seed)
    non_members = df.drop(members.index)
    return members.reset_index(drop=True), non_members.reset_index(drop=True)


def _numeric_cols(df, feature_cols):
    cols = feature_cols or list(df.columns)
    return [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]


def membership_inference_auc(members_df, non_members_df, synth_df,
                             feature_cols=None, max_synth=5000, max_query=2000,
                             low_fpr=0.01, seed=42):
    """Distance-to-nearest-synthetic membership inference. Returns metrics dict."""
    num = _numeric_cols(members_df, feature_cols)
    num = [c for c in num if c in non_members_df.columns and c in synth_df.columns]
    if not num or len(synth_df) == 0:
        return {"mia_auc": float("nan"), "mia_tpr_at_fpr": float("nan"), "n_features": 0}

    def mat(d, scaler):
        X = d[num].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
        return scaler.transform(X)

    scaler = StandardScaler().fit(
        members_df[num].apply(pd.to_numeric, errors="coerce").fillna(0.0))

    synth = synth_df if len(synth_df) <= max_synth else synth_df.sample(max_synth, random_state=seed)
    S = mat(synth, scaler)

    k = min(len(members_df), len(non_members_df), max_query)
    if k < 5:
        return {"mia_auc": float("nan"), "mia_tpr_at_fpr": float("nan"), "n_features": len(num)}
    M = mat(members_df.sample(k, random_state=seed), scaler)
    N = mat(non_members_df.sample(k, random_state=seed), scaler)

    score_mem = -cdist(M, S).min(axis=1)   # closer to synth => higher score => more member-like
    score_non = -cdist(N, S).min(axis=1)
    scores = np.concatenate([score_mem, score_non])
    labels = np.concatenate([np.ones(k), np.zeros(k)])

    auc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.searchsorted(fpr, low_fpr, side="right") - 1
    tpr_at = float(tpr[idx]) if idx >= 0 else 0.0
    return {"mia_auc": round(float(auc), 4), "mia_tpr_at_fpr": round(tpr_at, 4),
            "n_features": len(num), "n_query_each": int(k)}


def run_mia_audit(real_df, levels, architectures, seeds, n_synth, epochs,
                  member_frac=0.5):
    """MIA audit per (architecture, level, seed) under the member-split protocol."""
    records = []
    for level in levels:
        spec = STRICTNESS_LADDER[level]
        for seed in seeds:
            set_global_seed(seed)
            members, non_members = _member_split(real_df, member_frac, seed)
            synths = train_sdv_models(
                members, epochs=epochs, n_synth=n_synth,
                constraint_level=spec["constraint_level"], architectures=architectures,
                post_filter=cag_rejection_filter if spec["reject"] else None)
            for arch, synth_df in synths.items():
                res = membership_inference_auc(members, non_members, synth_df,
                                               feature_cols=PRIVACY_FIDELITY_NUM_COLS)
                print(f"  [MIA] {arch:10s} {level:14s} seed={seed}: "
                      f"AUC={res['mia_auc']}  TPR@1%FPR={res['mia_tpr_at_fpr']}")
                for m in ("mia_auc", "mia_tpr_at_fpr"):
                    records.append({"architecture": arch, "level": level, "seed": seed,
                                    "metric": m, "value": res[m]})
    return pd.DataFrame.from_records(records)


def evaluate_synth(real_df, synth_df, trtr, feature_cols, cv_strategy="temporal") -> dict:
    """Compute the four-pillar metrics for one synthetic dataset."""
    out = {}
    # Privacy
    priv = compute_dcr_nnaa(real_df, synth_df)
    out["dcr_mean"] = priv.get("dcr_mean", np.nan)
    out["dcr_p5"] = priv.get("dcr_p5", np.nan)
    out["nnaa"] = priv.get("nnaa", np.nan)
    # Utility
    tstr = tstr_evaluate(
        real_df, synth_df,
        trtr_f1=trtr["weighted_f1_mean"],
        trtr_bus=trtr["business_utility_mean"],
        trtr_l2_recall=trtr["label2_recall_mean"],
        feature_cols=feature_cols, cv_strategy=cv_strategy,
    )
    out["business_utility"] = tstr.get("business_utility", np.nan)
    out["f1_gap"] = tstr.get("f1_gap_vs_trtr", np.nan)
    out["l2_recall"] = tstr.get("label2_recall", np.nan)
    out["bus_ratio"] = tstr.get("bus_ratio_vs_trtr", np.nan)
    # Fidelity
    fid = compute_fidelity(real_df, synth_df)
    out["ks_mean"] = fid.get("ks_mean", np.nan)
    out["js_mean"] = fid.get("js_mean", np.nan)
    # Integrity
    out["cvr"] = integrity_check_synthetic(synth_df)
    return out


def run_sweep(real_df, levels, architectures, seeds, n_synth, epochs, feature_cols):
    """Returns a long-form DataFrame: one row per (arch, level, seed, metric)."""
    # TRTR is a property of the real data + fixed-seed RF; compute once and reuse.
    print("Computing TRTR baseline (real-data ceiling)...")
    trtr = trtr_baseline(real_df, feature_cols=feature_cols)
    print(f"  TRTR F1={trtr['weighted_f1_mean']}  BUS={trtr['business_utility_mean']}  "
          f"L2recall={trtr['label2_recall_mean']}")

    records = []
    for level in levels:
        spec = STRICTNESS_LADDER[level]
        for seed in seeds:
            set_global_seed(seed)
            synths = train_sdv_models(
                real_df, epochs=epochs, n_synth=n_synth,
                constraint_level=spec["constraint_level"],
                architectures=architectures,
                post_filter=cag_rejection_filter if spec["reject"] else None,
            )
            for arch, synth_df in synths.items():
                if spec["reject"] and len(synth_df) == 0:
                    print(f"  [warn] {arch}/{level}/seed{seed}: all rows rejected")
                    continue
                metrics = evaluate_synth(real_df, synth_df, trtr, feature_cols)
                metrics["reject_kept_n"] = len(synth_df)
                ostats = synth_df.attrs.get("oversample_stats")
                if ostats:
                    metrics["reject_drawn"] = ostats["drawn"]
                    metrics["reject_survived"] = ostats["survived"]
                    metrics["reject_survival_rate"] = ostats["survival_rate"]
                    metrics["reject_cap_hit"] = int(ostats["cap_hit"])
                    metrics["label2_raw_rate"] = ostats["raw_label2_rate"]
                    metrics["label2_kept_n"] = ostats["kept_label2_n"]
                    print(f"  [reject] {arch}/{level}/seed={seed}: survival_rate="
                          f"{ostats['survival_rate']:.4f} ({ostats['survived']}/{ostats['drawn']} drawn) "
                          f"cap_hit={ostats['cap_hit']} label2_raw_rate={ostats['raw_label2_rate']:.4f} "
                          f"label2_kept_n={ostats['kept_label2_n']}")
                print(f"  {arch:10s} {level:14s} seed={seed}: "
                      f"BUS={metrics['business_utility']:.3f} "
                      f"DCR={metrics['dcr_mean']:.4f} CVR={metrics['cvr']:.2f}%")
                for m, v in metrics.items():
                    records.append({"architecture": arch, "level": level,
                                    "seed": seed, "metric": m, "value": v})
    return pd.DataFrame.from_records(records), trtr


def aggregate(long_df) -> pd.DataFrame:
    """Mean + 95% CI per (architecture, level, metric) across seeds."""
    rows = []
    for (arch, level, metric), grp in long_df.groupby(["architecture", "level", "metric"]):
        ci = confidence_interval(grp["value"].dropna().tolist())
        rows.append({"architecture": arch, "level": level, "metric": metric,
                     "mean": ci["mean"], "std": ci["std"],
                     "ci95_low": ci["ci95_low"], "ci95_high": ci["ci95_high"],
                     "n_seeds": grp["value"].notna().sum()})
    return pd.DataFrame(rows)


def significance_between_levels(long_df, metric="business_utility"):
    """Paired t-test on a metric between each pair of strictness levels (across seeds)."""
    from scipy import stats
    out = []
    levels = list(long_df["level"].unique())
    for arch in long_df["architecture"].unique():
        sub = long_df[(long_df.architecture == arch) & (long_df.metric == metric)]
        pivot = sub.pivot_table(index="seed", columns="level", values="value")
        for a, b in combinations(levels, 2):
            if a in pivot and b in pivot:
                pair = pivot[[a, b]].dropna()
                if len(pair) >= 2 and pair[a].std() + pair[b].std() > 0:
                    t, p = stats.ttest_rel(pair[a], pair[b])
                    out.append({"architecture": arch, "metric": metric,
                                "level_a": a, "level_b": b,
                                "mean_a": round(pair[a].mean(), 4),
                                "mean_b": round(pair[b].mean(), 4),
                                "t_stat": round(float(t), 3), "p_value": round(float(p), 4)})
    return pd.DataFrame(out)


def plot_frontier(summary, out_dir):
    """Privacy-utility frontier (DCR vs BUS) per architecture across strictness."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    piv = summary.pivot_table(index=["architecture", "level"], columns="metric", values="mean").reset_index()
    order = {lvl: i for i, lvl in enumerate(DEFAULT_LEVELS + ["strict+reject"])}
    piv = piv.sort_values("level", key=lambda s: s.map(order))

    fig, ax = plt.subplots(figsize=(7, 5))
    for arch in piv["architecture"].unique():
        d = piv[piv.architecture == arch]
        ax.plot(d["dcr_mean"], d["business_utility"], "-o", label=arch)
        for _, r in d.iterrows():
            ax.annotate(r["level"], (r["dcr_mean"], r["business_utility"]),
                        fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Privacy  ->  DCR mean (distance to closest real record)")
    ax.set_ylabel("Utility  ->  Business Utility Score")
    ax.set_title("Privacy-Utility frontier under constraint strictness")
    ax.legend()
    fig.tight_layout()
    p = os.path.join(out_dir, "privacy_utility_frontier.png")
    fig.savefig(p, dpi=150)
    print(f"  wrote {p}")


def plot_mia(summary, out_dir):
    """Membership-inference AUC vs strictness (0.5 = no leakage / safe)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = summary[summary.metric == "mia_auc"]
    order = {lvl: i for i, lvl in enumerate(DEFAULT_LEVELS + ["strict+reject"])}
    fig, ax = plt.subplots(figsize=(7, 5))
    for arch in sub["architecture"].unique():
        d = sub[sub.architecture == arch].sort_values("level", key=lambda s: s.map(order))
        ax.errorbar(range(len(d)), d["mean"],
                    yerr=[d["mean"] - d["ci95_low"], d["ci95_high"] - d["mean"]],
                    marker="o", capsize=3, label=arch)
        ax.set_xticks(range(len(d)))
        ax.set_xticklabels(d["level"], rotation=20)
    ax.axhline(0.5, ls="--", color="gray", lw=1, label="0.5 = no leakage")
    ax.set_ylabel("Membership-inference AUC")
    ax.set_title("Privacy audit: MIA leakage vs constraint strictness")
    ax.legend()
    fig.tight_layout()
    p = os.path.join(out_dir, "mia_vs_strictness.png")
    fig.savefig(p, dpi=150)
    print(f"  wrote {p}")


def main():
    ap = argparse.ArgumentParser(description="Privacy-utility sweep under constraint strictness")
    ap.add_argument("--smoke", action="store_true", help="proxy data + mock synth; proves the flow")
    ap.add_argument("--real", type=str, default=None, help="path to real CSV")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-synth", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--architectures", nargs="+", default=["TVAE", "CopulaGAN", "CTGAN"])
    ap.add_argument("--levels", nargs="+", default=DEFAULT_LEVELS)
    ap.add_argument("--mia", action="store_true",
                    help="also run the membership-inference audit (member-split, doubles training)")
    ap.add_argument("--out", type=str, default="outputs/pu_sweep")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.smoke:
        print("[SMOKE] proxy dataset + mock synthesizer (deterministic).")
        real_df = generate_proxy_dataset(n=3000, seed=42)
        args.seeds = min(args.seeds, 2)
        args.n_synth = min(args.n_synth, 1500)
    elif args.real:
        from pipeline import load_real_dataset
        real_df = load_real_dataset(args.real)
    else:
        raise SystemExit("Provide --smoke or --real PATH")

    # Include _enc columns even though they don't exist pre-encoding: get_XY
    # calls encode_features() internally, so _enc columns are always available
    # in the encoded dataframe. Filtering by real_df.columns alone silently
    # drops every encoded categorical, leaving only distance_km and
    # promised_transit_days -- a one-or-two-feature path that collapses to a
    # constant on DataCo (distance_km = 0 always) and under-reports utility.
    # Finding E fix: mirror the same guard used in pipeline.py:522.
    feature_cols = [c for c in FEATURE_COLS if c in real_df.columns or c.endswith("_enc")]
    seeds = list(range(42, 42 + args.seeds))

    long_df, trtr = run_sweep(real_df, args.levels, args.architectures, seeds,
                              args.n_synth, args.epochs, feature_cols)

    if args.mia:
        print("\n[MIA] membership-inference audit (member-split protocol)...")
        mia_df = run_mia_audit(real_df, args.levels, args.architectures, seeds,
                               args.n_synth, args.epochs)
        long_df = pd.concat([long_df, mia_df], ignore_index=True)

    summary = aggregate(long_df)
    sig_bus = significance_between_levels(long_df, "business_utility")
    sig_dcr = significance_between_levels(long_df, "dcr_mean")
    sig_tables = [sig_bus, sig_dcr]
    if args.mia and "mia_auc" in long_df["metric"].values:
        # A null here (p >= 0.05, means ~0.50 throughout) is itself the rigorous way to
        # report "the attack found no membership signal at any strictness level" --
        # report it rather than omit it.
        sig_tables.append(significance_between_levels(long_df, "mia_auc"))

    long_df.to_csv(os.path.join(args.out, "sweep_long.csv"), index=False)
    summary.to_csv(os.path.join(args.out, "sweep_summary.csv"), index=False)
    pd.concat(sig_tables).to_csv(os.path.join(args.out, "sweep_significance.csv"), index=False)
    with open(os.path.join(args.out, "trtr_baseline.json"), "w") as f:
        json.dump({k: (v if not isinstance(v, list) else v) for k, v in trtr.items()
                   if k != "fold_f1s"}, f, indent=2)
    plot_frontier(summary, args.out)
    if args.mia and "mia_auc" in long_df["metric"].values:
        plot_mia(summary, args.out)

    print("\n=== SWEEP SUMMARY (mean over seeds) ===")
    metrics_to_show = ["dcr_mean", "nnaa", "business_utility", "f1_gap", "cvr"]
    if args.mia:
        metrics_to_show += ["mia_auc", "mia_tpr_at_fpr"]
    show = summary[summary.metric.isin(metrics_to_show)]
    print(show.pivot_table(index=["architecture", "level"], columns="metric", values="mean").round(4))
    print(f"\nOutputs written to: {args.out}/")


if __name__ == "__main__":
    main()
