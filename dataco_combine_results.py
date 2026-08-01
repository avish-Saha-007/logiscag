"""
dataco_combine_results.py
=========================
Post-process the two separate sweep outputs (CopulaGAN full-fidelity and
TVAE/CTGAN reduced-grid) into the four canonical CSVs that §5 and NOTES.md
treat as the single source of truth for the DataCo headline:

  outputs/copulagan_final/sweep_summary_combined.csv
  outputs/copulagan_final/sweep_significance_combined.csv
  outputs/copulagan_final/strict_reject_overhead.csv
  outputs/copulagan_final/ctgan_minority_class_stability.csv

Run after BOTH sweep jobs finish:
  python dataco_combine_results.py \
      --copulagan outputs/dataco_v2_copulagan \
      --tvae_ctgan outputs/dataco_v2_tvae_ctgan \
      --out outputs/copulagan_final
"""
import argparse
import os
import json

import numpy as np
import pandas as pd
from itertools import combinations
from scipy import stats


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def load_long(out_dir):
    p = os.path.join(out_dir, "sweep_long.csv")
    return pd.read_csv(p)


def paired_significance(long_df, metric):
    out = []
    for arch in long_df["architecture"].unique():
        sub = long_df[(long_df.architecture == arch) & (long_df.metric == metric)]
        pivot = sub.pivot_table(index="seed", columns="level", values="value")
        levels = list(pivot.columns)
        for a, b in combinations(levels, 2):
            pair = pivot[[a, b]].dropna()
            if len(pair) >= 2 and pair[a].std() + pair[b].std() > 0:
                t, p = stats.ttest_rel(pair[a], pair[b])
                out.append({
                    "architecture": arch, "metric": metric,
                    "level_a": a, "level_b": b,
                    "mean_a": round(pair[a].mean(), 4),
                    "mean_b": round(pair[b].mean(), 4),
                    "t_stat": round(float(t), 3),
                    "p_value": round(float(p), 4),
                })
            else:
                out.append({
                    "architecture": arch, "metric": metric,
                    "level_a": a, "level_b": b,
                    "mean_a": round(pivot[a].mean(), 4) if a in pivot else None,
                    "mean_b": round(pivot[b].mean(), 4) if b in pivot else None,
                    "t_stat": None, "p_value": None,
                })
    return pd.DataFrame(out)


def aggregate(long_df):
    from pipeline.data import confidence_interval
    rows = []
    for (arch, level, metric), grp in long_df.groupby(["architecture", "level", "metric"]):
        ci = confidence_interval(grp["value"].dropna().tolist())
        rows.append({
            "architecture": arch, "level": level, "metric": metric,
            "mean": ci["mean"], "std": ci["std"],
            "ci95_low": ci["ci95_low"], "ci95_high": ci["ci95_high"],
            "n_seeds": grp["value"].notna().sum(),
        })
    return pd.DataFrame(rows)


def make_overhead_csv(long_df, arch="CopulaGAN"):
    """Per-seed strict+reject overhead stats for CopulaGAN."""
    sub = long_df[(long_df.architecture == arch) & (long_df.level == "strict+reject")]
    metrics = ["reject_drawn", "reject_survived", "reject_survival_rate", "reject_cap_hit",
               "business_utility", "dcr_mean"]
    pivot = sub[sub.metric.isin(metrics)].pivot_table(
        index="seed", columns="metric", values="value").reset_index()
    rename = {
        "reject_drawn": "rows_drawn",
        "reject_survived": "rows_survived",
        "reject_survival_rate": "survival_rate",
        "reject_cap_hit": "cap_hit",
    }
    pivot = pivot.rename(columns=rename)
    if "cap_hit" in pivot.columns:
        pivot["cap_hit"] = pivot["cap_hit"].apply(lambda x: bool(x) if pd.notna(x) else False)
    return pivot


def make_ctgan_minority_csv(long_df):
    """Per-seed CTGAN strict+reject minority-class stability stats."""
    sub = long_df[(long_df.architecture == "CTGAN") & (long_df.level == "strict+reject")]
    metrics = ["business_utility", "label2_raw_rate", "label2_kept_n", "reject_survival_rate", "dcr_mean"]
    pivot = sub[sub.metric.isin(metrics)].pivot_table(
        index="seed", columns="metric", values="value").reset_index()
    rename = {"reject_survival_rate": "survival_rate"}
    return pivot.rename(columns=rename)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copulagan", required=True, help="CopulaGAN sweep output dir")
    ap.add_argument("--tvae_ctgan", required=True, help="TVAE/CTGAN sweep output dir")
    ap.add_argument("--out", default="outputs/copulagan_final")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Loading CopulaGAN long CSV from {args.copulagan}...")
    cg_long = load_long(args.copulagan)
    print(f"  {len(cg_long)} rows, architectures={cg_long.architecture.unique().tolist()}")

    print(f"Loading TVAE/CTGAN long CSV from {args.tvae_ctgan}...")
    tc_long = load_long(args.tvae_ctgan)
    print(f"  {len(tc_long)} rows, architectures={tc_long.architecture.unique().tolist()}")

    combined_long = pd.concat([cg_long, tc_long], ignore_index=True)

    # ---- sweep_summary_combined.csv ----
    print("Aggregating sweep_summary_combined...")
    summary = aggregate(combined_long)
    out_summary = os.path.join(args.out, "sweep_summary_combined.csv")
    summary.to_csv(out_summary, index=False)
    print(f"  wrote {out_summary}  ({len(summary)} rows)")

    # ---- sweep_significance_combined.csv ----
    print("Computing paired significance...")
    sig_parts = []
    for metric in ("business_utility", "dcr_mean", "mia_auc"):
        if metric in combined_long["metric"].values:
            sig_parts.append(paired_significance(combined_long, metric))
    sig_df = pd.concat(sig_parts, ignore_index=True)
    out_sig = os.path.join(args.out, "sweep_significance_combined.csv")
    sig_df.to_csv(out_sig, index=False)
    print(f"  wrote {out_sig}  ({len(sig_df)} rows)")

    # ---- strict_reject_overhead.csv ----
    print("Building strict_reject_overhead.csv (CopulaGAN)...")
    overhead = make_overhead_csv(cg_long, arch="CopulaGAN")
    out_overhead = os.path.join(args.out, "strict_reject_overhead.csv")
    overhead.to_csv(out_overhead, index=False)
    print(f"  wrote {out_overhead}")
    print(overhead.to_string(index=False))

    # ---- ctgan_minority_class_stability.csv ----
    print("Building ctgan_minority_class_stability.csv...")
    ctgan_minority = make_ctgan_minority_csv(tc_long)
    out_minority = os.path.join(args.out, "ctgan_minority_class_stability.csv")
    ctgan_minority.to_csv(out_minority, index=False)
    print(f"  wrote {out_minority}")
    print(ctgan_minority.to_string(index=False))

    # ---- TRTR baselines (save the CopulaGAN one as the canonical reference) ----
    for src in (args.copulagan, args.tvae_ctgan):
        p = os.path.join(src, "trtr_baseline.json")
        if os.path.exists(p):
            with open(p) as f:
                trtr = json.load(f)
            print(f"\nTRTR from {src}:")
            for k, v in trtr.items():
                if not isinstance(v, list):
                    print(f"  {k}: {v}")
            break

    # ---- Quick headline table ----
    print("\n=== CORRECTED DataCo HEADLINE TABLE ===")
    headline_metrics = ["business_utility", "dcr_mean", "mia_auc", "cvr"]
    show = summary[summary.metric.isin(headline_metrics) & summary.architecture.isin(["CopulaGAN"])]
    if len(show) > 0:
        piv = show.pivot_table(index="level", columns="metric", values=["mean", "std"])
        print(piv.round(4))

    print(f"\nDone. All outputs in {args.out}/")


if __name__ == "__main__":
    main()
