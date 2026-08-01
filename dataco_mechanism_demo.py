"""
dataco_mechanism_demo.py
========================
A torch-free demonstration of the CAG mechanism on REAL DataCo supply-chain data.

SDV (CTGAN/TVAE/CopulaGAN) needs PyTorch, which will not fit in this sandbox, so
this uses a lightweight Gaussian-copula synthesizer that generates genuinely NEW
rows (not bootstrap copies), then applies the constraint ladder via tiered
rejection and measures the four pillars + membership-inference at each strictness
level using the pipeline's OWN metric functions.

This is a MECHANISM test, not a replacement for the SDV results: it asks whether,
on real supply-chain data, tightening constraints moves privacy and utility in the
predicted direction (DCR down, utility up, MIA -> 0.5). Reproduce with full SDV on
a machine with torch/GPU using privacy_utility_sweep.py.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from pipeline.metrics import compute_dcr_nnaa, compute_fidelity
from pipeline.constraints import integrity_check_synthetic
from pipeline.evaluation import trtr_baseline, tstr_evaluate
from pipeline.data import FEATURE_COLS, _derive_delay_labels_from_sla_buffer
from dataco_adapter import load_dataco, dataco_to_canonical
from privacy_utility_sweep import membership_inference_auc

RNG = np.random.RandomState(42)
NUM_COLS = ["transit_duration_days", "promised_transit_days", "shipment_buffer_days", "sla_buffer"]
CAT_COLS = ["dpe_id", "item_id", "line_type", "dpe_shipnode", "last_shipnode",
            "last_scac", "last_shipment_carrier_service", "carrier_service_code", "order_type"]
# Leakage-safe: exclude transit_duration_days (the label source) from predictors.
LEAKAGE_SAFE_FEATURES = [c for c in FEATURE_COLS if c != "transit_duration_days"]


# --------------------------- copula synthesizer --------------------------- #
def fit_sample_copula(train, n):
    """Gaussian-copula on numerics (fresh rows) + independent categorical marginals."""
    from scipy.stats import norm, rankdata
    # use only numeric columns with non-zero variance; constants are filled back after
    usable = [c for c in NUM_COLS if pd.to_numeric(train[c], errors="coerce").std(skipna=True) > 1e-9]
    consts = {c: float(pd.to_numeric(train[c], errors="coerce").fillna(0.0).mean())
              for c in NUM_COLS if c not in usable}
    num = train[usable].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
    U = np.column_stack([rankdata(num[:, j]) / (len(num) + 1) for j in range(num.shape[1])])
    Z = norm.ppf(np.clip(U, 1e-4, 1 - 1e-4))
    cov = np.corrcoef(Z, rowvar=False)
    cov = np.atleast_2d(cov)
    cov += np.eye(cov.shape[0]) * 1e-6  # ridge for numerical stability
    Zs = RNG.multivariate_normal(np.zeros(len(usable)), cov, size=n)
    Us = norm.cdf(Zs)
    synth_num = np.column_stack([np.quantile(num[:, j], np.clip(Us[:, j], 0, 1)) for j in range(len(usable))])
    out = pd.DataFrame(synth_num, columns=usable)
    for c, val in consts.items():
        out[c] = val
    # categoricals: independent draws from empirical marginals (no row copying)
    for c in CAT_COLS:
        if c in train.columns:
            vc = train[c].astype(str).value_counts(normalize=True)
            out[c] = RNG.choice(vc.index.to_numpy(), size=n, p=vc.to_numpy())
    # label: independent marginal draw -> creates some SLA-inconsistency for rejection to remove
    lab = train["delay_label"].value_counts(normalize=True).sort_index()
    out["delay_label"] = RNG.choice(lab.index.to_numpy(), size=n, p=lab.to_numpy())
    out["distance_km"] = 0.0
    return out


# --------------------------- tiered rejection ----------------------------- #
def reject(df, level):
    """Cumulative rejection tiers on the synthesized space."""
    if level == "none":
        return df
    m = pd.Series(True, index=df.index)
    if level in ("moderate", "strict"):                      # R2 + R5
        m &= pd.to_numeric(df["transit_duration_days"], errors="coerce") > 0
        m &= pd.to_numeric(df["promised_transit_days"], errors="coerce") >= 0
        m &= pd.to_numeric(df["delay_label"], errors="coerce").isin([0, 1, 2])
    if level == "strict":                                     # + R3b label-SLA consistency
        derived = _derive_delay_labels_from_sla_buffer(df["sla_buffer"])
        m &= pd.to_numeric(df["delay_label"], errors="coerce") == derived
    return df[m.fillna(False)].copy()


def evaluate(members, non_members, synth):
    trtr = trtr_baseline(members, feature_cols=LEAKAGE_SAFE_FEATURES)
    tstr = tstr_evaluate(members, synth, trtr_f1=trtr["weighted_f1_mean"],
                         trtr_bus=trtr["business_utility_mean"],
                         trtr_l2_recall=trtr["label2_recall_mean"],
                         feature_cols=LEAKAGE_SAFE_FEATURES)
    priv = compute_dcr_nnaa(members, synth)
    fid = compute_fidelity(members, synth)
    mia = membership_inference_auc(members, non_members, synth, feature_cols=NUM_COLS)
    return {"kept": len(synth), "dcr_mean": priv["dcr_mean"], "dcr_p5": priv["dcr_p5"],
            "nnaa": priv["nnaa"], "mia_auc": mia["mia_auc"], "business_utility": tstr["business_utility"],
            "f1_gap": tstr["f1_gap_vs_trtr"], "ks_mean": fid["ks_mean"],
            "cvr": integrity_check_synthetic(synth)}


def main():
    canon = dataco_to_canonical(load_dataco())
    # subsample for speed (torch-free, but keep it brisk)
    canon = canon.sample(n=min(20000, len(canon)), random_state=42).reset_index(drop=True)
    # member / non-member split (stratified)
    members = canon.groupby("delay_label", group_keys=False).apply(lambda g: g.sample(frac=0.5, random_state=42))
    non_members = canon.drop(members.index).reset_index(drop=True)
    members = members.reset_index(drop=True)
    print(f"DataCo canonical: {len(canon):,} rows  | members {len(members):,} / non-members {len(non_members):,}")
    print(f"TRTR ceiling (leakage-safe): computing...")

    N = 8000
    rows = []
    for level in ["none", "moderate", "strict"]:
        # over-generate then reject down (resample-to-N keeps size comparable)
        cand = fit_sample_copula(members, int(N * 2.5))
        kept = reject(cand, level).head(N)
        r = evaluate(members, non_members, kept)
        r["level"] = level
        rows.append(r)
        print(f"  {level:9s}: kept={r['kept']:5d}  CVR={r['cvr']:.2f}%  "
              f"DCR={r['dcr_mean']:.4f}  MIA_AUC={r['mia_auc']:.3f}  "
              f"BUS={r['business_utility']:.3f}  f1_gap={r['f1_gap']:+.3f}  KS={r['ks_mean']:.3f}")

    res = pd.DataFrame(rows)[["level", "kept", "cvr", "dcr_mean", "dcr_p5", "nnaa",
                              "mia_auc", "business_utility", "f1_gap", "ks_mean"]]
    res.to_csv("outputs/dataco_mechanism.csv", index=False)
    print("\n=== DataCo mechanism result (real public supply-chain data) ===")
    print(res.to_string(index=False))
    # directional read
    d = res.set_index("level")
    print("\nDirectional check (none -> strict):")
    print(f"  CVR:      {d.loc['none','cvr']:.2f}%  -> {d.loc['strict','cvr']:.2f}%   (expect down)")
    print(f"  DCR mean: {d.loc['none','dcr_mean']:.4f} -> {d.loc['strict','dcr_mean']:.4f}   (expect down = closer/safer-utility tradeoff)")
    print(f"  MIA AUC:  {d.loc['none','mia_auc']:.3f}  -> {d.loc['strict','mia_auc']:.3f}    (0.5 = no leakage)")
    print(f"  BUS:      {d.loc['none','business_utility']:.3f}  -> {d.loc['strict','business_utility']:.3f}    (expect up/hold)")


if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)
    main()
