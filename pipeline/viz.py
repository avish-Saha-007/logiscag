"""
Visualization utilities for the Viva pipeline.

This module provides functions to generate plots for evaluating the quality of synthetic data
produced by different generative models. It includes functions for plotting marginal distributions,
fidelity metrics, and other relevant visualizations to compare real and synthetic datasets.
"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from .data import FEATURE_COLS, encode_features, get_XY
from .metrics import ks_complement, wasserstein_metric, js_divergence_numeric


def plot_results(real_df, synths, tl_results, trtr, tstr_all, smote_res, priv_tradeoff, fidelity_all=None, privacy_all=None, feat_imp=None, selected_feature_cols=None, output_dir="viva_plots"):
    """Generates a suite of plots to evaluate synthetic data quality and model performance.

	- **Functional**: Produces all visual evidence required for viva storytelling and diagnostics.
	- **Technical**: Generates and saves multi-figure PNG suite (marginals, fidelity bars, transfer curve, PCA overlap, confusion matrices, privacy-distance histograms, feature importance), with deterministic styling and robust column-availability handling.
        
    - **Functional**: Creates visualizations for marginal distributions, fidelity metrics, transfer learning curves, PCA overlap, feature importance, confusion matrices, and privacy distance distributions.
    - **Technical**: Uses Matplotlib and Seaborn to create and save plots comparing real vs synthetic data across multiple dimensions and metrics, with configurable output directory and optional inclusion of fidelity/ privacy analyses.
    Args:
        real_df (pd.DataFrame): The original real dataset.
        synths (dict): A dictionary of synthetic datasets keyed by model name.
        tl_results (dict): Transfer learning results for different real data fractions.
        trtr (dict): TRTR baseline results.
        tstr_all (dict): TSTR evaluation results for all synthetic models.
        smote_res (dict): SMOTE baseline results.
        priv_tradeoff (dict): Privacy-utility tradeoff results.
        fidelity_all (dict, optional): Fidelity metrics for all models. Defaults to None.
        privacy_all (dict, optional): Privacy metrics for all models. Defaults to None.
        feat_imp (dict, optional): Feature importance scores. Defaults to None.
        selected_feature_cols (list, optional): List of feature columns used for modeling. Defaults to None.
        output_dir (str, optional): Directory to save the generated plots. Defaults to "viva_plots".
    Output:
        Saves plots to the specified output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {"TVAE": "#1f77b4", "CopulaGAN": "#2ca02c", "CTGAN": "#d62728", "SMOTE": "#ff7f0e", "TRTR": "#9467bd"}

    primary_synth = synths.get("TVAE", list(synths.values())[0])

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle("Marginal Distributions: Real vs Synthetic", fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        0.01,
        "Interpretation: KS higher is better; Wasserstein (W) and JS lower are better.",
        ha="center",
        fontsize=10,
        color="#333333",
    )
    cols = ["transit_duration", "sla_buffer", "distance_km", "carrier_pickup_delay_days", "fc_cutoff_hour", "carrier_delay_days"]
    for ax, col in zip(axes.flatten(), cols):
        if col not in real_df.columns:
            continue
        real_vals = real_df[col].dropna()
        synth_vals = primary_synth[col].dropna() if col in primary_synth.columns else pd.Series(dtype=float)
        ax.hist(real_vals, bins=40, alpha=0.6, color="#9467bd", label="Real", density=True)
        ax.hist(synth_vals, bins=40, alpha=0.6, color="#1f77b4", label="TVAE Synth", density=True)
        ks_val = ks_complement(real_vals, synth_vals)
        w_val = wasserstein_metric(real_vals, synth_vals)
        js_val = js_divergence_numeric(real_vals, synth_vals)
        ax.set_title(f"{col}\nKS={ks_val:.3f} | W={w_val:.3f} | JS={js_val:.3f}", fontsize=9)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_marginal_distributions.png"), dpi=150, bbox_inches="tight")
    plt.close()

    if fidelity_all:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        archs = [a for a in ["TVAE", "CopulaGAN", "CTGAN"] if a in fidelity_all]

        ks_vals = [fidelity_all[a].get("ks_mean", 0) for a in archs]
        wass_vals = [fidelity_all[a].get("wasserstein_mean", 0) for a in archs]
        js_vals = [fidelity_all[a].get("js_mean", 0) for a in archs]

        axes[0].bar(archs, ks_vals, color=[colors.get(a, "#7f7f7f") for a in archs], alpha=0.85)
        axes[0].set_title("KS Complement (higher better)")
        axes[0].set_ylim(0, 1)

        axes[1].bar(archs, wass_vals, color=[colors.get(a, "#7f7f7f") for a in archs], alpha=0.85)
        axes[1].set_title("Wasserstein Distance (lower better)")

        axes[2].bar(archs, js_vals, color=[colors.get(a, "#7f7f7f") for a in archs], alpha=0.85)
        axes[2].set_title("JS Divergence (lower better)")

        fig.suptitle("Fidelity Metrics by Architecture", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "02_fidelity_comparison.png"), dpi=150, bbox_inches="tight")
        plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    sp = tl_results["synth_pretrain"]
    ro = tl_results["real_only"]
    fracs_sp = sorted(sp.keys())
    f1s_sp = [sp[f]["weighted_f1"] for f in fracs_sp]
    ax.plot([f * 100 for f in fracs_sp], f1s_sp, "o-", color=colors["TVAE"], linewidth=2.5, markersize=8, label="Synth pretrain")
    fracs_ro = sorted(ro.keys())
    f1s_ro = [ro[f]["weighted_f1"] for f in fracs_ro]
    ax.plot([f * 100 for f in fracs_ro], f1s_ro, "s--", color=colors["SMOTE"], linewidth=2, markersize=7, label="Real-only")
    ax.axhline(trtr["weighted_f1_mean"], color=colors["TRTR"], linestyle="--", linewidth=2, label="TRTR")
    ax.set_xlabel("Real Data Fraction (%)")
    ax.set_ylabel("Weighted F1")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_transfer_learning_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()

    real_enc = encode_features(real_df)
    synth_enc = encode_features(primary_synth)
    pca_cols = [c for c in FEATURE_COLS if c in real_enc.columns and c in synth_enc.columns]
    scaler = StandardScaler()
    real_s = scaler.fit_transform(real_enc[pca_cols].fillna(0))
    synth_s = scaler.transform(synth_enc[pca_cols].fillna(0))

    n_pca = min(500, len(real_s), len(synth_s))
    pca = PCA(n_components=2, random_state=42)
    combined = pca.fit_transform(np.vstack([real_s[:n_pca], synth_s[:n_pca]]))
    real_pca = combined[:n_pca]
    synth_pca = combined[n_pca:]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.3, s=12, color=colors["TRTR"], label="Real")
    ax.scatter(synth_pca[:, 0], synth_pca[:, 1], alpha=0.3, s=12, color=colors["TVAE"], label="TVAE")
    ax.legend()
    ax.set_title("PCA overlap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "05_pca_overlap.png"), dpi=150, bbox_inches="tight")
    plt.close()

    if feat_imp:
        features = list(feat_imp.keys())
        scores = list(feat_imp.values())
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(features[::-1], scores[::-1], color="#1f77b4", alpha=0.85)
        for bar, score in zip(bars, scores[::-1]):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2, f"{score:.4f}", va="center")
        ax.set_title("Feature Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "07_feature_importance.png"), dpi=150, bbox_inches="tight")
        plt.close()

    X_real, y_real = get_XY(real_df, feature_cols=selected_feature_cols)
    X_tr, X_te, y_tr, y_te = train_test_split(X_real, y_real, test_size=0.2, stratify=y_real, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    clf.fit(X_tr, y_tr)
    trtr_cm = confusion_matrix(y_te, clf.predict(X_te), labels=[0, 1, 2])

    fig, axes = plt.subplots(1, len(tstr_all) + 1, figsize=(5 * (len(tstr_all) + 1), 4))
    if not hasattr(axes, "__len__"):
        axes = [axes]
    for ax, (title, cm) in zip(axes, [("TRTR", trtr_cm)] + [(f"TSTR-{a}", r["confusion_matrix"]) for a, r in tstr_all.items()]):
        sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues", cbar=False)
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_confusion_matrices.png"), dpi=150, bbox_inches="tight")
    plt.close()

    if privacy_all:
        archs = [a for a in ["TVAE", "CopulaGAN", "CTGAN"] if a in privacy_all]
        fig, axes = plt.subplots(1, len(archs), figsize=(5 * len(archs), 4), squeeze=False)
        axes = axes[0]
        for ax, arch in zip(axes, archs):
            prv = privacy_all[arch]
            d_real = np.asarray(prv.get("nn_to_real_values_sample", []), dtype=float)
            d_synth = np.asarray(prv.get("nn_to_synth_values_sample", []), dtype=float)

            if d_real.size > 0:
                ax.hist(d_real, bins=30, alpha=0.55, density=True, label="NN distance to real")
            if d_synth.size > 0:
                ax.hist(d_synth, bins=30, alpha=0.55, density=True, label="NN distance to synth")

            ax.set_title(
                f"{arch}\nNNAA={prv.get('nnaa', 0):.3f} | DCR p5={prv.get('dcr_p5', 0):.3f}",
                fontsize=10,
            )
            ax.set_xlabel("Nearest-neighbor distance")
            ax.set_ylabel("Density")
            ax.legend(fontsize=8)

        fig.suptitle("Privacy Distance Distributions by Architecture", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "08_privacy_distance_distributions.png"), dpi=150, bbox_inches="tight")
        plt.close()
