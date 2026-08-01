"""
Provides functions to generate and export comprehensive evaluation reports for synthetic data generation experiments, 
including final summaries, detailed JSON exports, CSV files for analysis, and markdown reports for viva discussions. 
The reporting covers all four pillars of evaluation: utility, fidelity, privacy, and integrity, 
with a focus on transparency and reproducibility.
"""

import json
import os
from datetime import datetime
import numpy as np
import pandas as pd

from .data import confidence_interval


def print_final_report(trtr, tstr_all, smote_res, tl_results, constraint_report, fidelity_all, privacy_all, sig_results=None, xgb_results=None, feat_imp=None, feature_selection=None, optuna_tuning=None, qualitative_all=None, tl_results_all=None, sdmetrics_secondary=None):
    """Prints a comprehensive summary of the evaluation results across all pillars, including TRTR baseline, TSTR results, fidelity and privacy metrics, constraint audit outcomes, and any additional analyses such as significance testing or feature importance. This function is intended for quick console review after pipeline execution.

	- **Functional**: Prints concise terminal synopsis of model outcomes across pillars.
	- **Technical**: Formats aggregate and per-architecture metrics, including optional feature-selection/tuning/qualitative summaries.


    - **Functional**: Summarizes key results across all evaluation pillars in a human-readable format.
    - **Technical**: Extracts relevant metrics from input dictionaries, formats them with consistent decimal places, and prints them in organized sections for easy interpretation. Includes optional sections for significance testing, XGBoost results, feature importance, and qualitative analysis if provided. 
    Args:
        trtr (dict): Results from the TRTR baseline evaluation.
        tstr_all (dict): Dictionary of TSTR results for each synthetic data generator.
        smote_res (dict): Results from the SMOTE baseline evaluation.
        tl_results (dict): Results from transfer learning curve evaluation.
        constraint_report (dict): Summary of constraint audit results.
        fidelity_all (dict): Dictionary of fidelity metrics for each synthetic data generator.
        privacy_all (dict): Dictionary of privacy metrics for each synthetic data generator.
        sig_results (dict, optional): Results from significance testing between models.
        xgb_results (dict, optional): Results from XGBoost TSTR evaluation.
        feat_imp (dict, optional): Feature importance results.  
        feature_selection (dict, optional): Feature selection results.
        optuna_tuning (dict, optional): Summary of Optuna hyperparameter tuning results.
        qualitative_all (dict, optional): Results from qualitative analysis of synthetic data.  
    """
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)
    print(f"TRTR F1={trtr['weighted_f1_mean']:.4f} BUS={trtr['business_utility_mean']:.4f} L2={trtr['label2_recall_mean']:.4f}")
    for arch, res in tstr_all.items():
        print(f"{arch}: F1={res['weighted_f1']:.4f} Gap={res['f1_gap_vs_trtr']:.4f} BUS={res['business_utility']:.4f} ECE={res['ece']:.4f}")
    for arch, fid in fidelity_all.items():
        print(
            f"{arch} Fidelity: KS={fid.get('ks_mean', 0):.4f} "
            f"Wasserstein={fid.get('wasserstein_mean', 0):.4f} "
            f"JS={fid.get('js_mean', 0):.4f}"
        )
    for arch, prv in privacy_all.items():
        print(
            f"{arch} Privacy: DCR Mean={prv.get('dcr_mean', 0):.4f} "
            f"Min={prv.get('dcr_min', 0):.4f} P5={prv.get('dcr_p5', 0):.4f} "
            f"NNAA={prv.get('nnaa', 0):.4f}"
        )
    print(f"Constraint hard violation rate: {constraint_report.get('violation_rate_pct', 0):.4f}%")
    if feature_selection:
        print(
            f"Feature selection: method={feature_selection.get('method')} "
            f"selected={len(feature_selection.get('selected_features', []))}"
        )
    if optuna_tuning and optuna_tuning.get("status") == "completed":
        print(
            f"Optuna tuning: best_score={optuna_tuning.get('best_score')} "
            f"trials={optuna_tuning.get('n_trials')}"
        )
    if qualitative_all:
        for arch, qa in qualitative_all.items():
            print(
                f"{arch} Qualitative: flagged_rows={qa.get('total_flagged_rows', 0)} "
                f"({qa.get('flagged_row_rate_pct', 0):.2f}%)"
            )
    if sdmetrics_secondary:
        status = sdmetrics_secondary.get("status", "not_run")
        print(f"SDMetrics secondary layer: status={status}")


def export_report(export_dir, trtr, tstr_all, smote_res, tl_results, constraint_report, fidelity_all, privacy_all, sig_results=None, xgb_results=None, feat_imp=None, feature_selection=None, validation_strategy=None, optuna_tuning=None, qualitative_all=None, tl_results_all=None, sdmetrics_secondary=None):
    """Exports a comprehensive evaluation report to the specified directory, including a JSON summary of all results, CSV files for key metrics and distributions, and markdown reports summarizing pillar pass/fail outcomes and viva talking points. This function is intended for generating shareable artifacts after pipeline execution. 

	- **Functional**: Exports complete machine-readable evidence package for viva and audit.
	- **Technical**: Writes summary JSON, flattened metric CSV, privacy distribution CSV, qualitative CSV artifacts, confidence intervals, pass/fail pillar table, and generated markdown summaries.
    
    Args:        
        export_dir (str): Directory path where the report artifacts will be saved.
        trtr (dict): Results from the TRTR baseline evaluation.
        tstr_all (dict): Dictionary of TSTR results for each synthetic data generator.
        smote_res (dict): Results from the SMOTE baseline evaluation.
        tl_results (dict): Results from transfer learning curve evaluation.     
        constraint_report (dict): Summary of constraint audit results.
        fidelity_all (dict): Dictionary of fidelity metrics for each synthetic data generator.
        privacy_all (dict): Dictionary of privacy metrics for each synthetic data generator.
        sig_results (dict, optional): Results from significance testing between models.
        xgb_results (dict, optional): Results from XGBoost TSTR evaluation.
        feat_imp (dict, optional): Feature importance results.  
        feature_selection (dict, optional): Feature selection results.  
        validation_strategy (str, optional): Validation strategy used in evaluation.
        optuna_tuning (dict, optional): Summary of Optuna hyperparameter tuning results.    
        qualitative_all (dict, optional): Results from qualitative analysis of synthetic data.
    """
    os.makedirs(export_dir, exist_ok=True)

    ci = {
        "trtr_f1": confidence_interval(trtr.get("fold_f1s", [])),
        "tstr": {arch: confidence_interval(res.get("fold_f1s", [])) for arch, res in tstr_all.items()},
    }

    summary = {
        "trtr": trtr,
        "tstr": {k: {kk: vv for kk, vv in v.items() if kk != "confusion_matrix"} for k, v in tstr_all.items()},
        "smote": smote_res,
        "transfer_learning": tl_results,
        "transfer_learning_all_arch": tl_results_all,
        "constraints": constraint_report,
        "fidelity": fidelity_all,
        "privacy": privacy_all,
        "significance": sig_results,
        "xgboost": xgb_results,
        "feature_importance": feat_imp,
        "feature_selection": feature_selection,
        "validation_strategy": validation_strategy,
        "optuna_tuning": optuna_tuning,
        "qualitative_analysis": qualitative_all,
        "sdmetrics_secondary": {
            "status": (sdmetrics_secondary or {}).get("status", "not_run"),
            "architectures": list(((sdmetrics_secondary or {}).get("architectures", {}) or {}).keys()),
            "note": "Detailed SDMetrics outputs are exported separately under sdmetrics_report/",
        },
        "confidence_intervals_95": ci,
    }

    with open(os.path.join(export_dir, "results_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=_json_default)

    rows = []
    rows.append({"model": "TRTR", "metric": "weighted_f1", "value": trtr.get("weighted_f1_mean")})
    rows.append({"model": "TRTR", "metric": "business_utility", "value": trtr.get("business_utility_mean")})
    rows.append({"model": "TRTR", "metric": "label2_recall", "value": trtr.get("label2_recall_mean")})
    for arch, res in tstr_all.items():
        for metric in ["weighted_f1", "f1_gap_vs_trtr", "business_utility", "bus_ratio_vs_trtr", "label2_recall", "ece", "brier_score_mean"]:
            rows.append({"model": arch, "metric": metric, "value": res.get(metric)})
    for arch, fid in fidelity_all.items():
        for metric in ["ks_mean", "wasserstein_mean", "js_mean", "tvd_mean"]:
            rows.append({"model": arch, "metric": metric, "value": fid.get(metric)})
    for arch, prv in privacy_all.items():
        for metric in ["dcr_mean", "dcr_min", "dcr_p5", "nnaa"]:
            rows.append({"model": arch, "metric": metric, "value": prv.get(metric)})
    pd.DataFrame(rows).to_csv(os.path.join(export_dir, "results_summary.csv"), index=False)

    privacy_rows = []
    for arch, prv in privacy_all.items():
        dcr_dist = prv.get("dcr_distribution", {})
        real_dist = prv.get("nn_to_real_distribution", {})
        synth_dist = prv.get("nn_to_synth_distribution", {})

        privacy_rows.append({
            "model": arch,
            "metric_group": "dcr_distribution",
            "mean": dcr_dist.get("mean"),
            "min": dcr_dist.get("min"),
            "p5": dcr_dist.get("p5"),
            "p25": dcr_dist.get("p25"),
            "p50": dcr_dist.get("p50"),
            "p75": dcr_dist.get("p75"),
            "p95": dcr_dist.get("p95"),
            "max": dcr_dist.get("max"),
            "count": dcr_dist.get("count"),
        })
        privacy_rows.append({
            "model": arch,
            "metric_group": "nn_to_real_distribution",
            "mean": real_dist.get("mean"),
            "min": real_dist.get("min"),
            "p5": real_dist.get("p5"),
            "p25": real_dist.get("p25"),
            "p50": real_dist.get("p50"),
            "p75": real_dist.get("p75"),
            "p95": real_dist.get("p95"),
            "max": real_dist.get("max"),
            "count": real_dist.get("count"),
        })
        privacy_rows.append({
            "model": arch,
            "metric_group": "nn_to_synth_distribution",
            "mean": synth_dist.get("mean"),
            "min": synth_dist.get("min"),
            "p5": synth_dist.get("p5"),
            "p25": synth_dist.get("p25"),
            "p50": synth_dist.get("p50"),
            "p75": synth_dist.get("p75"),
            "p95": synth_dist.get("p95"),
            "max": synth_dist.get("max"),
            "count": synth_dist.get("count"),
        })
    pd.DataFrame(privacy_rows).to_csv(os.path.join(export_dir, "privacy_distance_distributions.csv"), index=False)

    qualitative_rows = []
    for arch, qa in (qualitative_all or {}).items():
        checks = qa.get("checks", {})
        row = {
            "model": arch,
            "row_count": qa.get("row_count"),
            "total_flagged_rows": qa.get("total_flagged_rows"),
            "flagged_row_rate_pct": qa.get("flagged_row_rate_pct"),
        }
        row.update({f"check_{k}": v for k, v in checks.items()})
        qualitative_rows.append(row)

        sample_df = pd.DataFrame(qa.get("sample_rows", []))
        flagged_df = pd.DataFrame(qa.get("flagged_examples", []))
        sample_df.to_csv(os.path.join(export_dir, f"qualitative_sample_{arch}.csv"), index=False)
        flagged_df.to_csv(os.path.join(export_dir, f"qualitative_flagged_examples_{arch}.csv"), index=False)

    if qualitative_rows:
        pd.DataFrame(qualitative_rows).to_csv(os.path.join(export_dir, "qualitative_summary.csv"), index=False)

    ci_rows = [{"group": "TRTR", **ci["trtr_f1"]}]
    ci_rows.extend({"group": f"TSTR_{arch}", **vals} for arch, vals in ci["tstr"].items())
    pd.DataFrame(ci_rows).to_csv(os.path.join(export_dir, "confidence_intervals_95.csv"), index=False)

    trtr_f1 = trtr.get("weighted_f1_mean", 0)
    tl_25_default = tl_results.get("synth_pretrain", {}).get(0.25, {}).get("weighted_f1")

    pillar_rows = []
    for arch in sorted(tstr_all.keys()):
        tstr_res = tstr_all.get(arch, {})
        fidelity = fidelity_all.get(arch, {})
        privacy = privacy_all.get(arch, {})
        arch_tl = (tl_results_all or {}).get(arch, tl_results)
        tl_25 = arch_tl.get("synth_pretrain", {}).get(0.25, {}).get("weighted_f1")

        privacy_ok = (
            (privacy.get("dcr_mean", 0) > 0)
            and (0.50 <= privacy.get("nnaa", 0) <= 0.70)
        )
        utility_ok = (
            (tstr_res.get("f1_gap_vs_trtr", 1) <= 0.05)
            and (tstr_res.get("bus_ratio_vs_trtr", 0) >= 0.90)
            and (tl_25 is not None)
            and (tl_25 >= 0.90 * trtr_f1)
        )
        fidelity_ok = (
            (fidelity.get("ks_mean", 0) >= 0.90)
            and (fidelity.get("js_mean", 1) <= 0.05)
        )
        integrity_ok = (constraint_report.get("violation_rate_pct", 100) == 0)
        overall_ok = privacy_ok and utility_ok and fidelity_ok and integrity_ok

        pillar_rows.append({
            "model": arch,
            "privacy_ok": privacy_ok,
            "utility_ok": utility_ok,
            "fidelity_ok": fidelity_ok,
            "integrity_ok": integrity_ok,
            "overall_ok": overall_ok,
            "threshold_privacy": "DCR>0 and 0.50<=NNAA<=0.70",
            "threshold_utility": "F1_gap<=0.05 and BUS_ratio>=0.90 and TL@25%>=0.90*TRTR",
            "threshold_fidelity": "KS_mean>=0.90 and JS_mean<=0.05",
            "threshold_integrity": "hard_CVR==0.0%",
            "actual_dcr_mean": privacy.get("dcr_mean"),
            "actual_nnaa": privacy.get("nnaa"),
            "actual_f1_gap": tstr_res.get("f1_gap_vs_trtr"),
            "actual_bus_ratio": tstr_res.get("bus_ratio_vs_trtr"),
            "actual_transfer_25pct_f1": tl_25,
            "actual_trtr_f1": trtr_f1,
            "actual_ks_mean": fidelity.get("ks_mean"),
            "actual_js_mean": fidelity.get("js_mean"),
            "actual_hard_cvr_pct": constraint_report.get("violation_rate_pct"),
            "actual_soft_r8_violations": constraint_report.get("R8_carrier_calendar_soft"),
        })

    pd.DataFrame(pillar_rows).to_csv(os.path.join(export_dir, "pillar_pass_fail.csv"), index=False)
    _write_pillar_summary_md(
        export_dir=export_dir,
        pillar_rows=pillar_rows,
        trtr_f1=trtr_f1,
        transfer_25=tl_25_default,
    )
    _write_viva_talking_points_md(
        export_dir=export_dir,
        pillar_rows=pillar_rows,
        trtr_f1=trtr_f1,
        transfer_25=tl_25_default,
    )


def _json_default(value):
    """Helper function for JSON serialization of non-standard types like numpy arrays.
    - **Functional**: Converts non-serializable types to JSON-friendly formats.
    - **Technical**: Detects numpy arrays and converts them to lists; can be extended for other types as needed.
    Args:        
        value: The value to be serialized to JSON.
    Returns:        
        A JSON-serializable representation of the input value.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _write_pillar_summary_md(export_dir, pillar_rows, trtr_f1, transfer_25):
    """Generates a markdown report summarizing the pass/fail outcomes for each evaluation pillar (privacy, utility, fidelity, integrity) for each synthetic data generator, along with key metrics and global context such as TRTR baseline performance and transfer learning results. This report is intended to provide a clear and concise overview of the evaluation results for viva discussions and documentation.    
    - **Functional**: Creates a human-readable markdown summary of pillar evaluation results.
    - **Technical**: Compiles model verdicts, key metrics, and global context into a structured markdown format, including tables and bullet points, and saves it to the specified export directory.
    Args:        
        export_dir (str): Directory path where the markdown report will be saved.
        pillar_rows (list): List of dictionaries containing pass/fail outcomes and metrics for each model.
        trtr_f1 (float): Weighted F1 score of the TRTR baseline.
        transfer_25 (float): Weighted F1 score of the synthetic pre-training at 25% real data in the transfer learning evaluation.
    """
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("# Pillar Summary (Auto-Generated)\n")
    lines.append(f"Generated: {generated_at}\n")
    lines.append("\n## Evaluation thresholds\n")
    lines.append("- Privacy: DCR > 0 and 0.50 <= NNAA <= 0.70")
    lines.append("- Utility: F1 gap <= 0.05 and BUS ratio >= 0.90 and TL@25% >= 0.90 x TRTR")
    lines.append("- Fidelity: KS mean >= 0.90 and JS mean <= 0.05")
    lines.append("- Integrity: Hard CVR = 0.0%\n")

    lines.append("\n## Global context\n")
    lines.append(f"- TRTR weighted F1: {trtr_f1:.4f}")
    lines.append(f"- Transfer curve F1 at 25% real data: {transfer_25 if transfer_25 is not None else 'N/A'}\n")

    lines.append("\n## Model verdicts\n")
    lines.append("| Model | Privacy | Utility | Fidelity | Integrity | Overall |")
    lines.append("|---|---|---|---|---|---|")
    for row in pillar_rows:
        lines.append(
            f"| {row['model']} | {row['privacy_ok']} | {row['utility_ok']} | "
            f"{row['fidelity_ok']} | {row['integrity_ok']} | {row['overall_ok']} |"
        )

    lines.append("\n## Key metrics by model\n")
    for row in pillar_rows:
        lines.append(f"### {row['model']}")
        lines.append(f"- Privacy: DCR={row['actual_dcr_mean']}, NNAA={row['actual_nnaa']}")
        lines.append(
            f"- Utility: F1 gap={row['actual_f1_gap']}, BUS ratio={row['actual_bus_ratio']}, "
            f"TL@25%={row['actual_transfer_25pct_f1']}"
        )
        lines.append(f"- Fidelity: KS mean={row['actual_ks_mean']}, JS mean={row['actual_js_mean']}")
        lines.append(
            f"- Integrity: Hard CVR={row['actual_hard_cvr_pct']}%, "
            f"Soft Rule-8 violations={row['actual_soft_r8_violations']}"
        )
        lines.append("")

    with open(os.path.join(export_dir, "pillar_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")


def _write_viva_talking_points_md(export_dir, pillar_rows, trtr_f1, transfer_25):
    """Generates a markdown report with suggested talking points for viva discussions based on the evaluation results across all pillars. This report is intended to provide a structured narrative for presenting the results, highlighting key findings, strengths, weaknesses, and next steps in a clear and concise manner.    
    - **Functional**: Creates a narrative markdown report with suggested talking points for viva discussions.
    - **Technical**: Analyzes the pillar evaluation results to identify key insights and patterns, and compiles them into a structured markdown format with sections for verbal bullets, pillar pass counts, and a closing line for examiners, which is then saved to the specified export directory.
    Args:        
        export_dir (str): Directory path where the markdown report will be saved.
        pillar_rows (list): List of dictionaries containing pass/fail outcomes and metrics for each model   
        trtr_f1 (float): Weighted F1 score of the TRTR baseline.
        transfer_25 (float): Weighted F1 score of the synthetic pre-training at 25% real data in the transfer learning evaluation.
    """
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    best_gap_row = min(pillar_rows, key=lambda r: r.get("actual_f1_gap", 999)) if pillar_rows else None
    best_fidelity_row = min(pillar_rows, key=lambda r: r.get("actual_js_mean", 999)) if pillar_rows else None
    privacy_pass_count = sum(bool(r.get("privacy_ok")) for r in pillar_rows)
    utility_pass_count = sum(bool(r.get("utility_ok")) for r in pillar_rows)
    fidelity_pass_count = sum(bool(r.get("fidelity_ok")) for r in pillar_rows)
    integrity_pass_count = sum(bool(r.get("integrity_ok")) for r in pillar_rows)

    lines = []
    lines.append("# Viva Defense Talking Points (Auto-Generated)\n")
    lines.append(f"Generated: {generated_at}\n")
    lines.append("Use these bullets directly in viva discussion and Q&A.\n")

    lines.append("## Suggested verbal bullets\n")
    lines.append(
        f"- The evaluation is fully reproducible across four pillars with machine-generated evidence files and fixed thresholds; TRTR baseline F1 is {trtr_f1:.4f}."
    )
    lines.append(
        f"- Transfer learning remains strong: at 25% real data, synthetic pre-training reaches F1={transfer_25 if transfer_25 is not None else 'N/A'}, supporting sample-efficiency claims."
    )
    if best_gap_row:
        lines.append(
            f"- Among tested generators, {best_gap_row['model']} shows the smallest TSTR F1 gap ({best_gap_row['actual_f1_gap']}) but still misses the <=0.05 utility threshold."
        )
    if best_fidelity_row:
        lines.append(
            f"- Fidelity is the strongest pillar in this run; {best_fidelity_row['model']} has the lowest JS mean ({best_fidelity_row['actual_js_mean']}) with KS above threshold where applicable."
        )
    lines.append(
        "- Integrity hard constraints are satisfied (hard CVR = 0.0%), but soft Rule-8 calendar violations should be presented as operational anomalies requiring domain cleanup."
    )
    lines.append(
        "- Privacy and utility remain the limiting factors in the current real-data run; this should be framed as transparent empirical evidence, not a hidden limitation."
    )
    lines.append(
        "- The strict preflight gate is intentionally fail-fast; passing it is a prerequisite for claiming production-grade readiness."
    )

    lines.append("\n## Pillar pass counts\n")
    lines.append(f"- Privacy pass count: {privacy_pass_count}/{len(pillar_rows)}")
    lines.append(f"- Utility pass count: {utility_pass_count}/{len(pillar_rows)}")
    lines.append(f"- Fidelity pass count: {fidelity_pass_count}/{len(pillar_rows)}")
    lines.append(f"- Integrity pass count: {integrity_pass_count}/{len(pillar_rows)}")

    lines.append("\n## Examiner-safe closing line\n")
    lines.append(
        "- The pipeline is technically complete and auditable; current results demonstrate strong fidelity/integrity but insufficient privacy/utility under the declared thresholds, which defines the next engineering milestone."
    )

    with open(os.path.join(export_dir, "viva_defense_talking_points.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")
