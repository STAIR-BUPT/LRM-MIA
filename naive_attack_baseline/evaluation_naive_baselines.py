"""
Evaluation Script for Naive Baseline Methods

Evaluates naive baseline methods at both sequence-level and document-level.

Usage:
    python evaluation_naive_baselines.py \
        -m /path/to/member_data.xlsx \
        -n /path/to/nonmember_data.xlsx \
        -o /path/to/results.xlsx
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import ttest_ind


def evaluate_metrics(df, level_name="sequence", exclude_cols=None, label_col="Label"):
    """
    Evaluate discriminative power of numeric metrics.
    
    Args:
        df: DataFrame with Label column and metric columns
        level_name: Name of the evaluation level (for display)
        exclude_cols: List of columns to exclude from evaluation
        label_col: Name of the label column
        
    Returns:
        DataFrame with evaluation results
    """
    if exclude_cols is None:
        exclude_cols = []
    
    metric_cols = [col for col in df.columns if col not in exclude_cols + [label_col]]
    y_true = df[label_col]
    
    results = []
    
    for col in metric_cols:
        score = df[col].copy()
        
        # Determine direction based on metric name
        if any(key in col for key in ["CompressionRate", "ThinkingTokenCount", "LLMjudge"]):
            score = -score
            direction = "-"
        else:
            direction = "+"
        
        # Handle missing values
        score = score.fillna(df.groupby(label_col)[col].transform("mean"))
        
        # ROC metrics
        auc_val = roc_auc_score(y_true, score)
        fpr, tpr, _ = roc_curve(y_true, score)
        balanced_accuracies = (tpr + (1 - fpr)) / 2
        max_bal_acc = np.max(balanced_accuracies)
        tpr_at_5fpr = np.interp(0.05, fpr, tpr)
        
        # Statistical tests
        score_member = score[df[label_col] == 1]
        score_nonmember = score[df[label_col] == 0]
        
        _, p_value = ttest_ind(score_member, score_nonmember, equal_var=False)
        
        # Cohen's d effect size
        mean_diff = score_member.mean() - score_nonmember.mean()
        pooled_std = np.sqrt((score_member.var(ddof=1) + score_nonmember.var(ddof=1)) / 2)
        cohen_d = mean_diff / pooled_std if pooled_std > 0 else np.nan
        
        results.append({
            "Level": level_name,
            "Metric": col,
            "Direction": direction,
            "AUC": auc_val,
            "MaxBalancedAcc": max_bal_acc,
            "TPR@5%FPR": tpr_at_5fpr,
            "p-value": p_value,
            "EffectSize(Cohen_d)": cohen_d
        })
    
    return pd.DataFrame(results)


def format_results(results_df):
    """Format results for display."""
    formatted = results_df.copy()
    
    # Format p-values in scientific notation
    formatted["p-value"] = formatted["p-value"].apply(
        lambda x: f"{x:.3e}" if pd.notnull(x) else "NaN"
    )
    
    # Round other numeric columns to 3 decimals
    for col in ["AUC", "MaxBalancedAcc", "TPR@5%FPR", "EffectSize(Cohen_d)"]:
        if col in formatted.columns:
            formatted[col] = formatted[col].apply(lambda x: round(x, 3) if pd.notnull(x) else np.nan)
    
    return formatted


def main():
    # ====================================
    # Parse command line arguments
    # ====================================
    parser = argparse.ArgumentParser(description="Evaluate naive baseline methods")
    parser.add_argument("-m", "--member_data", required=True, help="Member data file")
    parser.add_argument("-n", "--nonmember_data", required=True, help="Non-member data file")
    parser.add_argument("-o", "--output", default=None, help="Output results file (optional)")
    args = parser.parse_args()
    
    # ====================================
    # 1. Load datasets
    # ====================================
    df_member = pd.read_excel(args.member_data)
    df_nonmember = pd.read_excel(args.nonmember_data)
    
    print(f"✅ Loaded {len(df_member)} member and {len(df_nonmember)} non-member samples")
    print(f"   Total: {len(df_member) + len(df_nonmember)} samples")
    
    df_member["Label"] = 1  # Member samples
    df_nonmember["Label"] = 0  # Non-member samples
    
    # Exclude non-metric columns
    exclude_cols = [
        "ID", "Item", "Label",
        "Reasoning_path_1", "Output_1",
        "Reasoning_path_2", "Output_2",
        "Reasoning_path_3", "Output_3",
        "Reasoning_Path_1", "Reasoning_Path_2", "Reasoning_Path_3"
    ]
    
    # ====================================
    # 2. Sequence-level evaluation
    # ====================================
    print("\n" + "="*80)
    print("SEQUENCE-LEVEL EVALUATION")
    print("="*80)
    
    df_all_seq = pd.concat([df_member, df_nonmember], ignore_index=True)
    results_seq = evaluate_metrics(df_all_seq, level_name="sequence", exclude_cols=exclude_cols)
    results_seq_formatted = format_results(results_seq)
    
    print(results_seq_formatted.to_string(index=False))
    
    # ====================================
    # 3. Document-level evaluation
    # ====================================
    print("\n" + "="*80)
    print("DOCUMENT-LEVEL EVALUATION (aggregated by ID)")
    print("="*80)
    
    # Aggregate numeric metrics by ID and Label
    numeric_cols = df_all_seq.select_dtypes(include=[np.number]).columns.tolist()
    agg_cols = [col for col in numeric_cols if col != "Label"]
    
    df_all_doc = df_all_seq.groupby(["ID", "Label"])[agg_cols].mean().reset_index()
    print(f"Aggregated to {len(df_all_doc)} unique documents")
    
    results_doc = evaluate_metrics(df_all_doc, level_name="document", exclude_cols=exclude_cols)
    results_doc_formatted = format_results(results_doc)
    
    print(results_doc_formatted.to_string(index=False))
    
    # ====================================
    # 4. Combine and save results
    # ====================================
    all_results = pd.concat([results_seq_formatted, results_doc_formatted], ignore_index=True)
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save combined results
        all_results.to_excel(output_path, index=False)
        print(f"\n✅ Results saved to: {output_path}")
        
        # Also save separate sheets
        output_detailed = output_path.parent / (output_path.stem + "_detailed.xlsx")
        with pd.ExcelWriter(output_detailed) as writer:
            results_seq_formatted.to_excel(writer, sheet_name="Sequence-Level", index=False)
            results_doc_formatted.to_excel(writer, sheet_name="Document-Level", index=False)
        print(f"✅ Detailed results saved to: {output_detailed}")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
