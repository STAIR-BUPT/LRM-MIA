"""
Naive Baseline: Trace Consistency Analysis

Compute character-level and token-level edit distance consistency across multiple reasoning paths.

Example usage:
    python naive_attack_baseline/naive_trace_consistency.py \
        -i <input_data.xlsx> \
        -o <output_analysis.xlsx>
"""

import argparse
import pandas as pd
import Levenshtein
from tqdm import tqdm


def compute_edit_distance_consistency_char(paths):
    """
    Compute character-level edit distance consistency across multiple reasoning paths.
    
    Args:
        paths: List of reasoning path strings
        
    Returns:
        Average similarity score (0-1)
    """
    paths = [p for p in paths if pd.notna(p)]
    if len(paths) < 2:
        return 0.0

    sim_scores = []
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            s1, s2 = str(paths[i]), str(paths[j])
            max_len = max(len(s1), len(s2))
            if max_len == 0:
                sim = 1.0
            else:
                dist = Levenshtein.distance(s1, s2)
                sim = 1 - dist / max_len
            sim_scores.append(sim)
    return sum(sim_scores) / len(sim_scores)


def compute_edit_distance_consistency_token(paths):
    """
    Compute token-level edit distance consistency (based on token mapping).
    
    Args:
        paths: List of reasoning path strings
        
    Returns:
        Average similarity score (0-1)
    """
    paths = [p for p in paths if pd.notna(p)]
    if len(paths) < 2:
        return 0.0

    sim_scores = []
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            tokens1 = str(paths[i]).split()
            tokens2 = str(paths[j]).split()
            max_len = max(len(tokens1), len(tokens2))

            if max_len == 0:
                sim = 1.0
            else:
                # Build token mapping to ensure unique and consistent representation
                all_tokens = set(tokens1 + tokens2)
                vocab = {tok: chr(65 + k % 26) if k < 26 else f"T{k}" for k, tok in enumerate(all_tokens)}
                s1 = "".join(vocab.get(t, "?") for t in tokens1)
                s2 = "".join(vocab.get(t, "?") for t in tokens2)
                dist = Levenshtein.distance(s1, s2)
                sim = 1 - dist / max_len

            sim_scores.append(sim)
    return sum(sim_scores) / len(sim_scores)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze trace consistency using edit distance"
    )
    parser.add_argument("-i", "--input", required=True, help="Input Excel file with reasoning paths")
    parser.add_argument("-o", "--output", required=True, help="Output Excel file path")
    parser.add_argument("--path-columns", nargs="+", 
                        default=["Reasoning_path_1", "Reasoning_path_2", "Reasoning_path_3"],
                        help="Column names containing reasoning paths (default: Reasoning_path_1 Reasoning_path_2 Reasoning_path_3)")
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from: {args.input}")
    df = pd.read_excel(args.input)
    print(f"✅ Successfully loaded data: {len(df)} rows")
    
    # Check if specified columns exist
    missing_cols = [col for col in args.path_columns if col not in df.columns]
    if missing_cols:
        print(f"⚠️  Warning: Columns not found: {missing_cols}")
        args.path_columns = [col for col in args.path_columns if col in df.columns]
        print(f"Using available columns: {args.path_columns}")
    
    if len(args.path_columns) < 2:
        raise ValueError(f"Need at least 2 reasoning path columns. Found: {args.path_columns}")
    
    # Compute consistency metrics
    char_edit_scores = []
    token_edit_scores = []
    
    print("⚙️  Computing character-level and token-level edit distance consistency...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        paths = [row.get(col) for col in args.path_columns]
        char_edit_scores.append(compute_edit_distance_consistency_char(paths))
        token_edit_scores.append(compute_edit_distance_consistency_token(paths))
    
    # Store results
    df["CharLevel_EditConsistency"] = char_edit_scores
    df["TokenLevel_EditConsistency"] = token_edit_scores
    
    # Save results
    df.to_excel(args.output, index=False)
    print(f"✅ Processing complete. Results saved to: {args.output}")
    print(f"   - CharLevel_EditConsistency: Mean = {sum(char_edit_scores)/len(char_edit_scores):.4f}")
    print(f"   - TokenLevel_EditConsistency: Mean = {sum(token_edit_scores)/len(token_edit_scores):.4f}")


if __name__ == "__main__":
    main()