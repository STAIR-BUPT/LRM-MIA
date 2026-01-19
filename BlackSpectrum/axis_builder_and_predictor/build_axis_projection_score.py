"""
Build Axis Projection Score

This script computes projection scores for reasoning embeddings along a 
memorization-generalization axis. It uses anchor data to establish the axis
and then projects query data onto this axis.

Usage:
    python build_axis_projection_score.py \
        --generalization_data <path> \
        --memorization_data <path> \
        --query_data <path> \
        --output <path> \
        --encoder <model_name>

Encoder Selection Tips:
    Different target LLMs work best with the following encoders:
    - QwQ: sentence-transformers/all-mpnet-base-v2
    - Gemini: sentence-transformers/all-MiniLM-L6-v2
    - Claude: sentence-transformers/all-distilroberta-v1
    - GPT-5: sentence-transformers/all-MiniLM-L6-v2
    
        Choose the optimal encoder based on small random selected samples for validation.
        Observe the anchor point distance (larger is better).

"""

import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


class AxisProjectionScorer:
    """
    Computes projection scores for reasoning embeddings along a 
    memorization-generalization axis.
    """
    
    def __init__(self, model_name="sentence-transformers/all-distilroberta-v1", device=None):
        """
        Initialize the scorer with a sentence transformer model.
        
        Args:
            model_name: Name or local path of the sentence transformer model
            device: Device to use ('cuda', 'cuda:0', 'cuda:1', 'cpu', or GPU number). Auto-detect if None.
        """
        # Handle different device input formats
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device.isdigit():
            # If device is a number string like '1', convert to 'cuda:1'
            self.device = f"cuda:{device}"
        else:
            # Otherwise use as is (e.g., 'cuda', 'cuda:0', 'cpu')
            self.device = device
        print(f"Using device: {self.device}")
        
        self.model = SentenceTransformer(model_name, device=self.device)
        self.hidden_size = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded: {model_name}")
        print(f"Embedding dimension: {self.hidden_size}")
        
    def encode_text(self, text):
        """Encode text to embedding vector."""
        return self.model.encode(text, convert_to_numpy=True, device=str(self.device))

    def remove_item_info(self, R, I):
        """
        Remove the component of R that aligns with I using projection.
        
        Args:
            R: Reasoning embedding vector
            I: Item embedding vector
            
        Returns:
            R with I component removed
        """
        if np.linalg.norm(I) < 1e-9:
            return R
        proj = (np.dot(R, I) / np.dot(I, I)) * I
        return R - proj

    def get_avg_reasoning_embeddings_with_item_removed(self, df, use_all_paths=True):
        """
        Compute mean reasoning embedding per row, removing item information.
        
        Args:
            df: DataFrame with columns 'Item', 'Reasoning_path_1', 'Reasoning_path_2', 'Reasoning_path_3'
            use_all_paths: If True, use all 3 reasoning paths. If False, only use path 1.
            
        Returns:
            embeddings: Array of cleaned reasoning embeddings
            texts_combined: List of combined text strings
        """
        embeddings = []
        texts_combined = []
        
        cols = ["Reasoning_path_1", "Reasoning_path_2", "Reasoning_path_3"] if use_all_paths else ["Reasoning_path_1"]
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing embeddings"):
            parts, part_embeddings = [], []
            
            # Encode reasoning paths
            for col in cols:
                text = row.get(col)
                if pd.notna(text) and isinstance(text, str) and text.strip():
                    parts.append(text.strip())
                    emb = self.encode_text(text.strip())
                    part_embeddings.append(emb)
            
            # Average reasoning embeddings
            if part_embeddings:
                R = np.mean(part_embeddings, axis=0)
            else:
                R = np.zeros(self.hidden_size)
            
            # Encode item
            item_text = row.get("Item")
            if pd.notna(item_text) and isinstance(item_text, str) and item_text.strip():
                I = self.encode_text(item_text.strip())
            else:
                I = np.zeros_like(R)
            
            # Remove item information from reasoning
            R_clean = self.remove_item_info(R, I)
            embeddings.append(R_clean)
            texts_combined.append(" || ".join(parts))
        
        return np.array(embeddings), texts_combined

    def remove_outliers_by_std(self, points, z_thresh=2.0):
        """
        Remove embedding outliers based on distance from centroid.
        
        Args:
            points: Array of embedding vectors
            z_thresh: Z-score threshold for outlier removal
            
        Returns:
            Filtered array with outliers removed
        """
        centroid = np.mean(points, axis=0)
        dists = np.linalg.norm(points - centroid, axis=1)
        z_scores = (dists - np.mean(dists)) / np.std(dists)
        mask = z_scores < z_thresh
        print(f"Outlier removal: total {len(points)}, kept {np.sum(mask)}, removed {np.sum(~mask)}")
        return points[mask]

    def build_axis(self, generalization_embeddings, memorization_embeddings, z_thresh=2.0):
        """
        Build the memorization-generalization axis from anchor embeddings.
        
        Args:
            generalization_embeddings: Embeddings from generalization data
            memorization_embeddings: Embeddings from memorization data
            z_thresh: Z-score threshold for outlier removal
            
        Returns:
            mem_center: Center of memorization cluster
            gen_center: Center of generalization cluster
            axis_unit: Unit vector along the axis (mem -> gen direction)
        """
        # Remove outliers
        gen_filtered = self.remove_outliers_by_std(generalization_embeddings, z_thresh)
        mem_filtered = self.remove_outliers_by_std(memorization_embeddings, z_thresh)
        
        # Compute centers
        gen_center = np.mean(gen_filtered, axis=0)
        mem_center = np.mean(mem_filtered, axis=0)
        
        # Compute axis unit vector
        axis_vec = gen_center - mem_center
        axis_unit = axis_vec / np.linalg.norm(axis_vec)
        
        print(f"Axis built: ||gen_center - mem_center|| = {np.linalg.norm(axis_vec):.4f}")
        
        return mem_center, gen_center, axis_unit
    
    def compute_projection_scores(self, embeddings, mem_center, gen_center, axis_unit):
        """
        Compute projection scores for query embeddings.
        
        Args:
            embeddings: Array of query embeddings
            mem_center: Memorization center
            gen_center: Generalization center
            axis_unit: Axis unit vector
            
        Returns:
            projection_from_mem: Distance from memorization center along axis
            projection_to_gen: Distance to generalization center along axis
        """
        projection_from_mem = []
        projection_to_gen = []

        for vec in tqdm(embeddings, desc="Computing projections"):
            # Distance from memorization center
            proj_from_mem = np.dot(vec - mem_center, axis_unit)
            projection_from_mem.append(proj_from_mem)
            
            # Distance to generalization center
            proj_to_gen = np.dot(gen_center - vec, axis_unit)
            projection_to_gen.append(proj_to_gen)
        
        return projection_from_mem, projection_to_gen


def main():
    parser = argparse.ArgumentParser(description="Build axis projection scores for MIA detection")
    
    # Data paths
    parser.add_argument("--generalization_data", type=str, required=True,
                        help="Path to generalization anchor data (Excel file)")
    parser.add_argument("--memorization_data", type=str, required=True,
                        help="Path to memorization anchor data (Excel file)")
    parser.add_argument("--query_data", type=str, required=True,
                        help="Path to query data (Excel file)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for results (Excel file)")

    # Model configuration
    parser.add_argument("--encoder", type=str, 
                        default="sentence-transformers/all-distilroberta-v1",
                        help="Sentence transformer model name or local path")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda/cpu). Auto-detect if not specified")
    
    # Processing options
    parser.add_argument("--z_thresh", type=float, default=2.0,
                        help="Z-score threshold for outlier removal")
    
    args = parser.parse_args()
    
    # Initialize scorer
    scorer = AxisProjectionScorer(model_name=args.encoder, device=args.device)
    
    # Load data
    print("\nLoading data...")
    df_gen = pd.read_excel(args.generalization_data)
    df_mem = pd.read_excel(args.memorization_data)
    df_query = pd.read_excel(args.query_data)
    print(f"Generalization samples: {len(df_gen)}")
    print(f"Memorization samples: {len(df_mem)}")
    print(f"Query samples: {len(df_query)}")
    
    # Compute embeddings for anchor data
    # Note: Use all 3 reasoning paths to build better anchor points for axis construction
    print("\n=== Processing Anchor Data ===")
    gen_embeddings, _ = scorer.get_avg_reasoning_embeddings_with_item_removed(
        df_gen, use_all_paths=True
    )
    mem_embeddings, _ = scorer.get_avg_reasoning_embeddings_with_item_removed(
        df_mem, use_all_paths=True
    )
    
    # Build axis
    print("\n=== Building Axis ===")
    mem_center, gen_center, axis_unit = scorer.build_axis(
        gen_embeddings, mem_embeddings, z_thresh=args.z_thresh
    )
    
    # Compute embeddings for query data
    # Note: Use only Reasoning_path_1 to demonstrate that the method is effective 
    # even with fewer queries (less information leakage)
    print("\n=== Processing Query Data ===")
    query_embeddings, texts_combined = scorer.get_avg_reasoning_embeddings_with_item_removed(
        df_query, use_all_paths=False
    )
    
    # Compute projection scores
    print("\n=== Computing Projection Scores ===")
    proj_from_mem, proj_to_gen = scorer.compute_projection_scores(
        query_embeddings, mem_center, gen_center, axis_unit
    )
    
    # Build result DataFrame
    print("\n=== Building Results ===")
    result_df = df_query[["ID", "Item"]].copy()
    result_df["Reasoning_path_combined"] = texts_combined
    result_df["CenterProjectionFromMemorization"] = proj_from_mem
    result_df["CenterProjectionToGeneralization"] = proj_to_gen
    result_df["ScoreMethod"] = "Center-based projection (Item info removed)"

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_excel(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    print(f"Total samples processed: {len(result_df)}")


if __name__ == "__main__":
    main()
