"""
Naive Baseline: LLM Judgement for Uncertainty Scoring

Use an LLM to evaluate uncertainty in reasoning traces.

Example usage:
    python naive_attack_baseline/naive_LLMjudgement.py \
        -i <input_data.xlsx> \
        -o <output_analysis.xlsx> \
        --api-key <your_api_key> \
        --base-url <api_base_url> \
        --model <model_name>
        
    # Or use environment variable for API key:
    export OPENAI_API_KEY="sk-xxxxxx"
    python naive_attack_baseline/naive_LLMjudgement.py \
        -i <input_data.xlsx> \
        -o <output_analysis.xlsx>
"""

import os
import argparse
import time
import pandas as pd
from openai import OpenAI
from tqdm import tqdm


def build_prompt(reasoning_text):
    """
    Build the evaluation prompt for uncertainty scoring.
    
    Args:
        reasoning_text: The reasoning path text to evaluate
        
    Returns:
        List of message dictionaries for the chat API
    """
    return [
        {
            "role": "system",
            "content": "You are an assistant that evaluates the uncertainty expressed in reasoning text.",
        },
        {
            "role": "user",
            "content": (
                "Given the following reasoning path, rate how uncertain or speculative "
                "the language sounds on a scale from 0 (very certain and factual) "
                "to 1 (very uncertain and speculative).\n\n"
                "Respond only with a number between 0 and 1.\n\n"
                f'Text:\n"""{reasoning_text}"""'
            ),
        },
    ]


def score_text(client, text, model_name, temperature=0):
    """
    Score a single text using the LLM.
    
    Args:
        client: OpenAI client instance
        text: Text to score
        model_name: Name of the model to use
        temperature: Sampling temperature
        
    Returns:
        Float score between 0-1, or None if failed
    """
    try:
        messages = build_prompt(text)
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature
        )
        
        reply = response.choices[0].message.content.strip()
        
        try:
            score = float(reply)
            score = max(0.0, min(1.0, score))  # Clamp between 0 and 1
            return score
        except ValueError:
            print(f"⚠️  Could not parse model reply: {reply}")
            return None
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Score reasoning traces using LLM judgement"
    )
    parser.add_argument("-i", "--input", required=True, help="Input Excel file with reasoning paths")
    parser.add_argument("-o", "--output", required=True, help="Output Excel file path")
    parser.add_argument("-c", "--column", default="Reasoning_path_1", help="Column name to evaluate (default: Reasoning_path_1)")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY environment variable)")
    parser.add_argument("--base-url", default="your base url here", help="API base URL")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="Model name (default: gpt-3.5-turbo)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default: 0.0)")
    parser.add_argument("--rate-limit-delay", type=float, default=1.0, help="Delay between requests in seconds (default: 1.0)")
    args = parser.parse_args()
    
    # Get API key from argument or environment variable
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key is required. Provide via --api-key or set OPENAI_API_KEY environment variable.")
    
    # Initialize OpenAI client
    print(f"Initializing OpenAI client with model: {args.model}")
    client = OpenAI(api_key=api_key, base_url=args.base_url)
    
    # Load data
    print(f"Loading data from: {args.input}")
    df = pd.read_excel(args.input)
    
    if args.column not in df.columns:
        raise ValueError(f"Column '{args.column}' not found. Available columns: {list(df.columns)}")
    
    texts = df[args.column].astype(str)
    print(f"✅ Successfully loaded {len(texts)} texts")
    
    # Process texts
    uncertainty_scores = []
    print(f"⚙️  Scoring texts using {args.model}...")
    
    for text in tqdm(texts, desc="Scoring"):
        score = score_text(client, text, args.model, args.temperature)
        uncertainty_scores.append(score)
        
        # Rate limiting
        if args.rate_limit_delay > 0:
            time.sleep(args.rate_limit_delay)
    
    # Save results
    df["LLM_Judge_Uncertainty"] = uncertainty_scores
    df.to_excel(args.output, index=False)
    
    # Print statistics
    valid_scores = [s for s in uncertainty_scores if s is not None]
    if valid_scores:
        print(f"✅ Processing complete. Results saved to: {args.output}")
        print(f"   - Successfully scored: {len(valid_scores)}/{len(uncertainty_scores)}")
        print(f"   - Mean uncertainty: {sum(valid_scores)/len(valid_scores):.4f}")
        print(f"   - Min uncertainty: {min(valid_scores):.4f}")
        print(f"   - Max uncertainty: {max(valid_scores):.4f}")
    else:
        print(f"⚠️  No valid scores obtained. Check your API configuration.")


if __name__ == "__main__":
    main()