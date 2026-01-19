"""
Naive Baseline: Thinking Token Count and Compression Rate Analysis
"""

import os
import argparse
from collections import Counter

import torch
import pandas as pd
import nltk
from tqdm import tqdm
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer, util


# Setup NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


# Thinking token seed phrases
THINKING_SEEDS = [
    "Hmm", "Wait", "So", "Therefore", "Now", "Maybe", "Since", "I think",
    "Hold on", "Okay", "appear", "However", "perhaps", "could"
]


class ThinkingTokenAnalyzer:
    """Extract thinking tokens from text."""
    
    def __init__(self, embed_model, stop_words, device, threshold=0.7):
        self.embed_model = embed_model
        self.stop_words = stop_words
        self.device = device
        self.threshold = threshold
        self.seed_embeddings = embed_model.encode(THINKING_SEEDS, convert_to_tensor=True, device=device)
    
    def extract(self, text):
        """Extract thinking tokens from text."""
        if not text or not isinstance(text, str):
            return Counter()
        
        tokens_set = set()
        try:
            sentences = sent_tokenize(text)
        except:
            return Counter()
        
        for sent in sentences:
            try:
                words = [w for w in word_tokenize(sent) if w.isalpha()]
            except:
                continue
            
            for i in range(len(words)):
                w = words[i].lower()
                if w not in self.stop_words:
                    tokens_set.add(w)
                if i + 1 < len(words):
                    tokens_set.add(f"{w} {words[i+1].lower()}")
                if i + 2 < len(words):
                    tokens_set.add(f"{w} {words[i+1].lower()} {words[i+2].lower()}")
        
        if not tokens_set:
            return Counter()
        
        candidates = list(tokens_set)
        try:
            candidate_embeddings = self.embed_model.encode(candidates, convert_to_tensor=True, device=self.device)
            sim_matrix = util.pytorch_cos_sim(candidate_embeddings, self.seed_embeddings)
        except:
            return Counter()
        
        selected = []
        for i, row in enumerate(sim_matrix):
            if torch.max(row).item() >= self.threshold:
                selected.append(candidates[i])
        
        return Counter(selected)


class CompressionAnalyzer:
    """Compute text compression using GPT-2."""
    
    def __init__(self, tokenizer, model, device, max_length=1024):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.max_length = max_length
    
    def compute(self, text):
        """Compute compression score (NLL)."""
        if not text or not isinstance(text, str):
            return 0.0
        
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=False)
            input_ids = inputs["input_ids"].squeeze(0).to(self.device)
            n_tokens = len(input_ids)
            
            if n_tokens == 0:
                return 0.0
            
            total_nll = 0.0
            
            if n_tokens <= self.max_length:
                with torch.no_grad():
                    outputs = self.model(input_ids.unsqueeze(0), labels=input_ids.unsqueeze(0))
                total_nll = outputs.loss.item() * n_tokens
            else:
                for i in range(0, n_tokens, self.max_length):
                    chunk = input_ids[i:i + self.max_length]
                    if len(chunk) < 2:
                        continue
                    with torch.no_grad():
                        outputs = self.model(chunk.unsqueeze(0), labels=chunk.unsqueeze(0))
                    total_nll += outputs.loss.item() * len(chunk)
            
            return total_nll
        except:
            return 0.0


def main():
    parser = argparse.ArgumentParser(description="Analyze thinking tokens and compression rates")
    parser.add_argument("-i", "--input", required=True, help="Input Excel file")
    parser.add_argument("-o", "--output", required=True, help="Output Excel file")
    parser.add_argument("-c", "--column", default="Reasoning_path_1", help="Text column name")
    parser.add_argument("--gpt2-model", default="gpt2", help="GPT-2 model path")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model path")
    parser.add_argument("--threshold", type=float, default=0.8, help="Similarity threshold")
    parser.add_argument("--device", default="auto", help="Device: cuda, cuda:0, 0, 1, cpu, or auto")
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device.isdigit():
        # If device is a number string like '0' or '1', convert to 'cuda:0' or 'cuda:1'
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load models
    print(f"Loading GPT-2 model: {args.gpt2_model}")
    tokenizer = GPT2Tokenizer.from_pretrained(args.gpt2_model)
    gpt2_model = GPT2LMHeadModel.from_pretrained(args.gpt2_model).to(device)
    gpt2_model.eval()
    
    print(f"Loading embedding model: {args.embedding_model}")
    embed_model = SentenceTransformer(args.embedding_model, device=device)
    
    stop_words = set(stopwords.words('english'))
    
    # Initialize analyzers
    thinking_analyzer = ThinkingTokenAnalyzer(embed_model, stop_words, device, args.threshold)
    compression_analyzer = CompressionAnalyzer(tokenizer, gpt2_model, device)
    
    # Load data
    print(f"Reading input: {args.input}")
    df = pd.read_excel(args.input)
    
    if args.column not in df.columns:
        raise ValueError(f"Column '{args.column}' not found. Available: {list(df.columns)}")
    
    texts = df[args.column].astype(str)
    print(f"Processing {len(texts)} texts")
    
    # Process
    compression_list = []
    thinking_count_list = []
    
    for text in tqdm(texts, desc="Processing"):
        compression = compression_analyzer.compute(text)
        tokens = thinking_analyzer.extract(text)
        
        compression_list.append(compression)
        thinking_count_list.append(sum(tokens.values()))
    
    # Save results
    df["ThinkingTokenCount"] = thinking_count_list
    df["CompressionRate"] = compression_list
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    df.to_excel(args.output, index=False)
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
