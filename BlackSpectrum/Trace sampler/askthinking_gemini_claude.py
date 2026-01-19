#!/usr/bin/env python3
"""
Trace Sampler for Thinking Models (Gemini/Claude)

This script collects multiple reasoning traces and outputs from thinking-enabled LLMs
(e.g., Claude Sonnet 4 Thinking, Gemini 2.5 Flash Preview Thinking) for membership
inference attack evaluation.

Usage:
    python askthinking_gemini_claude.py \
        --input_path /path/to/input.xlsx \
        --output_path /path/to/output.xlsx \
        --api_key YOUR_API_KEY \
        --base_url https://api.example.com/v1/ \
        --model_name claude-sonnet-4-20250514-thinking \
        --n_responses 3 \
        --sleep_seconds 1.2 \
        --save_interval 10

"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any

import pandas as pd
from tqdm import tqdm
from openai import OpenAI


# === Logging Configuration ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


# === Configuration Class ===
class Config:
    """Configuration for the trace sampler."""
    
    def __init__(self, args: argparse.Namespace):
        self.api_key = args.api_key
        self.base_url = args.base_url
        self.model_name = args.model_name
        self.n_responses = args.n_responses
        self.sleep_seconds = args.sleep_seconds
        self.save_interval = args.save_interval
        self.input_path = args.input_path
        self.output_path = args.output_path
        
        # Validate configuration
        self._validate()
    
    def _validate(self):
        """Validate configuration parameters."""
        if not self.api_key or self.api_key == "YOUR_API_KEY_HERE":
            raise ValueError("Please provide a valid API key using --api_key or set API_KEY environment variable")
        
        if not Path(self.input_path).exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        
        if self.n_responses < 1:
            raise ValueError("n_responses must be at least 1")
        
        if self.sleep_seconds < 0:
            raise ValueError("sleep_seconds must be non-negative")
        
        # Create output directory if it doesn't exist
        output_dir = Path(self.output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")


# === Trace Collector ===
class TraceCollector:
    """Collects reasoning traces from thinking-enabled LLMs."""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        logger.info(f"Initialized TraceCollector with model: {config.model_name}")
    
    def get_reasoning_paths(self, query: str, n: int = None) -> List[Tuple[str, str]]:
        """
        Sends a query to the model multiple times to obtain different reasoning paths and outputs.
        
        Args:
            query: The text query to send to the model
            n: Number of responses to collect (default: uses config value)
        
        Returns:
            List of tuples containing (reasoning_path, output)
        """
        if n is None:
            n = self.config.n_responses
        
        results = []
        attempt = 0
        max_attempts = n * 3  # Prevent infinite loops
        
        while len(results) < n and attempt < max_attempts:
            attempt += 1
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {
                            "role": "user",
                            "content": (
                                f"Consider this passage. "
                                f"First, reflect on whether you have encountered it before. "
                                f"Next, try to identify its source — such as a book, article, website, or dataset. "
                                f"Finally, answer: What is the next word: {query}"
                            ),
                        },
                    ],
                )
                
                reasoning_path = getattr(response.choices[0].message, "reasoning_content", "").strip()
                output = response.choices[0].message.content.strip()
                
                if reasoning_path and output:
                    logger.debug(f"Successfully collected reasoning path {len(results) + 1}/{n} (Attempt {attempt})")
                    
                    # Print reasoning path to terminal
                    print("\n" + "="*80)
                    print(f"Reasoning Path {len(results) + 1}/{n} (Attempt: {attempt})")
                    print("="*80)
                    print(f"Reasoning Process:\n{reasoning_path}")
                    print("-"*80)
                    print(f"Output:\n{output}")
                    print("="*80 + "\n")
                    
                    results.append((reasoning_path, output))
                else:
                    logger.warning(f"Incomplete result in attempt {attempt}, retrying...")
            
            except Exception as e:
                logger.error(f"Error in attempt {attempt} for query '{query[:50]}...': {e}")
                # If we get too many errors, wait a bit longer
                if attempt % 5 == 0:
                    logger.info(f"Multiple errors detected, waiting {self.config.sleep_seconds * 2}s...")
                    time.sleep(self.config.sleep_seconds * 2)
            
            time.sleep(self.config.sleep_seconds)
        
        if len(results) < n:
            logger.warning(f"Could only collect {len(results)}/{n} responses after {attempt} attempts")
        
        return results


# === Data Processor ===
class DataProcessor:
    """Processes Excel files and collects reasoning traces."""
    
    def __init__(self, config: Config, collector: TraceCollector):
        self.config = config
        self.collector = collector
    
    def process_excel(self) -> pd.DataFrame:
        """
        Processes an Excel file, sends each row's content to the model,
        and stores reasoning and output results.
        
        Returns:
            DataFrame containing all results
        """
        logger.info(f"Loading input file: {self.config.input_path}")
        df = pd.read_excel(self.config.input_path)
        logger.info(f"Loaded {len(df)} items to process")
        
        results = []
        
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing items"):
            item_id = row.get("ID", index)
            item_text = str(row.get("Item", ""))
            
            logger.info(f"Processing ID {item_id}...")
            
            try:
                responses = self.collector.get_reasoning_paths(item_text, n=self.config.n_responses)
                
                result_entry = {
                    "ID": item_id,
                    "Item": item_text,
                }
                
                for i, (reasoning, output) in enumerate(responses, start=1):
                    result_entry[f"Reasoning_Path_{i}"] = reasoning
                    result_entry[f"Output_{i}"] = output
                
                # Fill missing columns if we didn't get all responses
                for i in range(len(responses) + 1, self.config.n_responses + 1):
                    result_entry[f"Reasoning_Path_{i}"] = ""
                    result_entry[f"Output_{i}"] = ""
                
                results.append(result_entry)
                
                # Save progress at intervals
                if (index + 1) % self.config.save_interval == 0:
                    self._save_progress(results)
                    logger.info(f"Saved progress: {index + 1}/{len(df)} items")
            
            except Exception as e:
                logger.error(f"Failed to process item {item_id}: {e}")
                # Add empty entry to maintain data consistency
                result_entry = {"ID": item_id, "Item": item_text}
                for i in range(1, self.config.n_responses + 1):
                    result_entry[f"Reasoning_Path_{i}"] = ""
                    result_entry[f"Output_{i}"] = ""
                results.append(result_entry)
        
        # Final save
        result_df = pd.DataFrame(results)
        result_df.to_excel(self.config.output_path, index=False)
        logger.info(f"All reasoning paths and outputs saved to: {self.config.output_path}")
        
        return result_df
    
    def _save_progress(self, results: List[Dict[str, Any]]):
        """Save intermediate results."""
        temp_df = pd.DataFrame(results)
        temp_df.to_excel(self.config.output_path, index=False)


# === Argument Parser ===
def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Collect reasoning traces from thinking-enabled LLMs"
    )
    
    # Required arguments
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input Excel file (original text data from reasoningdataset directory, e.g., reasoningdataset/BookReasoning/128-main-results/Member_BookReasoning_128.xlsx)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to output Excel file for saving extracted reasoning traces and outputs"
    )
    
    # API configuration
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.getenv("API_KEY", "YOUR_API_KEY_HERE"),
        help="API key for the LLM service (default: from API_KEY env variable)"
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="https://api.openai.com/v1/",
        help="Base URL for the API endpoint (default: OpenAI API)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="claude-sonnet-4-thinking",
        help="Model name (e.g., claude-sonnet-4-thinking, gemini-2.5-flash-preview-05-20-thinking)"
    )
    
    # Sampling parameters
    parser.add_argument(
        "--n_responses",
        type=int,
        default=3,
        help="Number of reasoning traces to collect per item (default: 3)"
    )
    parser.add_argument(
        "--sleep_seconds",
        type=float,
        default=1.2,
        help="Sleep duration between API calls in seconds (default: 1.2)"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=10,
        help="Save progress every N items (default: 10)"
    )
    
    # Logging
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    return parser.parse_args()


# === Main Entry Point ===
def main():
    """Main entry point for the script."""
    try:
        # Parse arguments
        args = parse_args()
        
        # Set logging level
        if args.debug:
            logger.setLevel(logging.DEBUG)
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Initialize configuration
        logger.info("Initializing configuration...")
        config = Config(args)
        
        # Initialize components
        logger.info("Initializing trace collector...")
        collector = TraceCollector(config)
        
        logger.info("Initializing data processor...")
        processor = DataProcessor(config, collector)
        
        # Process data
        logger.info("Starting data processing...")
        start_time = time.time()
        result_df = processor.process_excel()
        elapsed_time = time.time() - start_time
        
        # Summary
        logger.info("=" * 60)
        logger.info("Processing completed successfully!")
        logger.info(f"Total items processed: {len(result_df)}")
        logger.info(f"Total time: {elapsed_time:.2f} seconds")
        logger.info(f"Average time per item: {elapsed_time/len(result_df):.2f} seconds")
        logger.info(f"Results saved to: {config.output_path}")
        logger.info("=" * 60)
        
        return 0
    
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        return 130
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())