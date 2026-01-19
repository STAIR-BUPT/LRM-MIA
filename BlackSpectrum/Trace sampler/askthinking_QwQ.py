#!/usr/bin/env python3
"""
Trace Sampler for QwQ and DeepSeek-R1 Models

This script collects multiple reasoning traces and outputs from QwQ/DeepSeek-R1 models
for membership inference attack evaluation.

Usage:
    python askthinking_QwQ.py \
        --input_path /path/to/input.xlsx \
        --output_path /path/to/output.xlsx \
        --api_key YOUR_API_KEY \
        --base_url https://api.example.com/v1 \
        --model_name Qwen/QwQ-32B \
        --n_responses 3 \
        --sleep_seconds 1.2 \
        --save_interval 50

"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any

import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE


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
        self.max_retries = args.max_retries
        
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


# === Utility Functions ===
class TextCleaner:
    """Clean text for Excel compatibility."""
    
    @staticmethod
    def safe_clean(value):
        """Remove characters incompatible with Excel."""
        try:
            if isinstance(value, str):
                value = ILLEGAL_CHARACTERS_RE.sub("", value)
                value = value.replace("\u2028", "").replace("\u2029", "")
                return value.strip()
            return value
        except Exception:
            return ""


# === Trace Collector ===
class TraceCollector:
    """Collects reasoning traces from QwQ/DeepSeek-R1 models."""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        logger.info(f"Initialized TraceCollector with model: {config.model_name}")
    
    def get_reasoning_paths(self, query: str, n: int = None) -> List[Tuple[str, str]]:
        """
        Generate N reasoning paths for a given query.
        
        Args:
            query: The text query to send to the model
            n: Number of responses to collect (default: uses config value)
        
        Returns:
            List of tuples containing (reasoning_content, output_content)
        """
        if n is None:
            n = self.config.n_responses
        
        results = []
        attempt_counts = [0] * n
        i = 0

        while i < n:
            if attempt_counts[i] >= self.config.max_retries:
                logger.warning(f"Response {i+1} reached max retries ({self.config.max_retries}), skipping.")
                results.append(("", ""))
                i += 1
                continue

            try:
                logger.debug(f"Generating response {i+1}/{n} from model...")
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {
                            "role": "user",
                            "content": (
                                "Consider this passage. First, reflect on whether you have encountered it before. "
                                "Next, try to identify its source — such as a book, article, website, or dataset. "
                                f"Finally, answer: What is the next word: {query}"
                            ),
                        },
                    ],
                    stream=False,
                )

                choice = response.choices[0].message
                reasoning = getattr(choice, "reasoning_content", "").strip()
                output = choice.content.strip()

                if reasoning:
                    results.append((reasoning, output))
                    logger.debug(f"Successfully collected reasoning path {i+1}/{n}")
                    i += 1
                else:
                    attempt_counts[i] += 1
                    logger.warning(
                        f"Response {i+1} missing reasoning content; "
                        f"retrying ({attempt_counts[i]}/{self.config.max_retries})"
                    )

            except Exception as e:
                attempt_counts[i] += 1
                logger.error(
                    f"Error in generation {i+1} (attempt {attempt_counts[i]}/{self.config.max_retries}): {e}"
                )

            time.sleep(self.config.sleep_seconds)

        return results


# === Data Processor ===
class DataProcessor:
    """Processes Excel files and collects reasoning traces."""
    
    def __init__(self, config: Config, collector: TraceCollector):
        self.config = config
        self.collector = collector
        self.cleaner = TextCleaner()
    
    def process_excel(self) -> pd.DataFrame:
        """
        Read input Excel, call model, and save outputs with reasoning paths.
        
        Returns:
            DataFrame containing all results
        """
        logger.info(f"Loading input file: {self.config.input_path}")
        try:
            df_input = pd.read_excel(self.config.input_path)
        except Exception as e:
            logger.error(f"Failed to read input file: {e}")
            raise
        
        logger.info(f"Loaded {len(df_input)} items to process")
        
        all_results = []
        new_batch = []

        for index, row in tqdm(df_input.iterrows(), total=len(df_input), desc="Processing items"):
            try:
                item_text = str(row.get("Item", "")).strip()
                item_id = str(row.get("ID", index))

                if not item_text:
                    logger.warning(f"Row {item_id} is empty, skipping.")
                    continue

                logger.info(f"Processing ID={item_id} | Preview: {item_text[:40]}...")

                responses = self.collector.get_reasoning_paths(item_text, n=self.config.n_responses)

                result_entry = {"ID": item_id, "Item": item_text}
                for j, (reasoning, output) in enumerate(responses, start=1):
                    result_entry[f"Reasoning_Path_{j}"] = reasoning
                    result_entry[f"Output_{j}"] = output
                
                # Fill missing columns if we didn't get all responses
                for j in range(len(responses) + 1, self.config.n_responses + 1):
                    result_entry[f"Reasoning_Path_{j}"] = ""
                    result_entry[f"Output_{j}"] = ""

                # Clean text for Excel compatibility
                cleaned_result = {k: self.cleaner.safe_clean(v) for k, v in result_entry.items()}

                # Test write to detect Excel encoding issues early
                try:
                    _ = pd.DataFrame([cleaned_result]).to_excel("/dev/null", engine="openpyxl")
                except Exception as e:
                    logger.warning(f"ID={item_id} contains invalid characters, skipped. Error: {e}")
                    continue

                all_results.append(cleaned_result)
                new_batch.append(cleaned_result)

                # Save batch
                if len(new_batch) >= self.config.save_interval:
                    self._save_progress(all_results)
                    logger.info(f"Saved progress: {len(all_results)}/{len(df_input)} items")
                    new_batch = []

            except Exception as e:
                logger.error(f"Skipping ID={item_id} due to processing error: {e}")
                continue

        # Final save
        if all_results:
            result_df = self._save_final(all_results)
            logger.info(f"Final save complete: {len(all_results)} rows processed")
            return result_df
        else:
            logger.warning("No results to save")
            return pd.DataFrame()
    
    def _save_progress(self, results: List[Dict[str, Any]]):
        """Save intermediate results."""
        try:
            df_batch = pd.DataFrame(results).applymap(self.cleaner.safe_clean)
            df_batch.to_excel(self.config.output_path, index=False, engine="openpyxl")
        except Exception as e:
            logger.error(f"Batch save failed: {e}")
    
    def _save_final(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Save final results."""
        try:
            df_final = pd.DataFrame(results).applymap(self.cleaner.safe_clean)
            df_final.to_excel(self.config.output_path, index=False, engine="openpyxl")
            return df_final
        except Exception as e:
            logger.error(f"Final save failed: {e}")
            raise


# === Argument Parser ===
def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Collect reasoning traces from QwQ/DeepSeek-R1 models"
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
        default="https://api.openai.com/v1",
        help="Base URL for the API endpoint (default: OpenAI API)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/QwQ-32B",
        help="Model name (e.g., Qwen/QwQ-32B, deepseek-ai/DeepSeek-R1)"
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
        default=50,
        help="Save progress every N items (default: 50)"
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=10,
        help="Maximum retries per response (default: 10)"
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
        if len(result_df) > 0:
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
