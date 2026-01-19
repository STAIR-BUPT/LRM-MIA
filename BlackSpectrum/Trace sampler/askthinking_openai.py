#!/usr/bin/env python3
"""
Trace Sampler for OpenAI Reasoning Models (o1/o3)

This script collects multiple reasoning traces and outputs from OpenAI reasoning models
(e.g., o1-preview, o1-mini, o3-mini) for membership inference attack evaluation.

Usage:
    python askthinking_openai.py \
        --input_path /path/to/input.xlsx \
        --output_path /path/to/output.xlsx \
        --api_key YOUR_API_KEY \
        --base_url https://api.openai.com/v1/ \
        --model_name o1-preview \
        --n_responses 2 \
        --sleep_seconds 1.2 \
        --save_interval 10

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
        self.prompt_prefix = args.prompt_prefix
        self.reasoning_effort = args.reasoning_effort
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


# === Response Parser ===
class ResponseParser:
    """Parse reasoning and output from OpenAI API responses."""
    
    @staticmethod
    def extract_summary_and_output(response) -> Tuple[str, str]:
        """
        Extract reasoning summary and final output text from a model response.
        Handles multiple possible response formats for robustness.
        
        Args:
            response: API response object
        
        Returns:
            Tuple of (reasoning_summary, output_text)
        """
        reasoning_summary_parts: List[str] = []
        output_text_parts: List[str] = []

        # Try to extract output_text directly
        try:
            if getattr(response, "output_text", None):
                output_text_parts.append(response.output_text)
        except Exception:
            pass

        # Parse structured output with reasoning and message types
        try:
            for item in getattr(response, "output", []) or []:
                if getattr(item, "type", None) == "reasoning":
                    summary = getattr(item, "summary", None)
                    if isinstance(summary, list):
                        for s in summary:
                            txt = getattr(s, "text", None)
                            if isinstance(txt, str) and txt.strip():
                                reasoning_summary_parts.append(txt)
                    elif isinstance(summary, str) and summary.strip():
                        reasoning_summary_parts.append(summary)

                if getattr(item, "type", None) == "message":
                    for c in getattr(item, "content", []) or []:
                        if getattr(c, "type", None) in ("output_text", "text"):
                            txt = getattr(c, "text", None)
                            if isinstance(txt, str) and txt.strip():
                                output_text_parts.append(txt)
        except Exception:
            pass

        # Try alternative reasoning structure
        try:
            reasoning = getattr(response, "reasoning", None)
            if reasoning:
                summary = getattr(reasoning, "summary", None)
                if isinstance(summary, list):
                    for s in summary:
                        txt = getattr(s, "text", None)
                        if isinstance(txt, str) and txt.strip():
                            reasoning_summary_parts.append(txt)
                elif isinstance(summary, str) and summary.strip():
                    reasoning_summary_parts.append(summary)
        except Exception:
            pass

        return (
            ResponseParser._join_unique(reasoning_summary_parts),
            ResponseParser._join_unique(output_text_parts)
        )
    
    @staticmethod
    def _join_unique(parts: List[str]) -> str:
        """Join parts while preserving order and removing duplicates."""
        seen = set()
        ordered = []
        for p in parts:
            if p not in seen:
                seen.add(p)
                ordered.append(p)
        return "\n".join(ordered).strip()


# === Trace Collector ===
class TraceCollector:
    """Collects reasoning traces from OpenAI reasoning models."""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        self.parser = ResponseParser()
        logger.info(f"Initialized TraceCollector with model: {config.model_name}")
    
    def get_reasoning_paths(self, query: str, n: int = None) -> List[Tuple[str, str]]:
        """
        Sends a query multiple times to obtain distinct reasoning and output pairs.
        
        Args:
            query: The text query to send to the model
            n: Number of responses to collect (default: uses config value)
        
        Returns:
            List of tuples containing (reasoning_summary, output_text)
        """
        if n is None:
            n = self.config.n_responses
        
        results: List[Tuple[str, str]] = []

        for i in range(n):
            retries = 0
            reasoning_summary, output_text = "", ""

            while retries < self.config.max_retries:
                try:
                    resp = self.client.responses.create(
                        model=self.config.model_name,
                        input=query,
                        reasoning={
                            "effort": self.config.reasoning_effort,
                            "summary": "auto"
                        },
                    )

                    reasoning_summary, output_text = self.parser.extract_summary_and_output(resp)

                    # Validate reasoning quality
                    if reasoning_summary.strip() and reasoning_summary.strip().lower() != "detailed":
                        logger.debug(f"Successfully collected reasoning path {i + 1}/{n}")
                        break
                    
                    logger.warning(
                        f"Empty or generic reasoning ('detailed'), "
                        f"retrying ({retries + 1}/{self.config.max_retries})..."
                    )

                except Exception as e:
                    logger.error(
                        f"Error in generation {i + 1} (retry {retries + 1}) "
                        f"for query '{query[:50]}...': {e}"
                    )

                retries += 1
                time.sleep(self.config.sleep_seconds)

            results.append((reasoning_summary, output_text))

        return results


# === Data Processor ===
class DataProcessor:
    """Processes Excel files and collects reasoning traces."""
    
    def __init__(self, config: Config, collector: TraceCollector):
        self.config = config
        self.collector = collector
    
    def process_excel(self) -> pd.DataFrame:
        """
        Processes an Excel file row by row, queries the model,
        and saves reasoning paths and outputs to a new Excel file.
        
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
            query_input = self.config.prompt_prefix + item_text

            logger.info(f"Processing ID {item_id}...")

            try:
                pairs = self.collector.get_reasoning_paths(query_input, n=self.config.n_responses)

                record = {"ID": item_id, "Item": item_text}
                for i, (r_sum, out_txt) in enumerate(pairs, start=1):
                    record[f"Reasoning_Path_{i}"] = r_sum
                    record[f"Output_{i}"] = out_txt
                
                # Fill missing columns if we didn't get all responses
                for i in range(len(pairs) + 1, self.config.n_responses + 1):
                    record[f"Reasoning_Path_{i}"] = ""
                    record[f"Output_{i}"] = ""

                results.append(record)

                # Save progress at intervals
                if (index + 1) % self.config.save_interval == 0:
                    self._save_progress(results)
                    logger.info(f"Saved progress: {index + 1}/{len(df)} items")
            
            except Exception as e:
                logger.error(f"Failed to process item {item_id}: {e}")
                # Add empty entry to maintain data consistency
                record = {"ID": item_id, "Item": item_text}
                for i in range(1, self.config.n_responses + 1):
                    record[f"Reasoning_Path_{i}"] = ""
                    record[f"Output_{i}"] = ""
                results.append(record)

        # Final save
        out_df = pd.DataFrame(results)
        out_df.to_excel(self.config.output_path, index=False)
        logger.info(f"All reasoning paths and outputs saved to: {self.config.output_path}")

        return out_df
    
    def _save_progress(self, results: List[Dict[str, Any]]):
        """Save intermediate results."""
        temp_df = pd.DataFrame(results)
        temp_df.to_excel(self.config.output_path, index=False)


# === Argument Parser ===
def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Collect reasoning traces from OpenAI reasoning models (o1/o3)"
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
        help="API key for OpenAI service (default: from API_KEY env variable)"
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
        default="o1-preview",
        help="Model name (e.g., o1-preview, o1-mini, o3-mini)"
    )
    
    # Sampling parameters
    parser.add_argument(
        "--n_responses",
        type=int,
        default=2,
        help="Number of reasoning traces to collect per item (default: 2)"
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
    parser.add_argument(
        "--max_retries",
        type=int,
        default=5,
        help="Maximum retries per query (default: 5)"
    )
    
    # Model-specific parameters
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default="medium",
        choices=["low", "medium", "high"],
        help="Reasoning effort level for OpenAI models (default: medium)"
    )
    parser.add_argument(
        "--prompt_prefix",
        type=str,
        default="Continue this text based on the given prefix: ",
        help="Prompt prefix to prepend to each item (default: 'Continue this text based on the given prefix: ')"
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