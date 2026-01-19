# Exploring MIA on LRMs


## Installation

```bash
pip install -r requirements.txt
```

## Extract Reasoning Traces

Use the scripts in `BlackSpectrum/Trace sampler/` to extract reasoning traces from various LLMs

**Pre-extracted Examples**: We provide Claude Sonnet 4 reasoning trace examples in `BlackSpectrum/reasoning_trace_example_results/` for a quick start.


## BlackSpectrum Quick Start

### Step 1: Build Projection Scores

```bash
python build_axis_projection_score.py \
    --encoder <model_path> \
    --generalization_data <generalization_anchor.xlsx> \
    --memorization_data <memorization_anchor.xlsx> \
    --query_data <test_samples.xlsx> \
    --output <output_scores.xlsx>
```

### Step 2: Evaluate Performance

Outputs both **sequence-level** and **document-level** evaluation results.

```bash
python evaluation_predictor.py \
    --member_data <member_scores.xlsx> \
    --nonmember_data <nonmember_scores.xlsx> \
    --output <evaluation_results.xlsx>
```

## Complete Example

Run the quick example script:

```bash
sh example_blackspectrum_run.sh
```

This script will:
1. Process member data with projection scoring
2. Process non-member data with projection scoring
3. Evaluate and output results to `BlackSpectrum/outputs/`


## Command Options

### build_axis_projection_score.py

| Option | Required | Description |
|--------|----------|-------------|
| `--generalization_data` | Yes | Generalization anchor file |
| `--memorization_data` | Yes | Memorization anchor file |
| `--query_data` | Yes | Query/test data file |
| `--output` | Yes | Output file path |
| `--encoder` | No | Model name or local path |
| `--device` | No | Device: `cuda`, `cuda:0`, `0`, `1`, or `cpu` (auto-detect if not specified) |
| `--z_thresh` | No | Outlier removal threshold (default: 2.0) |

### evaluation_predictor.py

Outputs two evaluation levels:
- **Sequence-level**: Evaluate each reasoning path independently
- **Document-level**: Aggregate by ID, evaluate per document

| Option | Required | Description |
|--------|----------|-------------|
| `--member_data` | Yes | Member scores file |
| `--nonmember_data` | Yes | Non-member scores file |
| `--output` | No | Results output (saves combined and detailed sheets) |


## Output Files

When `--output` is specified, `evaluation_predictor.py` generates:
- `<output>.xlsx`: Combined results (sequence + document level)
- `<output>_detailed.xlsx`: Separate sheets for each level

## Available Models

**encoder: pre-training model**:
- `sentence-transformers/all-distilroberta-v1`
- `sentence-transformers/all-mpnet-base-v2`
- `sentence-transformers/all-MiniLM-L6-v2`



## Naive Baseline Quick Start

### Step 1: Analyze Thinking Tokens & Compression Rate

```bash
python naive_attack_baseline/naive_thinkingtoken_and_compressionrate.py \
    -i <input_data.xlsx> \
    -o <output_analysis.xlsx>
```

### Step 2: Evaluate Baseline Performance

Outputs both **sequence-level** and **document-level** evaluation results.

```bash
python naive_attack_baseline/evaluation_naive_baselines.py \
    -m <member_analysis.xlsx> \
    -n <nonmember_analysis.xlsx> \
    -o <evaluation_results.xlsx>
```

## Complete Example

Run the example script to execute:

```bash
sh example_naive_thinkingtoken_and_compressionrate_run.sh
```

This script will:
1. Analyze member data (thinking tokens & compression rate)
2. Analyze non-member data (thinking tokens & compression rate)
3. Evaluate and output results to `naive_attack_baseline/outputs/`

## Command Options

### naive_thinkingtoken_and_compressionrate.py

| Option | Required | Description |
|--------|----------|-------------|
| `-i, --input` | Yes | Input data file with reasoning traces |
| `-o, --output` | Yes | Output analysis file path |
| `--device` | No | Device: `cuda`, `cuda:0`, `0`, `1`, `cpu`, or `auto` (default: auto) |
| `--gpt2-model` | No | GPT-2 model path (default: gpt2) |
| `--embedding-model` | No | Embedding model path (default: sentence-transformers/all-MiniLM-L6-v2) |
| `--threshold` | No | Similarity threshold (default: 0.8) |

### evaluation_naive_baselines.py

Outputs two evaluation levels:
- **Sequence-level**: Evaluate each reasoning path independently
- **Document-level**: Aggregate by ID, evaluate per document

| Option | Required | Description |
|--------|----------|-------------|
| `-m, --member_data` | Yes | Member analysis file |
| `-n, --nonmember_data` | Yes | Non-member analysis file |
| `-o, --output` | No | Results output (saves combined and detailed sheets) |

## Output Files

When `-o` is specified, `evaluation_naive_baselines.py` generates:
- `<output>.xlsx`: Combined results (sequence + document level)
- `<output>_detailed.xlsx`: Separate sheets for each level
