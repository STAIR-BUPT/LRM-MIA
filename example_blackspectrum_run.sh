#!/bin/bash

# BlackSpectrum Example Run Script
# This script demonstrates the complete workflow for Axis Projection Score MIA

cd BlackSpectrum/axis_builder_and_predictor

# Create output directory
mkdir -p ../outputs

# Step 1: Process Member data
echo "=== Step 1/3: Processing Member data ==="
python build_axis_projection_score.py \
    --encoder sentence-transformers/all-distilroberta-v1 \
    --device 1 \
    --generalization_data "../extreme samples example/extreme_set_with_traces/generalization_proverb_syn_prefixes_top_claude-sonnet-4-20250514-thinking.xlsx" \
    --memorization_data "../extreme samples example/extreme_set_with_traces/memorized_proverb_prefixes_claude-sonnet-4-20250514-thinking.xlsx" \
    --query_data "../reasoning_trace_example_results/Member128_BookReasoning_claude-sonnet-4-20250514-thinking.xlsx" \
    --output ../outputs/Member128_projection_scores.xlsx

if [ $? -ne 0 ]; then
    echo "Error: Failed to process member data"
    exit 1
fi

# Step 2: Process Non-member data
echo "=== Step 2/3: Processing Non-member data ==="
python build_axis_projection_score.py \
    --encoder sentence-transformers/all-distilroberta-v1 \
    --device 1 \
    --generalization_data "../extreme samples example/extreme_set_with_traces/generalization_proverb_syn_prefixes_top_claude-sonnet-4-20250514-thinking.xlsx" \
    --memorization_data "../extreme samples example/extreme_set_with_traces/memorized_proverb_prefixes_claude-sonnet-4-20250514-thinking.xlsx" \
    --query_data "../reasoning_trace_example_results/Nonmember128_BookReasoning_claude-sonnet-4-20250514-thinking.xlsx" \
    --output ../outputs/Nonmember128_projection_scores.xlsx

if [ $? -ne 0 ]; then
    echo "Error: Failed to process non-member data"
    exit 1
fi

# Step 3: Evaluate results
echo "=== Step 3/3: Evaluating results ==="
python evaluation_predictor.py \
    --member_data ../outputs/Member128_projection_scores.xlsx \
    --nonmember_data ../outputs/Nonmember128_projection_scores.xlsx \
    --output ../outputs/evaluation_results.xlsx

if [ $? -ne 0 ]; then
    echo "Error: Failed to evaluate results"
    exit 1
fi

echo "=== All steps completed successfully ==="
echo "Results saved in BlackSpectrum/outputs/"
