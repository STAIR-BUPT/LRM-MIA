#!/bin/bash

# Naive Baseline Example Run Script
# This script demonstrates the workflow for thinking token and compression rate baseline

# Create output directory
mkdir -p naive_attack_baseline/outputs

# Step 1: Analyze Member data
echo "=== Step 1/3: Analyzing Member data ==="
python naive_attack_baseline/naive_thinkingtoken_and_compressionrate.py \
    --device 0 \
    -i BlackSpectrum/reasoning_trace_example_results/Member128_BookReasoning_claude-sonnet-4-20250514-thinking.xlsx \
    -o naive_attack_baseline/outputs/Member128_naive_baseline.xlsx

if [ $? -ne 0 ]; then
    echo "Error: Failed to analyze member data"
    exit 1
fi

# Step 2: Analyze Non-member data
echo "=== Step 2/3: Analyzing Non-member data ==="
python naive_attack_baseline/naive_thinkingtoken_and_compressionrate.py \
    --device 0 \
    -i BlackSpectrum/reasoning_trace_example_results/Nonmember128_BookReasoning_claude-sonnet-4-20250514-thinking.xlsx \
    -o naive_attack_baseline/outputs/Nonmember128_naive_baseline.xlsx

if [ $? -ne 0 ]; then
    echo "Error: Failed to analyze non-member data"
    exit 1
fi

# Step 3: Evaluate baseline results
echo "=== Step 3/3: Evaluating baseline results ==="
python naive_attack_baseline/evaluation_naive_baselines.py \
    -m naive_attack_baseline/outputs/Member128_naive_baseline.xlsx \
    -n naive_attack_baseline/outputs/Nonmember128_naive_baseline.xlsx \
    -o naive_attack_baseline/outputs/naive_evaluation_results.xlsx

if [ $? -ne 0 ]; then
    echo "Error: Failed to evaluate baseline results"
    exit 1
fi

echo "=== All steps completed successfully ==="
echo "Results saved in naive_attack_baseline/outputs/"
