#!/bin/bash

# Exit on error
set -e

# echo "=== Step 1: Downloading Base Datasets (BEIR) ==="
# python prepare_dataset.py

echo -e "\n=== Step 2: Preparing Poisoned Corpus ==="
python scripts/prepare_poisoned_data.py

echo -e "\n=== Step 3: Building Clean E5 Index for HotpotQA ==="
# Building index for the full corpus might take a long time and lots of disk space/RAM
# Using default batch_size and shard_size from the script
python scripts/build_index.py \
    --corpus_path datasets/hotpotqa/corpus.jsonl \
    --output_dir datasets/hotpotqa/e5_index \
    --model_name /home/work/Redteaming/data1/VIDEO_HALLUCINATION/hf_cache/hub/models--intfloat--e5-large-v2/snapshots/f169b11e22de13617baa190a028a32f3493550b6

echo -e "\n=== Step 4: Building Poisoned E5 Index for HotpotQA ==="
python scripts/build_index.py \
    --corpus_path datasets/hotpotqa/poisoned_corpus.jsonl \
    --output_dir datasets/hotpotqa/e5_index_poisoned \
    --model_name /home/work/Redteaming/data1/VIDEO_HALLUCINATION/hf_cache/hub/models--intfloat--e5-large-v2/snapshots/f169b11e22de13617baa190a028a32f3493550b6


echo -e "\n=== Step 5: Mapping QID to Indices for ReAct Evaluation ==="
python scripts/map_qid_to_idx.py

echo -e "\n=== Data Preparation Completed Successfully! ==="
echo "Indices and mapping files are ready for evaluation."
