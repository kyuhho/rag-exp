#!/bin/bash
set -e

NUM_SHARDS=4
DATA_PATH="/home/work/Redteaming/rag-exp/ReAct/data/hotpot_dev_v1_simplified.json"
# DATA_PATH="/home/work/Redteaming/rag-exp/ReAct/data/test.json"

SHARD_DATA_PATH="/home/work/Redteaming/rag-exp/corag/data"
TRAJS_PATH="/home/work/Redteaming/rag-exp/results/dataset_results/corag"
OUTPUT_DIR="tmp/generate_trajs"

# Create save directory
mkdir -p ${SHARD_DATA_PATH}
mkdir -p ${TRAJS_PATH}
mkdir -p ${OUTPUT_DIR}

# Split data into shards and save
echo "Splitting data into ${NUM_SHARDS} shards..."
python3 src/split_shards.py \
    --data_path ${DATA_PATH} \
    --num_shards ${NUM_SHARDS} \
    --output_dir ${SHARD_DATA_PATH}

# Kill any existing torchrun processes to avoid port conflicts
echo "Cleaning up any existing torchrun processes..."
pkill -f "torchrun.*generate_trajs" || true
sleep 2

# Run inference for each shard in parallel
for shard_id in $(seq 0 $((NUM_SHARDS - 1))); do
    echo "Starting shard ${shard_id}/${NUM_SHARDS}..."
    CUDA_VISIBLE_DEVICES=${shard_id} \
    VLLM_PORT=$((8000 + shard_id)) \
    torchrun --nproc_per_node=1 --master_port=$((29500 + shard_id)) src/inference/generate_trajs.py \
        --eval_task hotpotqa \
        --eval_split validation \
        --data_path ${SHARD_DATA_PATH}/hotpot_shard_${shard_id}.json \
        --trajs_path ${TRAJS_PATH}/corag_shard_${shard_id}.json \
        --max_path_length 3 \
        --output_dir ${OUTPUT_DIR} \
        --do_eval \
        --decode_strategy greedy \
        --num_shards ${NUM_SHARDS} \
        --shard_id ${shard_id} \
        > ${OUTPUT_DIR}/shard_${shard_id}.log 2>&1 &
done

# wait
# echo "All shards completed!"