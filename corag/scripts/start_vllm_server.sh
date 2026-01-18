#!/usr/bin/env bash

set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd ../ && pwd )"
echo "working directory: ${DIR}"

MODEL_NAME_OR_PATH="/home/work/Redteaming/data1/VIDEO_HALLUCINATION/hf_cache/hub/models--corag--CoRAG-Llama3.1-8B-MultihopQA/snapshots/7ccbc2b805e313764f1db06031ad795e5593235b"
PORT=${VLLM_PORT:-8000}  # Use VLLM_PORT env var if set, otherwise default to 8000
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    MODEL_NAME_OR_PATH=$1
    shift
fi

# Check for --port argument (overrides env var)
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT=$2
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

if nc -z localhost ${PORT}; then
  echo "VLLM server already running on port $PORT."
else
  echo "Starting VLLM server on port $PORT..."

  if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
      PROC_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
      echo "CUDA_VISIBLE_DEVICES detected. Using ${PROC_PER_NODE} GPUs."
      export CUDA_VISIBLE_DEVICES
  else
      PROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
      echo "Using all ${PROC_PER_NODE} GPUs."
  fi
  vllm serve "${MODEL_NAME_OR_PATH}" \
    --dtype auto \
    --disable-log-requests --disable-custom-all-reduce \
    --enable_chunked_prefill --max_num_batched_tokens 2048 \
    --tensor-parallel-size "${PROC_PER_NODE}" \
    --max-model-len 8192 \
    --gpu_memory_utilization 0.5 \
    --api-key token-123 \
    --port $PORT > vllm_server_${PORT}.log 2>&1 &

  elapsed=0
  while ! nc -z localhost ${PORT}; do
    sleep 10
    elapsed=$((elapsed + 10))
    if [ $elapsed -ge 600 ]; then
      echo "Server did not start within 10 minutes. Exiting."
      exit 1
    fi
  done
  echo "VLLM server started on port $PORT."
fi
