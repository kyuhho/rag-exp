#!/usr/bin/env bash

PORT=${VLLM_PORT:-8000}  # Use VLLM_PORT env var if set, otherwise default to 8000

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

# Find vllm serve process with matching port
PID=""
ALL_VLLM_PIDS=$(ps aux | grep "[v]llm serve" | awk '{print $2}')

for VLLM_PID in $ALL_VLLM_PIDS; do
    PROC_CMD=$(ps -p $VLLM_PID -o args= 2>/dev/null)
    if [ -n "$PROC_CMD" ] && echo "$PROC_CMD" | grep -q "vllm serve"; then
        # Check if this process is using our port (using sed instead of grep -oP)
        PROC_PORT=$(echo "$PROC_CMD" | sed -n 's/.*--port[[:space:]]\+\([0-9]\+\).*/\1/p')
        if [ "$PROC_PORT" = "$PORT" ]; then
            PID=$VLLM_PID
            break
        fi
    fi
done

if [ -z "$PID" ]; then
    echo "No vLLM server found running on port $PORT"
else
    echo "Stopping vLLM server on port $PORT (PID: $PID)..."
    # Kill the main process and all its children to ensure GPU memory release
    pkill -9 -P $PID 2>/dev/null
    kill -9 $PID 2>/dev/null
    echo "Done. Please check nvidia-smi to ensure memory is released."
fi
