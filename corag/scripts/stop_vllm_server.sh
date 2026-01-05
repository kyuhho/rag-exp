#!/usr/bin/env bash

# Stop vLLM Server (Port 8000)
PORT=8000
PID=$(lsof -t -i:$PORT)

if [ -z "$PID" ]; then
    echo "No vLLM server found running on port $PORT"
else
    echo "Stopping vLLM server (PID: $PID)..."
    # Kill the main process and all its children to ensure GPU memory release
    pkill -9 -P $PID
    kill -9 $PID
    echo "Done. Please check nvidia-smi to ensure memory is released."
fi
