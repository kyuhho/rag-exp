#!/usr/bin/env bash

# Stop E5 Search Server (Port 8090)
PORT=8090
PID=$(lsof -t -i:$PORT)

if [ -z "$PID" ]; then
    echo "No E5 search server found running on port $PORT"
else
    echo "Stopping E5 search server (PID: $PID)..."
    kill -9 $PID
    echo "Done."
fi
