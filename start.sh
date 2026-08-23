#!/bin/bash
set -e
cd "$(dirname "$0")"

if ! pgrep -f "ollama serve" > /dev/null; then
    echo "Starting Ollama..."
    nohup ollama serve > ~/Library/Logs/ollama.log 2>&1 &
    disown
    sleep 2
else
    echo "Ollama already running."
fi

.venv/bin/python main.py
