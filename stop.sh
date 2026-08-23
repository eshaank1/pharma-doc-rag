#!/bin/bash
pkill -f "main.py" && echo "Stopped app." || echo "App wasn't running."
pkill -f "ollama serve" && echo "Stopped Ollama." || echo "Ollama wasn't running."
