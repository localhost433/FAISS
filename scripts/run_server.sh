#!/bin/bash

# Fail immediately if any command fails
set -e

# Navigate to backend directory
cd "$(dirname "$0")/../backend"

# Activate virtual environment
if [ -d "venv" ]; then
    echo "Activating virtual environment from ./venv..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "Activating virtual environment from ./.venv..."
    source .venv/bin/activate
else
    echo "No virtual environment found. Please create one in backend/."
    exit 1
fi

# Launch FastAPI server
echo "Starting FastAPI server..."
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000