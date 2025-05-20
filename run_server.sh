#!/bin/bash

# Run Nexus 1.1 API server and web UI

# Default ports
API_PORT=12000
UI_PORT=12001

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --api-port)
      API_PORT="$2"
      shift 2
      ;;
    --ui-port)
      UI_PORT="$2"
      shift 2
      ;;
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check if model path is provided
if [ -z "$MODEL_PATH" ]; then
  echo "Model path not provided. Using dummy model."
  # Create dummy model directory
  mkdir -p models
  touch models/dummy_model.pt
  MODEL_PATH="models/dummy_model.pt"
fi

# Start API server
echo "Starting API server on port $API_PORT..."
python -m src.server --model-path "$MODEL_PATH" --port "$API_PORT" &
API_PID=$!

# Wait for API server to start
echo "Waiting for API server to start..."
sleep 5

# Start web UI
echo "Starting web UI on port $UI_PORT..."
python -m src.web_ui --api-url "http://localhost:$API_PORT" --port "$UI_PORT" &
UI_PID=$!

# Wait for web UI to start
echo "Waiting for web UI to start..."
sleep 5

echo "Nexus 1.1 is running!"
echo "API server: http://localhost:$API_PORT"
echo "Web UI: http://localhost:$UI_PORT"

# Wait for Ctrl+C
echo "Press Ctrl+C to stop..."
trap "kill $API_PID $UI_PID; exit" INT
wait