#!/bin/bash

echo "Restarting Nexus 1.1 servers..."

# Kill existing servers
echo "Stopping existing servers..."
pkill -f "python /workspace/flask_server.py" || true
pkill -f "python /workspace/flask_server_2.py" || true

# Wait a moment for the servers to stop
sleep 2

# Start server 1
echo "Starting server 1 (Port 12000)..."
nohup python /workspace/flask_server.py > /workspace/flask_server.log 2>&1 &

# Wait a moment
sleep 2

# Start server 2
echo "Starting server 2 (Port 12001)..."
nohup python /workspace/flask_server_2.py > /workspace/flask_server_2.log 2>&1 &

# Wait a moment
sleep 2

# Check if servers are running
echo ""
echo "Checking server status..."
/workspace/check_servers.sh