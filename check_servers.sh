#!/bin/bash

echo "Checking Nexus 1.1 servers..."
echo ""

# Check server 1
echo "Server 1 (Port 12000):"
if pgrep -f "python /workspace/flask_server.py" > /dev/null; then
    echo "✅ RUNNING"
    echo "Access URL: https://work-1-vhgvqpfwcvfdmegs.prod-runtime.all-hands.dev"
    echo "Download URL: https://work-1-vhgvqpfwcvfdmegs.prod-runtime.all-hands.dev/download/nexus_1_1.zip"
else
    echo "❌ NOT RUNNING"
fi

echo ""

# Check server 2
echo "Server 2 (Port 12001):"
if pgrep -f "python /workspace/flask_server_2.py" > /dev/null; then
    echo "✅ RUNNING"
    echo "Access URL: https://work-2-vhgvqpfwcvfdmegs.prod-runtime.all-hands.dev"
    echo "Download URL: https://work-2-vhgvqpfwcvfdmegs.prod-runtime.all-hands.dev/download/nexus_1_1.zip"
else
    echo "❌ NOT RUNNING"
fi

echo ""
echo "GitHub Repository: https://github.com/Arthemax/Arthemax/tree/nexus-1-1"
echo ""

# Check if the zip file exists
if [ -f "/workspace/nexus_1_1.zip" ]; then
    echo "ZIP file: ✅ EXISTS ($(du -h /workspace/nexus_1_1.zip | cut -f1))"
else
    echo "ZIP file: ❌ MISSING"
fi