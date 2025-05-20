#!/usr/bin/env python3
from flask import Flask, send_from_directory, render_template_string
import os

app = Flask(__name__)

# HTML template for the landing page
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Nexus 1.1 - Advanced Autonomous AI Model</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
        }
        h2 {
            color: #3498db;
            margin-top: 30px;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .download-btn {
            display: inline-block;
            background-color: #2ecc71;
            color: white;
            padding: 12px 24px;
            text-decoration: none;
            border-radius: 4px;
            font-weight: bold;
            margin: 20px 0;
            text-align: center;
            transition: background-color 0.3s;
        }
        .download-btn:hover {
            background-color: #27ae60;
        }
        .github-btn {
            display: inline-block;
            background-color: #333;
            color: white;
            padding: 12px 24px;
            text-decoration: none;
            border-radius: 4px;
            font-weight: bold;
            margin: 20px 0;
            text-align: center;
            transition: background-color 0.3s;
        }
        .github-btn:hover {
            background-color: #555;
        }
        pre {
            background-color: #f8f8f8;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
        }
        .feature-list {
            list-style-type: none;
            padding: 0;
        }
        .feature-list li {
            margin-bottom: 10px;
            padding-left: 25px;
            position: relative;
        }
        .feature-list li:before {
            content: "✓";
            color: #2ecc71;
            position: absolute;
            left: 0;
            font-weight: bold;
        }
        .buttons {
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Nexus 1.1</h1>
        <p style="text-align: center; font-style: italic; margin-bottom: 30px;">
            The most super advanced highest level ultra high-tech Autonomous AI model
        </p>
        
        <h2>About Nexus 1.1</h2>
        <p>
            Nexus 1.1 is a cutting-edge autonomous AI model designed for advanced natural language processing tasks. 
            Built with state-of-the-art transformer architecture and enhanced attention mechanisms, it delivers 
            superior performance for text generation, summarization, and contextual understanding.
        </p>
        
        <h2>Key Features</h2>
        <ul class="feature-list">
            <li>Advanced transformer architecture with enhanced attention mechanism</li>
            <li>Multi-modal capabilities for processing text and structured data</li>
            <li>Efficient training pipeline with distributed training support</li>
            <li>Comprehensive API for easy integration</li>
            <li>Web-based UI for interactive model exploration</li>
            <li>Docker support for easy deployment</li>
            <li>Extensive documentation and usage examples</li>
        </ul>
        
        <div class="buttons">
            <a href="/download/nexus_1_1.zip" class="download-btn">Download Nexus 1.1 (ZIP)</a>
            <a href="https://github.com/Arthemax/Arthemax/tree/nexus-1-1" class="github-btn" target="_blank">View on GitHub</a>
        </div>
        
        <h2>Quick Start</h2>
        <pre>
# Install the package
pip install -e .

# Train the model
python run.py train --config-file config/default_config.json

# Generate text
python run.py generate --model-path outputs/best_model.pt --prompt "Hello, world!"

# Run the API server and web UI
python run.py server --model-path outputs/best_model.pt --port 8000
python run.py ui --api-url http://localhost:8000 --port 8001
        </pre>
        
        <h2>Documentation</h2>
        <p>
            For more detailed information, please refer to the documentation included in the download package:
        </p>
        <ul>
            <li><strong>architecture.md</strong> - Detailed description of the model architecture</li>
            <li><strong>api_reference.md</strong> - API documentation for the model</li>
            <li><strong>usage_guide.md</strong> - Guide for using the model</li>
            <li><strong>development_guide.md</strong> - Guide for developing the model</li>
        </ul>
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/download/<path:filename>')
def download_file(filename):
    return send_from_directory('/workspace', filename)

if __name__ == '__main__':
    # Use port 12001 for work-2
    port = 12001
    print(f"Starting server on port {port}...")
    print(f"Access the Nexus 1.1 download page at: https://work-2-vhgvqpfwcvfdmegs.prod-runtime.all-hands.dev")
    print(f"Direct download link: https://work-2-vhgvqpfwcvfdmegs.prod-runtime.all-hands.dev/download/nexus_1_1.zip")
    app.run(host='0.0.0.0', port=port, debug=False)