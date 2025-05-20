#!/usr/bin/env python
"""
Run script for Nexus 1.1
"""

import os
import sys
import argparse
import subprocess

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Nexus 1.1 - Advanced Autonomous AI Model")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train Nexus 1.1 model")
    train_parser.add_argument("--config-file", type=str, help="Path to config file")
    train_parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file")
    
    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate text with Nexus 1.1 model")
    generate_parser.add_argument("--model-path", type=str, help="Path to model file")
    generate_parser.add_argument("--prompt", type=str, help="Prompt for text generation")
    
    # Prepare data command
    data_parser = subparsers.add_parser("prepare-data", help="Prepare sample data for Nexus 1.1 model")
    data_parser.add_argument("--output-dir", type=str, default="data", help="Output directory")
    data_parser.add_argument("--num-samples", type=int, default=100, help="Number of samples to generate")
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Run Nexus 1.1 API server")
    server_parser.add_argument("--model-path", type=str, help="Path to model file")
    server_parser.add_argument("--port", type=int, default=12000, help="Port to run server on")
    
    # Web UI command
    ui_parser = subparsers.add_parser("ui", help="Run Nexus 1.1 web UI")
    ui_parser.add_argument("--api-url", type=str, default="http://localhost:12000", help="URL of the API server")
    ui_parser.add_argument("--port", type=int, default=12001, help="Port to run UI on")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run command
    if args.command == "train":
        cmd = [sys.executable, "-m", "src.main", "train"]
        if args.config_file:
            cmd.extend(["--config-file", args.config_file])
        if args.checkpoint:
            cmd.extend(["--checkpoint", args.checkpoint])
    elif args.command == "generate":
        cmd = [sys.executable, "-m", "src.main", "generate"]
        if args.model_path:
            cmd.extend(["--model-path", args.model_path])
        if args.prompt:
            cmd.extend(["--prompt", args.prompt])
    elif args.command == "prepare-data":
        cmd = [sys.executable, "-m", "src.main", "prepare-data"]
        if args.output_dir:
            cmd.extend(["--output-dir", args.output_dir])
        if args.num_samples:
            cmd.extend(["--num-samples", str(args.num_samples)])
    elif args.command == "server":
        cmd = [sys.executable, "-m", "src.server"]
        if args.model_path:
            cmd.extend(["--model-path", args.model_path])
        if args.port:
            cmd.extend(["--port", str(args.port)])
    elif args.command == "ui":
        cmd = [sys.executable, "-m", "src.web_ui"]
        if args.api_url:
            cmd.extend(["--api-url", args.api_url])
        if args.port:
            cmd.extend(["--port", str(args.port)])
    else:
        parser.print_help()
        return
    
    # Execute command
    subprocess.run(cmd)

if __name__ == "__main__":
    main()