"""
Web UI for Nexus 1.1
"""

import os
import json
import logging
import argparse
from typing import Dict, Any, List, Optional

import gradio as gr
import requests

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class NexusWebUI:
    """Web UI for Nexus 1.1."""
    
    def __init__(self, api_url: str):
        self.api_url = api_url
        
        # Check if API is running
        try:
            response = requests.get(f"{api_url}/health")
            if response.status_code != 200:
                logger.warning(f"API health check failed: {response.text}")
        except Exception as e:
            logger.warning(f"Could not connect to API: {e}")
    
    def generate_text(self, prompt: str, max_length: int, temperature: float) -> str:
        """Generate text using the API."""
        try:
            response = requests.post(
                f"{self.api_url}/generate",
                json={
                    "prompt": prompt,
                    "max_length": max_length,
                    "temperature": temperature
                }
            )
            
            if response.status_code != 200:
                return f"Error: {response.text}"
            
            return response.json()["generated_text"]
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return f"Error: {str(e)}"
    
    def get_model_info(self) -> str:
        """Get model information from the API."""
        try:
            response = requests.get(f"{self.api_url}/model-info")
            
            if response.status_code != 200:
                return f"Error: {response.text}"
            
            info = response.json()
            return (
                f"Model: Nexus 1.1\n"
                f"Embedding Dimension: {info['embed_dim']}\n"
                f"Number of Layers: {info['num_layers']}\n"
                f"Number of Attention Heads: {info['num_heads']}\n"
                f"Device: {info['device']}"
            )
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return f"Error: {str(e)}"
    
    def create_ui(self):
        """Create Gradio UI."""
        with gr.Blocks(title="Nexus 1.1") as ui:
            gr.Markdown("# Nexus 1.1 - Advanced Autonomous AI Model")
            
            with gr.Row():
                with gr.Column():
                    model_info = gr.Textbox(
                        label="Model Information",
                        value=self.get_model_info(),
                        interactive=False
                    )
            
            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Enter your prompt here...",
                        lines=5
                    )
                    
                    with gr.Row():
                        max_length = gr.Slider(
                            label="Max Length",
                            minimum=10,
                            maximum=500,
                            value=100,
                            step=10
                        )
                        temperature = gr.Slider(
                            label="Temperature",
                            minimum=0.1,
                            maximum=2.0,
                            value=1.0,
                            step=0.1
                        )
                    
                    generate_btn = gr.Button("Generate")
                
                with gr.Column():
                    output = gr.Textbox(
                        label="Generated Text",
                        lines=10
                    )
            
            generate_btn.click(
                fn=self.generate_text,
                inputs=[prompt, max_length, temperature],
                outputs=output
            )
        
        return ui

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Nexus 1.1 Web UI")
    parser.add_argument("--api-url", type=str, default="http://localhost:12000", help="URL of the API server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run server on")
    parser.add_argument("--port", type=int, default=12001, help="Port to run server on")
    parser.add_argument("--share", action="store_true", help="Create a public link")
    
    args = parser.parse_args()
    
    # Create web UI
    web_ui = NexusWebUI(args.api_url)
    ui = web_ui.create_ui()
    
    # Launch UI
    ui.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=["*"],
        show_api=False,
        inbrowser=True
    )

if __name__ == "__main__":
    main()