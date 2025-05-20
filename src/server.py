"""
Web server for Nexus 1.1
"""

import os
import json
import logging
import argparse
from typing import Dict, Any, List, Optional

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from .model import NexusModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Nexus 1.1 API",
    description="API for Nexus 1.1 - Advanced Autonomous AI Model",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model = None

class GenerateRequest(BaseModel):
    """Request model for text generation."""
    
    prompt: str = Field(..., description="Input prompt for text generation")
    max_length: int = Field(100, description="Maximum length of generated text")
    temperature: float = Field(1.0, description="Temperature for sampling")

class GenerateResponse(BaseModel):
    """Response model for text generation."""
    
    generated_text: str = Field(..., description="Generated text")

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Nexus 1.1 API is running"}

@app.get("/health")
async def health():
    """Health check endpoint."""
    if model is None:
        return {"status": "error", "message": "Model not loaded"}
    return {"status": "ok", "message": "Model is loaded and ready"}

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text endpoint."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Generate text
        generated_text = model.generate(
            input_text=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature
        )
        
        return {"generated_text": generated_text}
    except Exception as e:
        logger.error(f"Error generating text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
async def model_info():
    """Get model information."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_name": "Nexus 1.1",
        "embed_dim": model.embed_dim,
        "num_layers": len(model.transformer_blocks),
        "num_heads": model.transformer_blocks[0].attention.num_heads,
        "device": str(next(model.parameters()).device)
    }

def load_model(model_path: str):
    """Load model from disk."""
    global model
    
    try:
        logger.info(f"Loading model from {model_path}")
        model = NexusModel.load(model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Nexus 1.1 API Server")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model file")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run server on")
    parser.add_argument("--port", type=int, default=12000, help="Port to run server on")
    
    args = parser.parse_args()
    
    # Load model
    load_model(args.model_path)
    
    # Run server
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()