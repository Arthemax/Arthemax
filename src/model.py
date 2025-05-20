"""
Core model architecture for Nexus 1.1
"""

import os
import json
import math
import logging
from typing import Dict, List, Union, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

class NexusAttention(nn.Module):
    """Advanced multi-head attention mechanism with dynamic scaling."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Project inputs
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        
        # Final projection
        output = self.out_proj(context)
        
        return output, attn_weights


class NexusFeedForward(nn.Module):
    """Advanced feed-forward network with gated activations."""
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.gate = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(input_dim)
        
    def forward(self, x):
        residual = x
        x = self.norm(x)
        
        gate = torch.sigmoid(self.gate(x))
        x = F.gelu(self.linear1(x))
        x = x * gate
        x = self.dropout(x)
        x = self.linear2(x)
        
        return x + residual


class NexusTransformerBlock(nn.Module):
    """Advanced transformer block with enhanced capabilities."""
    
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attention = NexusAttention(embed_dim, num_heads, dropout)
        self.feed_forward = NexusFeedForward(embed_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        residual = x
        x = self.norm1(x)
        attn_output, _ = self.attention(x, x, x, mask)
        x = residual + self.dropout(attn_output)
        
        # Feed-forward network
        x = self.feed_forward(x)
        
        return x


class NexusModel(nn.Module):
    """
    Nexus 1.1 - Advanced Autonomous AI Model
    
    A cutting-edge neural architecture with self-learning capabilities.
    """
    
    def __init__(
        self,
        vocab_size: int = 50000,
        max_seq_length: int = 1024,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        ff_dim: int = 3072,
        dropout: float = 0.1,
        base_model: str = "bert-base-uncased",
        device: Optional[str] = None
    ):
        super().__init__()
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_seq_length = max_seq_length
        self.embed_dim = embed_dim
        
        # Load pre-trained language model as base
        self.base_model = AutoModel.from_pretrained(base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_seq_length, embed_dim))
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            NexusTransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layers
        self.output_norm = nn.LayerNorm(embed_dim)
        self.output_dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        
        # Initialize parameters
        self._init_parameters()
        
        # Move model to device
        self.to(self.device)
        
        logger.info(f"Initialized Nexus 1.1 model on {self.device}")
        
    def _init_parameters(self):
        """Initialize model parameters."""
        for name, param in self.named_parameters():
            if "base_model" not in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
                
        # Initialize positional encodings
        position = torch.arange(self.max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2) * -(math.log(10000.0) / self.embed_dim))
        pos_enc = torch.zeros(self.max_seq_length, self.embed_dim)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        self.pos_encoding.data = pos_enc.unsqueeze(0)
        
    def forward(self, input_ids, attention_mask=None):
        """Forward pass through the model."""
        batch_size, seq_length = input_ids.size()
        
        # Get embeddings from base model
        with torch.no_grad():
            base_outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            hidden_states = base_outputs.hidden_states[-1]
        
        # Add positional encodings
        if seq_length <= self.max_seq_length:
            pos_enc = self.pos_encoding[:, :seq_length, :]
            x = hidden_states + pos_enc
        else:
            logger.warning(f"Input sequence length {seq_length} exceeds maximum {self.max_seq_length}")
            x = hidden_states + self.pos_encoding[:, :self.max_seq_length, :][:, :seq_length, :]
        
        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, attention_mask)
        
        # Apply output layers
        x = self.output_norm(x)
        x = self.output_dropout(x)
        logits = self.output_projection(x)
        
        return logits
    
    def generate(self, input_text: str, max_length: int = 100, temperature: float = 1.0):
        """Generate text from the model."""
        self.eval()
        
        # Tokenize input text
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Generate tokens
        for _ in range(max_length):
            with torch.no_grad():
                outputs = self(input_ids, attention_mask)
                next_token_logits = outputs[:, -1, :] / temperature
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(next_token_probs, num_samples=1)
                
                # Append next token
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                attention_mask = torch.cat([
                    attention_mask, 
                    torch.ones((1, 1), device=self.device)
                ], dim=-1)
                
                # Check if EOS token is generated
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        # Decode generated tokens
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        return generated_text
    
    def save(self, path: str):
        """Save model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state
        torch.save({
            "model_state_dict": self.state_dict(),
            "config": {
                "vocab_size": self.embedding.num_embeddings,
                "max_seq_length": self.max_seq_length,
                "embed_dim": self.embed_dim,
                "num_heads": self.transformer_blocks[0].attention.num_heads,
                "num_layers": len(self.transformer_blocks),
                "ff_dim": self.transformer_blocks[0].feed_forward.linear1.out_features,
            }
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, device: Optional[str] = None):
        """Load model from disk."""
        checkpoint = torch.load(path, map_location=device or "cpu")
        
        # Create model with saved config
        model = cls(**checkpoint["config"], device=device)
        
        # Load state dict
        model.load_state_dict(checkpoint["model_state_dict"])
        
        logger.info(f"Model loaded from {path}")
        
        return model
    
    def train_step(self, input_ids, attention_mask, labels, optimizer):
        """Perform a single training step."""
        self.train()
        
        # Forward pass
        logits = self(input_ids, attention_mask)
        
        # Compute loss
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()