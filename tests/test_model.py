"""
Tests for Nexus 1.1 model
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import NexusModel, NexusAttention, NexusTransformerBlock

class TestNexusModel(unittest.TestCase):
    """Test cases for Nexus 1.1 model."""
    
    @patch("src.model.AutoModel")
    @patch("src.model.AutoTokenizer")
    def test_model_initialization(self, mock_tokenizer, mock_auto_model):
        """Test model initialization."""
        # Mock AutoModel and AutoTokenizer
        mock_auto_model.from_pretrained.return_value = MagicMock()
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        
        # Initialize model
        model = NexusModel(
            vocab_size=1000,
            max_seq_length=128,
            embed_dim=256,
            num_heads=4,
            num_layers=2,
            ff_dim=512,
            dropout=0.1,
            base_model="bert-base-uncased",
            device="cpu"
        )
        
        # Check model attributes
        self.assertEqual(model.max_seq_length, 128)
        self.assertEqual(model.embed_dim, 256)
        self.assertEqual(len(model.transformer_blocks), 2)
        self.assertEqual(model.transformer_blocks[0].attention.num_heads, 4)
        
        # Check if AutoModel and AutoTokenizer were called
        mock_auto_model.from_pretrained.assert_called_once_with("bert-base-uncased")
        mock_tokenizer.from_pretrained.assert_called_once_with("bert-base-uncased")
    
    @patch("src.model.AutoModel")
    @patch("src.model.AutoTokenizer")
    def test_attention_mechanism(self, mock_tokenizer, mock_auto_model):
        """Test attention mechanism."""
        # Create attention module
        attention = NexusAttention(embed_dim=256, num_heads=4)
        
        # Create random inputs
        batch_size = 2
        seq_length = 10
        embed_dim = 256
        
        query = torch.randn(batch_size, seq_length, embed_dim)
        key = torch.randn(batch_size, seq_length, embed_dim)
        value = torch.randn(batch_size, seq_length, embed_dim)
        
        # Forward pass
        output, attention_weights = attention(query, key, value)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_length, embed_dim))
        self.assertEqual(attention_weights.shape, (batch_size, 4, seq_length, seq_length))
    
    @patch("src.model.AutoModel")
    @patch("src.model.AutoTokenizer")
    def test_transformer_block(self, mock_tokenizer, mock_auto_model):
        """Test transformer block."""
        # Create transformer block
        transformer_block = NexusTransformerBlock(
            embed_dim=256,
            num_heads=4,
            ff_dim=512,
            dropout=0.1
        )
        
        # Create random inputs
        batch_size = 2
        seq_length = 10
        embed_dim = 256
        
        x = torch.randn(batch_size, seq_length, embed_dim)
        
        # Forward pass
        output = transformer_block(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_length, embed_dim))
    
    @patch("src.model.AutoModel")
    @patch("src.model.AutoTokenizer")
    def test_model_forward(self, mock_tokenizer, mock_auto_model):
        """Test model forward pass."""
        # Mock AutoModel and AutoTokenizer
        mock_base_model = MagicMock()
        mock_base_model_output = MagicMock()
        mock_base_model_output.hidden_states = [torch.randn(2, 5, 256)] * 13
        mock_base_model.return_value = mock_base_model_output
        mock_auto_model.from_pretrained.return_value = mock_base_model
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        
        # Initialize model
        model = NexusModel(
            vocab_size=1000,
            max_seq_length=128,
            embed_dim=256,
            num_heads=4,
            num_layers=2,
            ff_dim=512,
            dropout=0.1,
            base_model="bert-base-uncased",
            device="cpu"
        )
        
        # Create random inputs
        batch_size = 2
        seq_length = 5
        
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        
        # Forward pass
        with patch.object(model.base_model, "__call__", return_value=mock_base_model_output):
            logits = model(input_ids, attention_mask)
        
        # Check output shape
        self.assertEqual(logits.shape, (batch_size, seq_length, 1000))

if __name__ == "__main__":
    unittest.main()