"""
Data processing utilities for Nexus 1.1
"""

import os
import json
import logging
from typing import Dict, List, Union, Any, Optional, Tuple
from dataclasses import dataclass

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

@dataclass
class NexusDataConfig:
    """Configuration for data processing."""
    
    max_seq_length: int = 512
    batch_size: int = 32
    tokenizer_name: str = "bert-base-uncased"
    train_file: Optional[str] = None
    val_file: Optional[str] = None
    test_file: Optional[str] = None
    data_dir: str = "data"
    num_workers: int = 4
    shuffle: bool = True


class NexusDataset(Dataset):
    """Dataset for Nexus 1.1 model."""
    
    def __init__(
        self,
        file_path: str,
        tokenizer: AutoTokenizer,
        max_seq_length: int = 512,
        is_training: bool = True
    ):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.is_training = is_training
        
        # Load data
        self.data = self._load_data()
        
        logger.info(f"Loaded {len(self.data)} examples from {file_path}")
        
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from file."""
        if not os.path.exists(self.file_path):
            logger.warning(f"Data file {self.file_path} does not exist")
            return []
        
        # Determine file type
        if self.file_path.endswith(".json"):
            with open(self.file_path, "r") as f:
                data = json.load(f)
        elif self.file_path.endswith(".jsonl"):
            data = []
            with open(self.file_path, "r") as f:
                for line in f:
                    data.append(json.loads(line))
        elif self.file_path.endswith(".csv"):
            data = pd.read_csv(self.file_path).to_dict("records")
        elif self.file_path.endswith(".txt"):
            data = []
            with open(self.file_path, "r") as f:
                for line in f:
                    data.append({"text": line.strip()})
        else:
            raise ValueError(f"Unsupported file format: {self.file_path}")
        
        return data
    
    def __len__(self) -> int:
        """Return number of examples."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get example at index."""
        example = self.data[idx]
        
        # Get text from example
        if "text" in example:
            text = example["text"]
        elif "input" in example:
            text = example["input"]
        elif "source" in example:
            text = example["source"]
        else:
            raise ValueError(f"Could not find text field in example: {example}")
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Remove batch dimension
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        # Add labels for training
        if self.is_training and "target" in example:
            target_encoding = self.tokenizer(
                example["target"],
                max_length=self.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            encoding["labels"] = target_encoding["input_ids"].squeeze(0)
        
        return encoding


class NexusDataModule:
    """Data module for Nexus 1.1 model."""
    
    def __init__(self, config: NexusDataConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        
        # Set up data paths
        self.train_file = config.train_file or os.path.join(config.data_dir, "train.jsonl")
        self.val_file = config.val_file or os.path.join(config.data_dir, "val.jsonl")
        self.test_file = config.test_file or os.path.join(config.data_dir, "test.jsonl")
        
        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def setup(self):
        """Set up datasets."""
        # Training dataset
        if os.path.exists(self.train_file):
            self.train_dataset = NexusDataset(
                self.train_file,
                self.tokenizer,
                self.config.max_seq_length,
                is_training=True
            )
        
        # Validation dataset
        if os.path.exists(self.val_file):
            self.val_dataset = NexusDataset(
                self.val_file,
                self.tokenizer,
                self.config.max_seq_length,
                is_training=False
            )
        
        # Test dataset
        if os.path.exists(self.test_file):
            self.test_dataset = NexusDataset(
                self.test_file,
                self.tokenizer,
                self.config.max_seq_length,
                is_training=False
            )
    
    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        if self.train_dataset is None:
            raise ValueError("Training dataset not set up")
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader."""
        if self.val_dataset is None:
            raise ValueError("Validation dataset not set up")
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers
        )
    
    def test_dataloader(self) -> DataLoader:
        """Get test dataloader."""
        if self.test_dataset is None:
            raise ValueError("Test dataset not set up")
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers
        )
    
    def prepare_sample_data(self, output_dir: str, num_samples: int = 100):
        """Prepare sample data for testing."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate sample data
        samples = []
        for i in range(num_samples):
            sample = {
                "text": f"This is sample text number {i}.",
                "target": f"Sample response for text {i}."
            }
            samples.append(sample)
        
        # Split into train/val/test
        train_samples = samples[:int(0.7 * num_samples)]
        val_samples = samples[int(0.7 * num_samples):int(0.9 * num_samples)]
        test_samples = samples[int(0.9 * num_samples):]
        
        # Write to files
        with open(os.path.join(output_dir, "train.jsonl"), "w") as f:
            for sample in train_samples:
                f.write(json.dumps(sample) + "\n")
        
        with open(os.path.join(output_dir, "val.jsonl"), "w") as f:
            for sample in val_samples:
                f.write(json.dumps(sample) + "\n")
        
        with open(os.path.join(output_dir, "test.jsonl"), "w") as f:
            for sample in test_samples:
                f.write(json.dumps(sample) + "\n")
        
        logger.info(f"Prepared {num_samples} sample data points in {output_dir}")