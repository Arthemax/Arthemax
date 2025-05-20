"""
Training utilities for Nexus 1.1
"""

import os
import time
import json
import logging
from typing import Dict, List, Union, Any, Optional
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from .model import NexusModel
from .data import NexusDataModule, NexusDataConfig

logger = logging.getLogger(__name__)

@dataclass
class NexusTrainingConfig:
    """Configuration for training Nexus 1.1 model."""
    
    # Model parameters
    vocab_size: int = 50000
    max_seq_length: int = 512
    embed_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    ff_dim: int = 3072
    dropout: float = 0.1
    base_model: str = "bert-base-uncased"
    
    # Training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_epochs: int = 10
    warmup_steps: int = 1000
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Data parameters
    batch_size: int = 32
    num_workers: int = 4
    train_file: Optional[str] = None
    val_file: Optional[str] = None
    test_file: Optional[str] = None
    data_dir: str = "data"
    
    # Logging and saving
    log_every_n_steps: int = 100
    save_every_n_epochs: int = 1
    output_dir: str = "outputs"
    
    # Hardware
    device: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "NexusTrainingConfig":
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save(self, path: str):
        """Save config to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "NexusTrainingConfig":
        """Load config from file."""
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class NexusTrainer:
    """Trainer for Nexus 1.1 model."""
    
    def __init__(self, config: NexusTrainingConfig):
        self.config = config
        self.device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model
        self.model = NexusModel(
            vocab_size=config.vocab_size,
            max_seq_length=config.max_seq_length,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            ff_dim=config.ff_dim,
            dropout=config.dropout,
            base_model=config.base_model,
            device=self.device
        )
        
        # Create data module
        data_config = NexusDataConfig(
            max_seq_length=config.max_seq_length,
            batch_size=config.batch_size,
            tokenizer_name=config.base_model,
            train_file=config.train_file,
            val_file=config.val_file,
            test_file=config.test_file,
            data_dir=config.data_dir,
            num_workers=config.num_workers
        )
        self.data_module = NexusDataModule(data_config)
        
        # Create optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Create loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Initialize training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.train_losses = []
        self.val_losses = []
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        logger.info(f"Initialized trainer with model on {self.device}")
    
    def train(self):
        """Train the model."""
        logger.info("Starting training")
        
        # Set up data
        self.data_module.setup()
        train_dataloader = self.data_module.train_dataloader()
        val_dataloader = self.data_module.val_dataloader()
        
        # Training loop
        for epoch in range(self.current_epoch, self.config.max_epochs):
            self.current_epoch = epoch
            logger.info(f"Starting epoch {epoch+1}/{self.config.max_epochs}")
            
            # Train epoch
            train_loss = self._train_epoch(train_dataloader)
            self.train_losses.append(train_loss)
            
            # Validate epoch
            val_loss = self._validate_epoch(val_dataloader)
            self.val_losses.append(val_loss)
            
            logger.info(f"Epoch {epoch+1}/{self.config.max_epochs} - "
                       f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self._save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint("best_model.pt")
        
        # Save final model
        self._save_checkpoint("final_model.pt")
        
        # Plot training curves
        self._plot_training_curves()
        
        logger.info("Training completed")
    
    def _train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {self.current_epoch+1} (train)")
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            logits = self.model(batch["input_ids"], batch["attention_mask"])
            
            # Compute loss
            if "labels" in batch:
                labels = batch["labels"]
            else:
                # Use input_ids shifted right as labels
                labels = batch["input_ids"][:, 1:].contiguous()
                logits = logits[:, :-1, :].contiguous()
            
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights if needed
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.max_grad_norm
                )
                
                # Update weights
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Update global step
                self.global_step += 1
                
                # Log progress
                if self.global_step % self.config.log_every_n_steps == 0:
                    logger.info(f"Step {self.global_step} - Loss: {loss.item():.4f}")
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item() * self.config.gradient_accumulation_steps})
            
            # Accumulate loss
            total_loss += loss.item() * self.config.gradient_accumulation_steps
        
        # Compute average loss
        avg_loss = total_loss / len(dataloader)
        
        return avg_loss
    
    def _validate_epoch(self, dataloader: DataLoader) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc=f"Epoch {self.current_epoch+1} (val)")
            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                logits = self.model(batch["input_ids"], batch["attention_mask"])
                
                # Compute loss
                if "labels" in batch:
                    labels = batch["labels"]
                else:
                    # Use input_ids shifted right as labels
                    labels = batch["input_ids"][:, 1:].contiguous()
                    logits = logits[:, :-1, :].contiguous()
                
                loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                # Update progress bar
                progress_bar.set_postfix({"loss": loss.item()})
                
                # Accumulate loss
                total_loss += loss.item()
        
        # Compute average loss
        avg_loss = total_loss / len(dataloader)
        
        return avg_loss
    
    def _save_checkpoint(self, filename: str):
        """Save checkpoint."""
        checkpoint_path = os.path.join(self.config.output_dir, filename)
        
        # Create checkpoint
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses
        }
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load training state
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def _plot_training_curves(self):
        """Plot training curves."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plot_path = os.path.join(self.config.output_dir, "training_curves.png")
        plt.savefig(plot_path)
        
        logger.info(f"Saved training curves to {plot_path}")
    
    def generate_sample(self, input_text: str, max_length: int = 100):
        """Generate sample text."""
        self.model.eval()
        
        # Generate text
        generated_text = self.model.generate(
            input_text=input_text,
            max_length=max_length
        )
        
        return generated_text