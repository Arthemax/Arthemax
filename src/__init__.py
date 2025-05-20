"""
Nexus 1.1 - Advanced Autonomous AI Model
"""

from .model import NexusModel, NexusAttention, NexusTransformerBlock
from .data import NexusDataModule, NexusDataset, NexusDataConfig
from .trainer import NexusTrainer, NexusTrainingConfig

__version__ = "0.1.0"
__all__ = [
    "NexusModel",
    "NexusAttention",
    "NexusTransformerBlock",
    "NexusDataModule",
    "NexusDataset",
    "NexusDataConfig",
    "NexusTrainer",
    "NexusTrainingConfig",
]