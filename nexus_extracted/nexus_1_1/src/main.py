"""
Main entry point for Nexus 1.1
"""

import os
import argparse
import logging
import json
from typing import Dict, Any

from .model import NexusModel
from .data import NexusDataModule, NexusDataConfig
from .trainer import NexusTrainer, NexusTrainingConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def train(args: argparse.Namespace):
    """Train Nexus 1.1 model."""
    logger.info("Starting training")
    
    # Load or create config
    if args.config_file and os.path.exists(args.config_file):
        logger.info(f"Loading config from {args.config_file}")
        config = NexusTrainingConfig.load(args.config_file)
    else:
        logger.info("Creating default config")
        config = NexusTrainingConfig()
    
    # Update config with command line arguments
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.max_epochs is not None:
        config.max_epochs = args.max_epochs
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    
    # Create trainer
    trainer = NexusTrainer(config)
    
    # Load checkpoint if provided
    if args.checkpoint:
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)
    
    # Save config
    config_path = os.path.join(config.output_dir, "config.json")
    config.save(config_path)
    logger.info(f"Saved config to {config_path}")
    
    # Train model
    trainer.train()
    
    logger.info("Training completed")

def generate(args: argparse.Namespace):
    """Generate text with Nexus 1.1 model."""
    logger.info("Starting text generation")
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = NexusModel.load(args.model_path)
    
    # Generate text
    logger.info(f"Generating text from prompt: {args.prompt}")
    generated_text = model.generate(
        input_text=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature
    )
    
    # Print generated text
    print("\nGenerated Text:")
    print("-" * 50)
    print(generated_text)
    print("-" * 50)
    
    # Save generated text if output file is provided
    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(generated_text)
        logger.info(f"Saved generated text to {args.output_file}")
    
    logger.info("Text generation completed")

def prepare_data(args: argparse.Namespace):
    """Prepare sample data for Nexus 1.1 model."""
    logger.info("Preparing sample data")
    
    # Create data config
    data_config = NexusDataConfig(
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        tokenizer_name=args.tokenizer_name,
        data_dir=args.data_dir
    )
    
    # Create data module
    data_module = NexusDataModule(data_config)
    
    # Prepare sample data
    data_module.prepare_sample_data(
        output_dir=args.output_dir,
        num_samples=args.num_samples
    )
    
    logger.info(f"Prepared {args.num_samples} sample data points in {args.output_dir}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Nexus 1.1 - Advanced Autonomous AI Model")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train Nexus 1.1 model")
    train_parser.add_argument("--config-file", type=str, help="Path to config file")
    train_parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file")
    train_parser.add_argument("--learning-rate", type=float, help="Learning rate")
    train_parser.add_argument("--batch-size", type=int, help="Batch size")
    train_parser.add_argument("--max-epochs", type=int, help="Maximum number of epochs")
    train_parser.add_argument("--output-dir", type=str, help="Output directory")
    
    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate text with Nexus 1.1 model")
    generate_parser.add_argument("--model-path", type=str, required=True, help="Path to model file")
    generate_parser.add_argument("--prompt", type=str, required=True, help="Prompt for text generation")
    generate_parser.add_argument("--max-length", type=int, default=100, help="Maximum length of generated text")
    generate_parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    generate_parser.add_argument("--output-file", type=str, help="Path to output file")
    
    # Prepare data command
    data_parser = subparsers.add_parser("prepare-data", help="Prepare sample data for Nexus 1.1 model")
    data_parser.add_argument("--max-seq-length", type=int, default=512, help="Maximum sequence length")
    data_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    data_parser.add_argument("--tokenizer-name", type=str, default="bert-base-uncased", help="Tokenizer name")
    data_parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    data_parser.add_argument("--output-dir", type=str, default="data", help="Output directory")
    data_parser.add_argument("--num-samples", type=int, default=100, help="Number of samples to generate")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run command
    if args.command == "train":
        train(args)
    elif args.command == "generate":
        generate(args)
    elif args.command == "prepare-data":
        prepare_data(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()