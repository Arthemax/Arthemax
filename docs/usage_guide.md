# Nexus 1.1 Usage Guide

This guide provides instructions for using Nexus 1.1 in various scenarios.

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/nexus-1-1.git
cd nexus-1-1

# Install dependencies
pip install -e .
```

### Using Docker

```bash
# Build Docker image
docker build -t nexus-1-1 .

# Run Docker container
docker run -p 12000:12000 -p 12001:12001 nexus-1-1
```

## Basic Usage

### Preparing Sample Data

```bash
# Prepare sample data
python run.py prepare-data --output-dir data --num-samples 100
```

### Training the Model

```bash
# Train the model
python run.py train --config-file config/default_config.json
```

### Generating Text

```bash
# Generate text
python run.py generate --model-path outputs/best_model.pt --prompt "Hello, world!"
```

### Running the API Server

```bash
# Run the API server
python run.py server --model-path outputs/best_model.pt --port 12000
```

### Running the Web UI

```bash
# Run the web UI
python run.py ui --api-url http://localhost:12000 --port 12001
```

## Python API

### Loading a Pre-trained Model

```python
from nexus_1_1 import NexusModel

# Load model
model = NexusModel.load("outputs/best_model.pt")

# Generate text
generated_text = model.generate(
    input_text="Hello, world!",
    max_length=100,
    temperature=1.0
)

print(generated_text)
```

### Training a Model

```python
from nexus_1_1 import NexusTrainer, NexusTrainingConfig

# Create config
config = NexusTrainingConfig(
    max_epochs=5,
    batch_size=16,
    learning_rate=1e-4,
    output_dir="outputs"
)

# Create trainer
trainer = NexusTrainer(config)

# Train model
trainer.train()
```

### Using the Data Module

```python
from nexus_1_1 import NexusDataModule, NexusDataConfig

# Create config
config = NexusDataConfig(
    max_seq_length=512,
    batch_size=32,
    tokenizer_name="bert-base-uncased",
    data_dir="data"
)

# Create data module
data_module = NexusDataModule(config)

# Set up data
data_module.setup()

# Get dataloaders
train_dataloader = data_module.train_dataloader()
val_dataloader = data_module.val_dataloader()
test_dataloader = data_module.test_dataloader()
```

## REST API

### Generating Text

```bash
# Generate text using the API
curl -X POST http://localhost:12000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world!", "max_length": 100, "temperature": 1.0}'
```

### Getting Model Information

```bash
# Get model information
curl -X GET http://localhost:12000/model-info
```

## Web UI

The web UI provides a user-friendly interface for interacting with the model. To use it:

1. Start the API server:

```bash
python run.py server --model-path outputs/best_model.pt --port 12000
```

2. Start the web UI:

```bash
python run.py ui --api-url http://localhost:12000 --port 12001
```

3. Open a web browser and navigate to `http://localhost:12001`

4. Enter a prompt and click "Generate" to generate text

## Advanced Usage

### Fine-tuning on Custom Data

To fine-tune the model on your own data:

1. Prepare your data in JSONL format:

```json
{"text": "This is an example input.", "target": "This is an example output."}
```

2. Create a training config:

```python
from nexus_1_1 import NexusTrainingConfig

config = NexusTrainingConfig(
    max_epochs=5,
    batch_size=16,
    learning_rate=1e-4,
    train_file="path/to/train.jsonl",
    val_file="path/to/val.jsonl",
    output_dir="outputs"
)
```

3. Train the model:

```python
from nexus_1_1 import NexusTrainer

trainer = NexusTrainer(config)
trainer.train()
```

### Customizing the Model

To customize the model architecture:

```python
from nexus_1_1 import NexusModel

# Create a custom model
model = NexusModel(
    vocab_size=50000,
    max_seq_length=1024,
    embed_dim=1024,
    num_heads=16,
    num_layers=24,
    ff_dim=4096,
    dropout=0.1,
    base_model="bert-large-uncased"
)
```

### Deploying to Production

For production deployment, we recommend:

1. Using Docker for containerization
2. Setting up a load balancer for high availability
3. Implementing monitoring and logging
4. Using a production-grade web server like Nginx

Example Docker Compose configuration:

```yaml
version: '3'

services:
  api:
    image: nexus-1-1
    command: server --model-path /app/models/best_model.pt --port 12000
    ports:
      - "12000:12000"
    volumes:
      - ./models:/app/models
    restart: always

  ui:
    image: nexus-1-1
    command: ui --api-url http://api:12000 --port 12001
    ports:
      - "12001:12001"
    depends_on:
      - api
    restart: always
```