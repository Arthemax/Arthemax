# Nexus 1.1 API Reference

## Model API

### `NexusModel`

The main model class for Nexus 1.1.

```python
from nexus_1_1 import NexusModel

model = NexusModel(
    vocab_size=50000,
    max_seq_length=512,
    embed_dim=768,
    num_heads=12,
    num_layers=12,
    ff_dim=3072,
    dropout=0.1,
    base_model="bert-base-uncased",
    device=None  # Defaults to CUDA if available, otherwise CPU
)
```

#### Parameters

- `vocab_size` (int, optional): Size of the vocabulary. Defaults to 50000.
- `max_seq_length` (int, optional): Maximum sequence length. Defaults to 512.
- `embed_dim` (int, optional): Dimension of embeddings. Defaults to 768.
- `num_heads` (int, optional): Number of attention heads. Defaults to 12.
- `num_layers` (int, optional): Number of transformer layers. Defaults to 12.
- `ff_dim` (int, optional): Dimension of feed-forward layer. Defaults to 3072.
- `dropout` (float, optional): Dropout rate. Defaults to 0.1.
- `base_model` (str, optional): Base model to use. Defaults to "bert-base-uncased".
- `device` (str, optional): Device to use. Defaults to CUDA if available, otherwise CPU.

#### Methods

##### `forward(input_ids, attention_mask=None)`

Forward pass through the model.

```python
logits = model(input_ids, attention_mask)
```

##### `generate(input_text, max_length=100, temperature=1.0)`

Generate text from the model.

```python
generated_text = model.generate(
    input_text="Hello, world!",
    max_length=100,
    temperature=1.0
)
```

##### `save(path)`

Save model to disk.

```python
model.save("models/my_model.pt")
```

##### `load(path, device=None)`

Load model from disk.

```python
model = NexusModel.load("models/my_model.pt", device="cuda")
```

## Data API

### `NexusDataModule`

Data module for Nexus 1.1.

```python
from nexus_1_1 import NexusDataModule, NexusDataConfig

config = NexusDataConfig(
    max_seq_length=512,
    batch_size=32,
    tokenizer_name="bert-base-uncased",
    train_file="data/train.jsonl",
    val_file="data/val.jsonl",
    test_file="data/test.jsonl",
    data_dir="data",
    num_workers=4,
    shuffle=True
)

data_module = NexusDataModule(config)
data_module.setup()
```

#### Methods

##### `setup()`

Set up datasets.

```python
data_module.setup()
```

##### `train_dataloader()`

Get training dataloader.

```python
train_dataloader = data_module.train_dataloader()
```

##### `val_dataloader()`

Get validation dataloader.

```python
val_dataloader = data_module.val_dataloader()
```

##### `test_dataloader()`

Get test dataloader.

```python
test_dataloader = data_module.test_dataloader()
```

##### `prepare_sample_data(output_dir, num_samples=100)`

Prepare sample data for testing.

```python
data_module.prepare_sample_data(
    output_dir="data",
    num_samples=100
)
```

## Trainer API

### `NexusTrainer`

Trainer for Nexus 1.1.

```python
from nexus_1_1 import NexusTrainer, NexusTrainingConfig

config = NexusTrainingConfig(
    vocab_size=50000,
    max_seq_length=512,
    embed_dim=768,
    num_heads=12,
    num_layers=12,
    ff_dim=3072,
    dropout=0.1,
    base_model="bert-base-uncased",
    learning_rate=1e-4,
    weight_decay=0.01,
    max_epochs=10,
    warmup_steps=1000,
    gradient_accumulation_steps=1,
    max_grad_norm=1.0,
    batch_size=32,
    num_workers=4,
    train_file="data/train.jsonl",
    val_file="data/val.jsonl",
    test_file="data/test.jsonl",
    data_dir="data",
    log_every_n_steps=100,
    save_every_n_epochs=1,
    output_dir="outputs",
    device=None
)

trainer = NexusTrainer(config)
```

#### Methods

##### `train()`

Train the model.

```python
trainer.train()
```

##### `load_checkpoint(checkpoint_path)`

Load checkpoint.

```python
trainer.load_checkpoint("outputs/checkpoint_epoch_1.pt")
```

##### `generate_sample(input_text, max_length=100)`

Generate sample text.

```python
generated_text = trainer.generate_sample(
    input_text="Hello, world!",
    max_length=100
)
```

## Web API

### REST API

The Nexus 1.1 REST API provides the following endpoints:

#### `GET /`

Root endpoint.

```
GET /
```

Response:

```json
{
  "message": "Nexus 1.1 API is running"
}
```

#### `GET /health`

Health check endpoint.

```
GET /health
```

Response:

```json
{
  "status": "ok",
  "message": "Model is loaded and ready"
}
```

#### `POST /generate`

Generate text endpoint.

```
POST /generate
```

Request:

```json
{
  "prompt": "Hello, world!",
  "max_length": 100,
  "temperature": 1.0
}
```

Response:

```json
{
  "generated_text": "Hello, world! This is a generated response from Nexus 1.1."
}
```

#### `GET /model-info`

Get model information.

```
GET /model-info
```

Response:

```json
{
  "model_name": "Nexus 1.1",
  "embed_dim": 768,
  "num_layers": 12,
  "num_heads": 12,
  "device": "cuda:0"
}
```