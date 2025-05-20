# Nexus 1.1 Access Instructions

## Web Access Options

You can access Nexus 1.1 using one of the following URLs:

### Option 1: Primary Server

Access the Nexus 1.1 download page at:
[https://work-1-vhgvqpfwcvfdmegs.prod-runtime.all-hands.dev](https://work-1-vhgvqpfwcvfdmegs.prod-runtime.all-hands.dev)

Direct download link:
[https://work-1-vhgvqpfwcvfdmegs.prod-runtime.all-hands.dev/download/nexus_1_1.zip](https://work-1-vhgvqpfwcvfdmegs.prod-runtime.all-hands.dev/download/nexus_1_1.zip)

### Option 2: Backup Server

Access the Nexus 1.1 download page at:
[https://work-2-vhgvqpfwcvfdmegs.prod-runtime.all-hands.dev](https://work-2-vhgvqpfwcvfdmegs.prod-runtime.all-hands.dev)

Direct download link:
[https://work-2-vhgvqpfwcvfdmegs.prod-runtime.all-hands.dev/download/nexus_1_1.zip](https://work-2-vhgvqpfwcvfdmegs.prod-runtime.all-hands.dev/download/nexus_1_1.zip)

### Option 3: GitHub Repository

The code is also available on GitHub:
[https://github.com/Arthemax/Arthemax/tree/nexus-1-1](https://github.com/Arthemax/Arthemax/tree/nexus-1-1)

## Installation Instructions

1. Download and extract the ZIP file:
   ```bash
   unzip nexus_1_1.zip
   cd nexus_1_1
   ```

2. Install the package and dependencies:
   ```bash
   pip install -e .
   ```

3. Prepare sample data:
   ```bash
   python run.py prepare-data --output-dir data --num-samples 100
   ```

4. Train the model:
   ```bash
   python run.py train --config-file config/default_config.json
   ```

5. Generate text:
   ```bash
   python run.py generate --model-path outputs/best_model.pt --prompt "Hello, world!"
   ```

6. Run the API server:
   ```bash
   python run.py server --model-path outputs/best_model.pt --port 8000
   ```

7. Run the web UI:
   ```bash
   python run.py ui --api-url http://localhost:8000 --port 8001
   ```

## Docker Support

You can also run Nexus 1.1 using Docker:

```bash
docker build -t nexus-1-1 .
docker run -p 8000:8000 -p 8001:8001 nexus-1-1
```

## Documentation

For more detailed information, please refer to the documentation in the `docs/` directory:

- `architecture.md` - Detailed description of the model architecture
- `api_reference.md` - API documentation for the model
- `usage_guide.md` - Guide for using the model
- `development_guide.md` - Guide for developing the model