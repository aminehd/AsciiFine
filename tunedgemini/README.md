# TunedGemini

A Python module for fine-tuning Gemini models on Google Vertex AI with ASCII art generation.

## Features

- Fine-tune Gemini models on Google Vertex AI
- Process and prepare ASCII art datasets for training
- Evaluate fine-tuned models
- Generate ASCII art using the fine-tuned model

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd AsciiFine/tunedgemini

# Easy setup with the setup script (installs dependencies and creates basic config)
./setup.sh

# Or manually:
# Install dependencies with Poetry
poetry install

# Activate virtual environment
poetry shell
```

## Basic Configuration

You can use the CLI tool to check your configuration without connecting to GCP:

```bash
# Check your configuration
poetry run tunedgemini config

# Create a default .env file from the template
poetry run tunedgemini config --create-env
```

## Setting up Google Cloud and Vertex AI credentials

When you're ready to connect to GCP:

1. Create a Google Cloud project and enable Vertex AI API
2. Set up authentication:
   ```bash
   # Export your credentials path
   export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/credentials.json"
   ```
3. Or edit the `.env` file with your credentials and project settings

## Usage

### Preparing the dataset

```python
from tunedgemini.data import prepare_ascii_dataset

# Prepare the dataset for fine-tuning
dataset = prepare_ascii_dataset("path/to/ascii_art_data.jsonl")
```

### Fine-tuning Gemini

```python
from tunedgemini.trainer import GeminiFinetuner

# Initialize the trainer
trainer = GeminiFinetuner(
    project_id="your-gcp-project-id",
    location="us-central1",  # or other appropriate region
    model_name="gemini-1.5-pro"
)

# Start fine-tuning
job = trainer.start_finetuning(
    dataset=dataset,
    epochs=3,
    batch_size=8
)

# Monitor training progress
trainer.monitor_job(job)
```

### Generating ASCII art with fine-tuned model

```python
from tunedgemini.generator import ASCIIArtGenerator

generator = ASCIIArtGenerator(
    project_id="your-gcp-project-id",
    location="us-central1",
    model_id="your-finetuned-model-id"
)

# Generate ASCII art
ascii_art = generator.generate("a cat sitting on a windowsill")
print(ascii_art)
```

## Command Line Interface

The package includes a command-line interface:

```bash
# Check configuration
tunedgemini config

# Create .env file from template
tunedgemini config --create-env

# Generate ASCII art (after configuring GCP)
# Coming soon
```

## Project Structure

```
tunedgemini/
├── tunedgemini/
│   ├── __init__.py
│   ├── trainer.py       # Fine-tuning functionality
│   ├── data.py          # Dataset preparation utilities
│   ├── generator.py     # Generation utilities 
│   ├── config.py        # Configuration utilities
│   ├── cli.py           # Command-line interface
│   └── utils.py         # Misc utilities
├── tests/               # Unit tests
├── examples/            # Example scripts
├── setup.sh             # Easy setup script
├── pyproject.toml       # Poetry configuration
└── README.md            # This file
```

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 