# PDF Benchmarking

This project generates mock genetic reports using lei_mockup_generator, extracts data from the PDFs using LLMs, and validates the extraction accuracy.

## Prerequisites

1. Access to the private `lei_mockup_generator` repository
2. GitHub Personal Access Token or SSH key configured
3. Python 3.8+ and R installed
4. An API key 

## Setup

### Step 1: Clone and Setup
```bash
# Clone this repository
git clone https://github.com/yourusername/PDF_benchmarking.git
cd PDF_benchmarking

# Set your GitHub token for accessing private lei_mockup_generator
export GITHUB_TOKEN="your_personal_access_token"

# Run setup
./setup.sh
```

### Step 2: Set API Keys
```bash
# Set your LLM API keys
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

## Usage

### Generate reports and validate extraction:
```bash
./scripts/run_pipeline.sh -a 10 -p openai -m gpt-4-vision-preview
```

### Options:
- `-a, --amount`: Number of reports to generate (default: 5)
- `-o, --outdir`: Output directory (default: data/)
- `-p, --provider`: LLM provider (openai|anthropic) (default: openai)
- `-m, --model`: Model name (default: gpt-4-vision-preview)

## Project Structure

- `src/lei_mockup_generator/` - Git submodule with report generation
- `src/pdf_extractor/` - LLM-based PDF data extraction
- `src/validator/` - JSON comparison and validation
- `data/` - Generated files and results
- `config/` - Configuration files and schemas
- `scripts/` - Main pipeline and utility scripts
