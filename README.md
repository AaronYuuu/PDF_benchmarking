# PDF Benchmarking for LLM Genetic Report Extraction

A benchmarking framework that evaluates Large Language Model (LLM) performance on extracting structured data from genetic laboratory reports. The system generates synthetic genetic reports as PDFs and tests various LLMs' ability to extract accurate structured information.

## Purpose

This project evaluates how well different LLMs can process complex genetic laboratory reports and extract structured clinical data. It provides:

- Realistic test data using synthetic genetic reports
- Multi-provider evaluation via OpenRouter, OpenAI, and local Ollama instances
- Standardized benchmarking across different models and providers
- Comprehensive performance metrics and accuracy analysis
- End-to-end automation from report generation to performance analysis

## Architecture

### 3-Stage Pipeline

1. **Report Generation** (makeTemplatePDF/)
   - Creates mock genetic reports using R and LaTeX
   - Generates ground truth data for validation
   - Supports multiple hospital report formats

2. **Data Extraction** (getJSON/)
   - Converts PDFs to text
   - Processes reports through multiple LLMs
   - Extracts structured JSON data

3. **Validation** (getJSON/)
   - Compares extracted data against ground truth
   - Provides accuracy metrics per model
   - Generates performance reports

## Quick Start

### Prerequisites

Required software:
- R 4.0+ (with biomaRt, yaml, httr, RJSONIO packages)
- Python 3.8+
- LaTeX distribution (MacTeX on macOS, TeXLive on Linux)
- Git

Optional:
- Ollama (for local LLM testing)

### Setup and Installation

1. Clone the repository and install dependencies:
```bash
git clone <repository-url>
cd PDF_benchmarking
pip install -r requirements.txt
```

2. Install R packages:
```r
install.packages(c("biomaRt", "yaml", "httr", "RJSONIO"))
```

3. Configure API keys in api_keys.txt:
```
OPENROUTER_API_KEY=your-openrouter-key-here
OPENAI_API_KEY=your-openai-key-here
```

4. Install Ollama (optional):
```bash
# macOS
brew install ollama
# Linux
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve
```

### Generate Reports

```bash
./generate_reports.sh
```

### Run Benchmarking

Complete pipeline:
```bash
./run_all_llms.sh
```

Individual components:
```bash
cd getJSON
python3 jsonOllama.py      # Local models
python3 openAItoJSON.py    # OpenAI models
python3 openRouterLLMs.py  # OpenRouter models
python3 compareJSON.py     # Performance analysis
```

### Review Results

- Raw outputs: getJSON/outJSON/
- Performance metrics: getJSON/Hospital.csv
- Generated reports: output_pdfs/

## Project Structure

```
PDF_benchmarking/
├── makeTemplatePDF/              # Report generation system
│   ├── data/                     # Configuration and gene data
│   ├── scripts/                  # R generation scripts
│   └── templates/                # LaTeX report templates
├── getJSON/                      # LLM processing and evaluation
│   ├── openRouterLLMs.py        # OpenRouter interface
│   ├── openAItoJSON.py          # OpenAI interface
│   ├── jsonOllama.py            # Ollama interface
│   ├── compareJSON.py           # Performance evaluation
│   └── outJSON/                 # LLM results
├── output_pdfs/                  # Generated reports
├── api_keys.txt                  # API credentials
├── requirements.txt              # Python dependencies
└── README.md                     # Documentation
```

## Supported Models

### OpenRouter Models (Free Tier)
- Google Gemini 2.0 Flash
- Meta Llama 4 Scout
- Mistral Devstral Small
- Qwen 2.5-VL 72B
- Reka Flash 3

### OpenAI Models
- GPT-4o Mini

### Local Ollama Models
- Llama 3.2 variants
- Gemma 3 variants
- Granite 3.3 2B
- Phi 3 Mini
- SmolLM2 135M
- Qwen 2.5-VL 3B

## Understanding Results

The system generates accuracy metrics in getJSON/Hospital.csv with:
- Total Keys: Number of extractable fields
- Extracted Keys: Successfully identified fields
- Accuracy Percentage: Correct extractions / Total fields
- Missing Fields: Fields not extracted
- Incorrect Values: Fields with wrong values

## Configuration

### Adding Models

Edit the respective Python files:
- openRouterLLMs.py for OpenRouter models
- jsonOllama.py for Ollama models
- openAItoJSON.py for OpenAI models

### Customizing Templates

Modify files in makeTemplatePDF/data/:
- field_values.yml for gene panels
- text_pieces.yml for template components

## Troubleshooting

Common issues:
- LaTeX errors: Install complete distribution
- R package issues: Update R version
- Ollama not found: Ensure service is running
- API limits: Built-in retry logic handles this
- Permission errors: Make scripts executable with chmod +x

## Manual Steps

If automation fails:

1. Generate reports:
```bash
cd makeTemplatePDF/scripts
Rscript generate_mock_data.R
Rscript hospital1.r
Rscript hospital2.r
```

2. Convert PDFs:
```bash
cd getJSON
python3 pdfToText.py
```

3. Test models:
```bash
cd getJSON
python3 jsonOllama.py
python3 compareJSON.py
```
