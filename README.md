# PDF Benchmarking for LLM Genetic Report Extraction

A comprehensive benchmarking framework that evaluates Large Language Model (LLM) performance on extracting structured data from genetic laboratory reports. The system generates synthetic genetic reports as PDFs and tests various LLMs' ability to extract accurate structured information through multiple API providers.

## 🎯 Purpose

This project addresses the critical need to evaluate how well different LLMs can process complex genetic laboratory reports and extract structured clinical data. It provides:

- **Realistic Test Data**: Generates synthetic genetic reports with authentic formatting
- **Multi-Provider Evaluation**: Tests models via OpenRouter, OpenAI, and local Ollama instances
- **Standardized Benchmarking**: Consistent evaluation framework across different models and providers
- **Advanced Performance Metrics**: Comprehensive scoring system with key-level accuracy analysis
- **Automated Pipeline**: End-to-end automation from report generation to performance analysis

## 🏗️ Architecture

### 3-Stage Pipeline

1. **📋 Report Generation** (`makeTemplatePDF/`)
   - Creates realistic mock genetic reports using R and LaTeX
   - Generates ground truth data for validation
   - Supports multiple hospital report formats

2. **🤖 Data Extraction** (`getJSON/`)
   - Converts PDFs to text
   - Processes reports through multiple LLMs
   - Extracts structured JSON data

3. **📊 Validation** (`getJSON/`)
   - Compares extracted data against ground truth
   - Provides accuracy metrics per model
   - Generates comprehensive performance reports

## 🚀 Quick Start Guide

### Prerequisites
Ensure you have the following installed:
```bash
# Required software
- R 4.0+ (with biomaRt, yaml, httr, RJSONIO packages)
- Python 3.8+ 
- LaTeX distribution (MacTeX on macOS, TeXLive on Linux)
- Git (for cloning the repository)

# Optional for local models
- Ollama (for local LLM testing)
```

### Step 1: Setup and Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd PDF_benchmarking
```

2. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

3. **Install R packages** (if not already installed):
```r
# Open R console and run:
install.packages(c("biomaRt", "yaml", "httr", "RJSONIO"))
```

4. **Configure API keys**:
Create `api_keys.txt` in the root directory with your credentials:
```
OPENROUTER_API_KEY=your-openrouter-key-here
OPENAI_API_KEY=your-openai-key-here
```

Or set environment variables:
```bash
export OPENROUTER_API_KEY="your-openrouter-key-here"
export OPENAI_API_KEY="your-openai-key-here"
```

5. **Install Ollama** (optional, for local models):
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve
```

### Step 2: Generate Synthetic Reports

Generate realistic genetic laboratory reports:

```bash
# Generate PDFs and convert to text files
./generate_reports.sh
```

This will:
- Create mock genetic data using R scripts
- Generate PDF reports using LaTeX templates
- Convert PDFs to text files for LLM processing
- Store outputs in `makeTemplatePDF/scripts/out/` and `output_pdfs/`

### Step 3: Run LLM Benchmarking

**Option A: Run Complete Pipeline** (Recommended for first-time users):
```bash
# Run all LLM providers and generate performance analysis
./run_all_llms.sh
```

**Option B: Run Individual Components**:

1. **Local Ollama Models**:
```bash
cd getJSON
python3 jsonOllama.py
```

2. **OpenAI Models**:
```bash
cd getJSON
python3 openAItoJSON.py
```

3. **OpenRouter Models**:
```bash
cd getJSON
python3 openRouterLLMs.py
```

4. **Performance Analysis**:
```bash
cd getJSON
python3 compareJSON.py
```

### Step 4: Review Results

After running the pipeline, check the following locations:

- **Raw LLM Outputs**: `getJSON/outJSON/`
  - `OllamaOut/` - Local Ollama model results
  - `OpenAIOut/` - OpenAI model results
  - `OpenRouter/` - OpenRouter model results
- **Performance Analysis**: `getJSON/Hospital.csv` - Accuracy scores and metrics
- **Generated Reports**: `output_pdfs/` - Original PDFs and extracted text

## 📁 Project Structure

```
PDF_benchmarking/
├── 📋 makeTemplatePDF/           # Report generation system
│   ├── data/                     # Configuration and gene data
│   │   ├── exon_info.csv        # Exon coordinate data
│   │   ├── field_values.yml     # Template field configurations
│   │   ├── gene_info.csv        # Gene information database
│   │   └── text_pieces.yml      # Text template components
│   ├── scripts/                  # R generation and processing scripts
│   │   ├── fetch_gene_info.R    # BioMart gene data fetching
│   │   ├── generate_mock_data.R # Mock patient data generation
│   │   ├── hospital1.r          # Hospital format 1 generator
│   │   ├── hospital2.r          # Hospital format 2 generator
│   │   ├── interpolate.R        # Template interpolation
│   │   ├── sharedFunctions.r    # Common R utilities
│   │   ├── run.sh               # Main generation pipeline
│   │   └── out/                 # Generated PDF outputs
│   └── templates/                # LaTeX report templates
│       ├── fakeHospital1.tex    # Template for hospital format 1
│       └── fakeHospital2.tex    # Template for hospital format 2
├── 🤖 getJSON/                   # LLM processing and evaluation
│   ├── openRouterLLMs.py        # OpenRouter API interface
│   ├── openAItoJSON.py          # OpenAI API interface
│   ├── jsonOllama.py            # Ollama local models interface
│   ├── localLLM.py              # Local model utilities
│   ├── pdfToText.py             # PDF to text conversion
│   ├── compareJSON.py           # Performance evaluation engine
│   ├── genGraphs.py             # Visualization generation
│   ├── visResults.ipynb         # Results visualization notebook
│   ├── prompt.txt               # LLM extraction prompt template
│   ├── ollamaPrompt.txt         # Ollama-specific prompt
│   ├── fakehospital1.txt        # Sample hospital 1 text
│   ├── hospital2.txt            # Sample hospital 2 text
│   ├── Hospital.csv             # Performance results summary
│   └── outJSON/                 # LLM extraction results
│       ├── OllamaOut/           # Local Ollama model outputs
│       ├── OpenAIOut/           # OpenAI model outputs
│       ├── OpenRouter/          # OpenRouter model outputs
│       └── OpenRouterVisionOut/ # Vision model outputs
├── 📄 output_pdfs/               # Generated reports and conversions
│   ├── images/                  # PDF page images
│   └── text/                    # Extracted text files
├── 🔧 Configuration Files
│   ├── api_keys.txt             # API credentials (create this)
│   ├── requirements.txt         # Python dependencies
│   ├── generate_reports.sh      # Report generation script
│   ├── run_all_llms.sh         # Complete benchmarking pipeline
│   └── README.md               # This documentation
```

## 🤖 Supported LLM Models

### OpenRouter Models (Free Tier)
- **Google Gemini 2.0 Flash** (`google/gemini-2.0-flash-exp:free`)
- **Meta Llama 4 Scout** (`meta-llama/llama-4-scout:free`) 
- **Mistral Devstral Small** (`mistralai/devstral-small:free`)
- **Qwen 2.5-VL 72B** (`qwen/qwen2.5-vl-72b-instruct:free`)
- **Reka Flash 3** (`rekaai/reka-flash-3:free`)

### OpenAI Models
- **GPT-4o Mini** (`gpt-4o-mini`)

### Local Ollama Models
*Models are automatically downloaded when first accessed:*
- **Llama 3.2 1B** (`llama3.2:1b`)
- **Llama 3.2 3B** (`llama3.2:3b`)
- **Phi 3 Mini** (`phi3:mini`)
- **Gemma 3 variants** (`gemma3:1b`, `gemma3:4b`)
- **Granite 3.3 2B** (`granite3.3:2b`)
- **SmolLM2 135M** (`smollm2:135m`)
- **Qwen 2.5-VL 3B** (`qwen2.5vl:3b`)

## 📊 Understanding Results

### Performance Metrics

The system generates comprehensive accuracy metrics in `getJSON/Hospital.csv`:

- **Total Keys**: Number of extractable fields in ground truth
- **String Keys**: Fields with string values for comparison
- **Extracted Keys**: Successfully identified fields by the model
- **Accuracy Percentage**: (Correct extractions / Total comparable fields) × 100
- **Missing Fields**: Fields not extracted by the model
- **Incorrect Values**: Fields extracted with wrong values

### Accuracy Interpretation

| Score Range | Performance Level | Description |
|-------------|------------------|-------------|
| 90-100% | Excellent | Production-ready, minimal errors |
| 80-89% | Good | Minor formatting issues, mostly accurate |
| 70-79% | Acceptable | Some field gaps, needs improvement |
| 60-69% | Poor | Significant extraction issues |
| <60% | Inadequate | Major problems, not suitable for use |

### Sample Output
```
Template loaded successfully.
Template has 45 total keys and 28 string value keys

--- fakeHospital1_google_gemini-2.0-flash-exp_free__response.json ---
Template: 45 total keys, 28 string keys
Data:     42 total keys, 25 string keys
✓ Found 2 differences out of 28 comparable string values
Accuracy: 92.9%
```

## 🔧 Advanced Configuration

### Customizing Models

**Add OpenRouter Models**:
Edit `getJSON/openRouterLLMs.py`:
```python
MODELS = [
    "google/gemini-2.0-flash-exp:free",
    "your/new-model:free",  # Add new models here
]
```

**Add Ollama Models**:
Edit `getJSON/jsonOllama.py`:
```python
MODELS = [
    "llama3.2:1b",
    "your-custom-model",  # Add locally installed models
]
```

### Modifying Report Templates

**Customize Gene Panels**:
Edit `makeTemplatePDF/data/field_values.yml`:
```yaml
genes:
  - BRCA1
  - BRCA2
  - TP53
  - your_gene_of_interest  # Add custom genes
```

**Hospital Format Customization**:
Modify LaTeX templates in `makeTemplatePDF/templates/`:
- `fakeHospital1.tex` - Clinical genetics format
- `fakeHospital2.tex` - Molecular pathology format

### Performance Analysis

**Focus on Specific Models**:
```python
# In compareJSON.py, filter by model type
json_files = [f for f in json_files if 'gemini' in f or 'llama' in f]
```

**Hospital Format Analysis**:
```python
# Analyze specific hospital formats
json_files = [f for f in json_files if 'fakeHospital1' in f]
```

## 🛠️ Troubleshooting

### Common Issues and Solutions

| Problem | Solution |
|---------|----------|
| **LaTeX compilation errors** | Install complete LaTeX distribution (MacTeX/TeXLive) |
| **R package installation fails** | Install system dependencies, update R version |
| **BioMart timeouts** | Script includes retry logic, check internet connection |
| **Ollama models not found** | Ensure Ollama is running: `ollama serve` |
| **API rate limits** | Built-in exponential backoff handles this automatically |
| **JSON parsing errors** | Check raw responses in output directories |
| **Permission denied on scripts** | Make scripts executable: `chmod +x *.sh` |

### Debug Mode

Enable detailed logging:
```bash
# For PDF generation
cd makeTemplatePDF/scripts
bash -x run.sh

# For LLM processing  
cd getJSON
python3 -u jsonOllama.py  # Unbuffered output
```

### Manual Steps (if automation fails)

1. **Generate reports manually**:
```bash
cd makeTemplatePDF/scripts
Rscript generate_mock_data.R
Rscript hospital1.r
Rscript hospital2.r
```

2. **Convert PDFs manually**:
```bash
cd getJSON
python3 pdfToText.py
```

3. **Test single model**:
```bash
cd getJSON
# Edit the respective script to test one model at a time
python3 jsonOllama.py
```

## 🚀 Usage Examples

### Basic Benchmarking Workflow
```bash
# 1. Set up environment
export OPENROUTER_API_KEY="your-key"
export OPENAI_API_KEY="your-key"

# 2. Generate test data
./generate_reports.sh

# 3. Run complete evaluation
./run_all_llms.sh

# 4. View results
cat getJSON/Hospital.csv
```

### Custom Evaluation Scenario
```bash
# Test only local models
cd getJSON
python3 jsonOllama.py
python3 compareJSON.py

# Test specific OpenRouter models
# Edit openRouterLLMs.py to include only desired models
python3 openRouterLLMs.py
python3 compareJSON.py
```

### Performance Analysis
```bash
# Generate visualizations
cd getJSON
python3 genGraphs.py

# Open Jupyter notebook for detailed analysis
jupyter notebook visResults.ipynb
```

## 🤝 Contributing

### Adding New LLM Providers

1. **Create provider script** following the pattern of existing providers
2. **Ensure consistent output format** matching the JSON schema
3. **Add error handling** and rate limiting
4. **Update compareJSON.py** to include new provider results
5. **Test thoroughly** with sample data

### Enhancing Evaluation

1. **Add new metrics** to `compareJSON.py`
2. **Implement field-importance weighting** for clinical relevance
3. **Create visualization tools** for performance analysis
4. **Add statistical significance testing**

---

**Note**: This framework uses only synthetic data for testing. No real patient information is processed. Ensure compliance with your organization's data policies and API terms of service.
