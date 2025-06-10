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

3. **📊 Validation** (`JSONout/`)
   - Compares extracted data against ground truth
   - Provides accuracy metrics per model

## 🚀 Quick Start

### Prerequisites
```bash
# Required software
- R 4.0+ (with biomaRt, yaml, httr, RJSONIO packages)
- Python 3.8+ 
- LaTeX distribution
- API keys for desired providers:
  - OpenRouter API key
  - OpenAI API key (optional)
  - Local Ollama installation (optional)
```

### Installation
```bash
# Install Python dependencies
pip install -r requirements.txt

# Set API keys in api_keys.txt or as environment variables
echo "OPENROUTER_API_KEY=your-key-here" >> api_keys.txt
echo "OPENAI_API_KEY=your-openai-key" >> api_keys.txt
```

### Run Complete Pipeline
```bash
# Generate reports, extract data from all providers, and analyze results
./run_all_llms.sh

# Or run individual components:
./generate_reports.sh                    # Generate PDFs
python getJSON/pdfToText.py             # Convert to text
python getJSON/openRouterLLMs.py        # OpenRouter models
python getJSON/openAItoJSON.py          # OpenAI models  
python getJSON/jsonOllama.py            # Local Ollama models
python getJSON/compareJSON.py           # Performance analysis
```

## 📁 Project Structure

```
PDF_benchmarking/
├── 📋 makeTemplatePDF/           # Report generation
│   ├── data/                     # Gene data & configuration
│   │   ├── exon_info.csv
│   │   ├── field_values.yml
│   │   ├── gene_info.csv
│   │   └── text_pieces.yml
│   ├── scripts/                  # R generation scripts
│   │   ├── fetch_gene_info.R
│   │   ├── generate_mock_data.R
│   │   ├── hospital1.r
│   │   ├── hospital2.r
│   │   └── run.sh
│   └── templates/                # LaTeX report templates
│       ├── fakeHospital1.tex
│       └── fakeHospital2.tex
├── 🤖 getJSON/                   # LLM extraction engines
│   ├── openRouterLLMs.py         # OpenRouter API models
│   ├── openAItoJSON.py           # OpenAI API models
│   ├── jsonOllama.py             # Local Ollama models
│   ├── localLLM.py               # Local model utilities
│   ├── pdfToText.py              # PDF conversion
│   ├── compareJSON.py            # Performance evaluation
│   ├── prompt.txt                # Extraction prompt
│   ├── blurb_hospital1.txt       # Template text samples
│   ├── blurb_hospital2.txt
│   └── outJSON/                  # Results by provider
│       ├── JSONout/              # OpenRouter results
│       ├── OllamaOut/            # Ollama results
│       └── OpenAIOut/            # OpenAI results
├── 📄 output_pdfs/               # Generated PDFs and text
│   └── text/                     # Extracted text files
├── 🔧 api_keys.txt               # API configuration
├── 🔧 requirements.txt           # Python dependencies
├── 🔧 generate_reports.sh        # Report generation pipeline
└── 🔧 run_all_llms.sh           # Complete benchmarking pipeline
```

## 🤖 Supported LLM Models - in use currently

### OpenRouter Models (Free Tier)
- **Google Gemini 2.0 Flash** (`google/gemini-2.0-flash-exp:free`)
- **Meta Llama 4 Scout** (`meta-llama/llama-4-scout:free`) 
- **Mistral Devstral Small** (`mistralai/devstral-small:free`)
- **Qwen 2.5-VL 72B** (`qwen/qwen2.5-vl-72b-instruct:free`)
- **Reka Flash 3** (`rekaai/reka-flash-3:free`)

### OpenAI Models
- **GPT 4o mini** (`gpt-4o mini`)

### Local Ollama Models
They are all pulled automatically from Ollama provided it is up and running. 
- **Llama 3.2 1B** (`llama3.2:1b`)
- **Phi 3 Mini** (`phi3:mini`)
- **Gemma 3 variants** (`gemma3:1b`, `gemma3:4b`)
- **Granite 3.3 2B** (`granite3.3:2b`)
- **SmolLM2 135M** (`smollm2:135m`)
- **Qwen 2.5-VL 3B** (`qwen2.5vl:3b`)

## 📊 Data Extraction Schema

The system extracts structured genetic information including:

### Clinical Metadata
- Patient demographics and identifiers
- Laboratory and ordering clinic information
- Report dates and testing methodology
- Sample types and analysis scope

### Genomic Variants
- Gene symbols and transcript IDs
- HGVS nomenclature (c., g., p. notation)
- Variant classifications and interpretations
- Population frequencies and clinical significance
- Reference genome coordinates

### Quality Metrics
- Coverage statistics
- Sequencing quality indicators
- Analysis confidence levels

## 📊 Advanced Performance Evaluation System

### Comprehensive Scoring with compareJSON.py

The project includes a sophisticated evaluation system that provides detailed accuracy analysis across multiple dimensions:

#### Key Features

**Multi-Level Comparison**:
- **Exact Matching**: Perfect field-by-field comparison
- **Key Structure Analysis**: Focuses on presence of required fields
- **Value-Level Accuracy**: Compares actual extracted values
- **Case-Insensitive Matching**: Handles string case variations
- **Type-Flexible Comparison**: Automatically handles numeric vs string differences

**Advanced Metrics**:
- **Total Key Count**: All keys at all nesting levels
- **String Value Keys**: Only keys with extractable string values
- **Missing Field Detection**: Identifies gaps in extraction
- **Error Classification**: Categorizes types of extraction failures

#### Running Performance Analysis

```bash
# After running LLM extractions
cd getJSON
python compareJSON.py
```

#### Sample Output Analysis Still a Work in Progress

```
Template loaded successfully.
Template has 45 total keys and 28 string value keys

--- fakeHospital2_google_gemini-2.0-flash-exp_free__response.json ---
Template: 45 total keys, 28 string keys
Data:     42 total keys, 25 string keys
✗ Found 3 differences out of 28 comparable string values
Accuracy: 89.3%

--- fakeHospital2_meta-llama_llama-4-scout_free__response.json ---
Template: 45 total keys, 28 string keys  
Data:     38 total keys, 22 string keys
✗ Found 8 differences out of 28 comparable string values
Accuracy: 71.4%
```

#### Performance Interpretation

**Accuracy Ranges**:
- **90-100%**: Excellent extraction, production-ready
- **80-89%**: Good extraction, minor formatting issues
- **70-79%**: Acceptable extraction, some field gaps
- **60-69%**: Poor extraction, significant issues
- **<60%**: Inadequate extraction, major problems

### Provider-Specific Analysis

The system is capable of analyzing specific provider types and ranking based on accuracy. 

### Error Pattern Analysis
## TODO
**Common Extraction Challenges**:
- **Date Formatting**: Inconsistent date representation across models
- **HGVS Nomenclature**: Complex genetic notation extraction errors
- **Nested JSON Structure**: Difficulty with deeply nested genetic variant arrays
- **Numeric Precision**: Allele frequency precision variations
- **Missing Optional Fields**: Selective field extraction by different models

## ⚙️ Configuration & Customization

### API Configuration

### Create an environment variable to store your api keys
However, you can also create a txt file and more them over
Create `api_keys.txt` with your credentials:
```
OPENROUTER_API_KEY=your-openrouter-key
OPENAI_API_KEY=your-openai-key
```


### Model Selection

**OpenRouter Models** (`openRouterLLMs.py`):
```python
MODELS = [
    "google/gemini-2.0-flash-exp:free",
    "meta-llama/llama-4-scout:free",
    # Add new models here
]
```

**Ollama Models** (`jsonOllama.py`):
```python
MODELS = [
    "llama3.2:1b",
    "phi3:mini", 
    # Add locally installed models
]
```

### Report Templates

**Hospital Format Selection**:
Modify the comparison script to focus on specific hospital formats:
```python
# In compareJSON.py main() function
json_files = [f for f in os.listdir(direc) 
              if f.endswith('.json') and f.__contains__('fakeHospital1')]
```

**Custom Gene Panels**:
Edit `makeTemplatePDF/data/field_values.yml` to customize:
- Target gene lists for different genetic conditions
- Clinical indication categories
- Report complexity levels

## 🔧 Advanced Usage Patterns

### Batch Processing Multiple Models
```bash
# Run all providers simultaneously
./run_all_llms.sh

# Or run specific providers
python getJSON/openRouterLLMs.py    # Cloud models
python getJSON/jsonOllama.py        # Local models (will automatically pull)
python getJSON/openAItoJSON.py      # OpenAI models
```

### Custom Evaluation Scenarios
```python
# Focus on specific model comparison
# Modify compareJSON.py to filter by model type:
json_files = [f for f in json_files if 'gemini' in f or 'llama' in f]

# Analyze specific extraction challenges
template_filtered = filter_complex_variants_only(template)
accuracy = compare_with_template_keys_only(template_filtered, extracted_data)
```

### Error Analysis & Debugging
```python
# In compareJSON.py, enable detailed difference reporting:
for i, difference in enumerate(differences):
    print(f"  {i+1}. {difference}")
```

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| LaTeX compilation errors | Install full LaTeX distribution (e.g., MacTeX, TeXLive) |
| BioMart API timeouts | Script includes automatic retry logic |
| LLM rate limits | Built-in exponential backoff handles rate limiting |
| JSON parsing failures | Raw responses saved for debugging |
| Missing dependencies | Run `pip install -r requirements.txt` |

## 🤝 Contributing

### Adding New Providers
1. Create new extraction script following the pattern of existing providers
2. Ensure output format matches the standard JSON structure
3. Add provider-specific output directory to `outJSON/`
4. Update `compareJSON.py` to include new provider in analysis

### Enhancing Evaluation Metrics
1. Extend `compareJSON.py` with new scoring algorithms
2. Add provider-specific performance characteristics
3. Implement field-importance weighting for clinical relevance

### Improving Model Coverage
1. Add new models to provider-specific scripts
2. Test context length requirements for genetic reports
3. Validate JSON schema compliance across models

## 📈 Research Applications

This framework is designed for:
- **Clinical AI Research**: Evaluating LLMs for healthcare applications
- **Model Comparison Studies**: Systematic evaluation across providers
- **Prompt Engineering**: Optimizing extraction accuracy through prompt refinement
- **Deployment Planning**: Comparing cloud vs local model performance
- **Cost-Benefit Analysis**: Evaluating accuracy vs computational cost trade-offs

---

**Note**: This framework generates synthetic data only. No real patient information is used or processed. Ensure compliance with relevant data handling policies and API terms of service.
