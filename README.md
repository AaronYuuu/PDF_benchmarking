# PDF Benchmarking

A comprehensive benchmarking system for evaluating Large Language Model (LLM) performance on genetic report data extraction. This project generates synthetic genetic laboratory reports as PDFs, extracts structured data using various LLMs, and validates extraction accuracy.

## Overview

This project benchmarks LLM capabilities for extracting structured clinical and genomic information from genetic laboratory reports. It consists of three main components:

1. **Report Generation**: Creates realistic mock genetic reports using R templates
2. **Data Extraction**: Uses multiple LLMs to extract structured JSON data from PDF reports
3. **Validation**: Compares extracted data against ground truth for accuracy assessment

## Project Structure

```
├── makeTemplatePDF/          # Report generation system
│   ├── data/                 # Gene information and configuration
│   ├── scripts/              # R scripts for data generation and PDF creation
│   └── templates/            # LaTeX templates for different hospital formats
├── getJSON/                  # LLM extraction system
│   ├── openRouterLLMs.py     # Multi-model LLM extraction script
│   ├── pdfToText.py          # PDF to text conversion
│   ├── filterJSON.py         # JSON processing utilities
│   └── prompt.txt            # Extraction prompt template
├── output_pdfs/              # Generated PDF reports and extracted text
│   ├── images/               # PDF page images
│   └── text/                 # Extracted text files
├── JSONout*/                 # LLM extraction results (timestamped folders)
└── generate_reports.sh       # Main pipeline script
```

## Prerequisites

- **R 4.0+** with required packages:
  - `yaml`, `biomaRt`, `httr`, `RJSONIO`
  - LaTeX distribution (for PDF generation)
- **Python 3.8+** with packages:
  - `openai`, `numpy`, `pdf2image`
- **API Access**: OpenRouter API key for LLM access

## Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd PDF_benchmarking

# Install Python dependencies
pip install -r requirements.txt

# Set your OpenRouter API key
export OPENROUTER_API_KEY="your-api-key-here"
```

### 2. Generate Mock Reports

```bash
# Generate synthetic genetic reports
./generate_reports.sh
```

This creates:
- Mock genetic data with realistic variants
- PDF reports in different hospital formats
- Ground truth JSON data for validation

### 3. Extract Data with LLMs

```bash
# Convert PDFs to text
python getJSON/pdfToText.py

# Run multi-model extraction
python getJSON/openRouterLLMs.py
```

## Detailed Workflow

### Report Generation Pipeline

The system generates realistic genetic laboratory reports through:

1. **Gene Data Fetching** (`fetch_gene_info.R`):
   - Queries Ensembl BioMart for gene information
   - Retrieves RefSeq transcript IDs and genomic coordinates
   - Generates exon boundary data

2. **Mock Data Generation** (`generate_mock_data.R`):
   - Creates synthetic genetic variants with realistic frequencies
   - Generates clinical metadata (dates, patient info, etc.)
   - Produces HGVS nomenclature for variants

3. **PDF Creation**:
   - Hospital-specific LaTeX templates (`fakeHospital1.tex`, `fakeHospital2.tex`)
   - R scripts populate templates with generated data
   - Outputs professional-looking genetic reports

### LLM Extraction System

The extraction system evaluates multiple LLM models:

- **Google Gemini 2.0 Flash** (free tier)
- **Qwen 2.5-VL 72B** (free tier)  
- **Meta Llama 4 Scout** (free tier)
- **Mistral Devstral Small** (free tier)
- **Reka Flash 3** (free tier)

**Extraction Process**:
1. PDFs converted to text using `pdf2image` and OCR
2. Structured prompt guides LLMs to extract specific data fields
3. JSON responses saved with model metadata
4. Fallback handling for failed extractions

### Data Schema

The system extracts structured data including:

**Clinical Information**:
- Sequencing scope and methodology
- Sample types and analysis methods
- Report dates and laboratory details

**Genomic Variants**:
- HGVS nomenclature (c., g., p. notations)
- Gene symbols and transcript IDs
- Variant interpretations and classifications
- Population frequencies and clinical significance

**Metadata**:
- Report types and testing contexts
- Ordering clinics and laboratories
- Reference genome versions

## Configuration

### Supported Models

Models are configured in `openRouterLLMs.py`. The system includes:
- Automatic retry logic for rate limits
- Exponential backoff for failed requests
- JSON extraction and validation
- Error handling and logging

### Hospital Templates

Two template formats are available:
- **fakeHospital1**: Traditional clinical report format
- **fakeHospital2**: Modern tabular layout

Templates can be customized in `makeTemplatePDF/templates/`.

### Gene Panel Configuration

Gene lists and clinical parameters are configured in:
- `makeTemplatePDF/data/field_values.yml`
- `makeTemplatePDF/data/gene_info.csv`

## Output and Results

### Generated Reports
- PDF files in `output_pdfs/`
- Page images in `output_pdfs/images/`
- Extracted text in `output_pdfs/text/`

### LLM Extraction Results
- JSON responses in timestamped `JSONout*/` folders
- Model performance metadata
- Success/failure statistics
- Raw responses for debugging

### Ground Truth Data
- Original mock data in JSON format
- Variant details and clinical annotations
- Reference data for accuracy validation

## Usage Examples

### Generate 5 reports and run full extraction:
```bash
./generate_reports.sh
python getJSON/pdfToText.py
python getJSON/openRouterLLMs.py
```

### Custom gene panel:
Edit `makeTemplatePDF/data/field_values.yml` to specify different genes, then regenerate reports.

### Single model testing:
Modify the `MODELS` list in `openRouterLLMs.py` to test specific LLM models.

## Troubleshooting

**Common Issues**:
- **LaTeX errors**: Ensure full LaTeX distribution is installed
- **BioMart timeouts**: Script includes retry logic for API failures
- **API rate limits**: Built-in exponential backoff handling
- **JSON parsing errors**: Fallback saves raw responses for debugging

**Log Files**:
Check console output for detailed error messages and processing status.

## Contributing

To add new hospital templates:
1. Create LaTeX template in `makeTemplatePDF/templates/`
2. Add corresponding R script in `makeTemplatePDF/scripts/`
3. Update field mappings in data configuration files

To add new LLM models:
1. Add model identifier to `MODELS` list in `openRouterLLMs.py`
2. Ensure model supports the required context length
3. Test extraction performance with sample reports

## License

This project is designed for research and benchmarking purposes. Ensure compliance with API terms of service for LLM providers.
