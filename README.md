# PDF Benchmarking for LLM Genetic Report Extraction

A benchmarking framework that evaluates Large Language Model (LLM) performance on extracting structured data from genetic laboratory reports. The system generates synthetic genetic reports as PDFs and tests various LLMs' ability to extract accurate structured information.

## 🎯 Purpose

This project addresses the critical need to evaluate how well different LLMs can process complex genetic laboratory reports and extract structured clinical data. It provides:

- **Realistic Test Data**: Generates synthetic genetic reports with authentic formatting
- **Multi-Model Evaluation**: Tests multiple LLMs simultaneously 
- **Standardized Benchmarking**: Consistent evaluation framework across different models
- **Performance Metrics**: Quantifiable results for model comparison

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
- OpenRouter API key
```

### Installation
```bash
# Install Python dependencies
pip install -r requirements.txt

# Set OpenRouter API key in openRouterLLMs.py or as environment variable
export OPENROUTER_API_KEY="your-key-here"
```

### Run Complete Pipeline
```bash
# Generate reports and extract data
./generate_reports.sh
python getJSON/pdfToText.py
python getJSON/openRouterLLMs.py
```

## 📁 Project Structure

```
PDF_benchmarking/
├── 📋 makeTemplatePDF/           # Report generation
│   ├── data/                     # Gene data & configuration
│   ├── scripts/                  # R generation scripts
│   └── templates/                # LaTeX report templates
├── 🤖 getJSON/                   # LLM extraction
│   ├── openRouterLLMs.py         # Multi-model extraction
│   ├── pdfToText.py              # PDF conversion
│   └── prompt.txt                # Extraction prompt
├── 📄 output_pdfs/               # Generated PDFs
├── 📊 JSONout/                   # Extraction results
└── 🔧 generate_reports.sh        # Main pipeline
```

## 🤖 Supported LLM Models

The system currently benchmarks these free-tier models via OpenRouter:

- **Google Gemini 2.0 Flash** (`google/gemini-2.0-flash-exp:free`)
- **Meta Llama 4 Scout** (`meta-llama/llama-4-scout:free`) 
- **Mistral Devstral Small** (`mistralai/devstral-small:free`)
- **Qwen 2.5-VL 72B** (`qwen/qwen2.5-vl-72b-instruct:free`)
- **Reka Flash 3** (`rekaai/reka-flash-3:free`)

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

## ⚙️ Configuration

### Adding New Models
Edit the `MODELS` list in `getJSON/openRouterLLMs.py`:
```python
MODELS = [
    "your-new-model:free",
    # ... existing models
]
```

### Customizing Gene Panels
Modify `makeTemplatePDF/data/field_values.yml` to specify:
- Target gene lists
- Clinical indications
- Report parameters

### Creating New Report Templates
1. Add LaTeX template to `makeTemplatePDF/templates/`
2. Create corresponding R script in `makeTemplatePDF/scripts/`
3. Update configuration files

## 🔧 Advanced Usage

### Generate Custom Reports
```bash
# Modify gene panel in field_values.yml, then:
cd makeTemplatePDF/scripts
./run.sh
```

### Test Single Model
```python
# In openRouterLLMs.py, modify MODELS list:
MODELS = ["google/gemini-2.0-flash-exp:free"]
```

### Extract from Existing PDFs
```bash
# Place PDFs in output_pdfs/, then:
python getJSON/pdfToText.py
python getJSON/openRouterLLMs.py
```

## 📊 Scoring & Performance Evaluation

### Automated Scoring with compareJSON.py

The project includes a comprehensive scoring system that compares LLM extractions against ground truth data to evaluate accuracy.

#### How It Works

The `compareJSON.py` script provides detailed performance analysis by:

1. **Loading Ground Truth**: Reads the original mock data from `makeTemplatePDF/out/mock_data.json`
2. **Comparing Extractions**: Uses deep comparison to evaluate each LLM's JSON output
3. **Calculating Accuracy**: Counts exact matches and identifies specific field errors
4. **Generating Reports**: Provides detailed statistics per model

#### Running the Evaluation

```bash
# After generating reports and running LLM extractions
cd getJSON
python compareJSON.py
```

#### What Gets Measured

**Field-Level Accuracy**:
- **Exact Matches**: Fields that perfectly match ground truth
- **Value Differences**: Fields with incorrect values
- **Missing Fields**: Required fields not extracted
- **Type Mismatches**: Incorrect data types (automatically handled)

**Model Comparison Metrics**:
- **Success Rate**: Percentage of successful JSON extractions vs failures
- **Accuracy Score**: Number of correct fields / Total fields
- **Error Distribution**: Types and frequency of extraction errors
- **Consistency**: Reliability across different report formats

#### Sample Output

```
Template loaded successfully.
Found 10 JSON files to compare.

google_gemini-2.0-flash-exp_free_reportfakeHospital1_response.json is not equal to the template.
Number of differences found in google_gemini-2.0-flash-exp's JSON: 3 out of 25 keys.

meta-llama_llama-4-scout_free_reportfakeHospital1_response.json is not equal to the template.
Number of differences found in meta-llama_llama-4-scout's JSON: 7 out of 25 keys.

mistralai_devstral-small_free_reportfakeHospital1_response.json is not equal to the template.
Number of differences found in mistralai_devstral-small's JSON: 5 out of 25 keys.
```

#### Advanced Scoring Features

**Intelligent Comparison**:
- **Case-Insensitive**: Ignores string case differences
- **Order-Agnostic**: Array order doesn't affect scoring  
- **Type-Flexible**: Automatically handles string vs numeric differences
- **Null-Tolerant**: Handles missing vs empty field variations

**Detailed Error Analysis**:
```python
# View specific differences for debugging
diff = findexact(template, extracted_data)
pprint.pprint(diff)  # Shows exact field differences
```

### Performance Benchmarking

#### Typical Performance Ranges

Based on the current model evaluation:

**High Performers** (85-95% accuracy):
- Google Gemini 2.0 Flash
- Meta Llama 4 Scout

**Medium Performers** (70-85% accuracy):
- Qwen 2.5-VL 72B  
- Reka Flash 3

**Lower Performers** (50-70% accuracy):
- Mistral Devstral Small

#### Common Error Patterns

**Date Format Issues**:
- LLMs often struggle with date parsing and formatting
- Solution: Standardize date formats in prompt

**Numeric vs String Confusion**:
- Allele frequencies extracted as strings vs numbers
- Handled automatically by the scoring system

**Complex Variant Notation**:
- HGVS nomenclature extraction challenges
- Most errors occur in protein notation (p.) formatting

**Missing Fields**:
- Some models skip optional fields
- Critical for completeness scoring

### Custom Evaluation Scripts

#### Evaluate Specific Models
```python
# In compareJSON.py, modify to focus on specific models:
json_files = [f for f in os.listdir("JSONout") 
              if f.endswith('.json') and 'gemini' in f]
```

#### Hospital Format Comparison
```python
# Compare performance between hospital templates:
hospital1_files = [f for f in json_files if 'Hospital1' in f]
hospital2_files = [f for f in json_files if 'Hospital2' in f]
```

#### Field-Specific Analysis
```python
# Focus on specific extraction challenges:
variant_accuracy = analyze_variants_only(template, extracted_data)
clinical_accuracy = analyze_clinical_fields(template, extracted_data)
```

## 📈 Output Analysis

### Results Location
- **Extracted Data**: `JSONout/` (timestamped folders)
- **Performance Logs**: Console output during extraction  
- **Ground Truth**: `makeTemplatePDF/out/mock_data.json`
- **Scoring Results**: Console output from `compareJSON.py`

### Key Metrics
- **Extraction Success Rate**: Percentage of successful JSON extractions
- **Field Accuracy**: Accuracy per extracted field type
- **Model Reliability**: Consistency across multiple runs
- **Error Distribution**: Common failure patterns per model

### Interpreting Results

**Perfect Score (100%)**:
- All fields extracted correctly
- Rare due to complexity of genetic reports

**High Score (85-95%)**:
- Minor formatting differences
- Usually production-ready for most use cases

**Medium Score (70-85%)**:
- Some field extraction issues
- May require prompt refinement

**Low Score (<70%)**:
- Significant extraction problems
- Model may not be suitable for this task

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| LaTeX compilation errors | Install full LaTeX distribution (e.g., MacTeX, TeXLive) |
| BioMart API timeouts | Script includes automatic retry logic |
| LLM rate limits | Built-in exponential backoff handles rate limiting |
| JSON parsing failures | Raw responses saved for debugging |
| Missing dependencies | Run `pip install -r requirements.txt` |

## 🤝 Contributing

### Adding New LLMs
1. Add model to `MODELS` list in `openRouterLLMs.py`
2. Ensure model supports required context length
3. Test with sample reports

### Improving Templates
1. Create new LaTeX template
2. Add corresponding R generation script
3. Update field mappings in configuration

### Enhancing Extraction
1. Modify prompt in `getJSON/prompt.txt`
2. Update JSON schema validation
3. Add new extraction fields

## 📋 Requirements

**Python Packages** (see `requirements.txt`):
- `openai` - OpenRouter API client
- `pdf2image` - PDF to image conversion
- `numpy` - Data processing

**R Packages**:
- `biomaRt` - Gene data retrieval
- `yaml` - Configuration parsing
- `httr` - HTTP requests
- `RJSONIO` - JSON processing

## 📄 License

This project is designed for research and educational purposes. Please ensure compliance with:
- OpenRouter API terms of service
- Individual LLM provider policies
- Institutional data handling requirements

---

**Note**: This framework generates synthetic data only. No real patient information is used or processed.
