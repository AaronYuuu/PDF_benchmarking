# PDF Benchmarking for LLM Genetic Report Extraction

A benchmarking framework that evaluates Large Language Model (LLM) performance on extracting structured data from genetic laboratory reports. The system generates synthetic genetic reports as PDFs and tests various LLMs' ability to extract accurate structured information.

## Purpose

This project evaluates how well different LLMs can process complex genetic laboratory reports and extract structured clinical data. It provides:

- Realistic test data using synthetic genetic reports
- Multi-provider evaluation via HuggingFace, OpenAI, and local Ollama instances
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
OPENAI_API_KEY=your-openai-key-here
```
### Run Benchmarking (Current Pipeline)

Run from repository root:

```bash
# 1) OCR from generated PDFs -> output_pdfs/text and output_pdfs/images
python3 getJSON/pdfToText.py

# 2) Model extraction runs (OpenRouter is deprecated and not used)
python3 getJSON/run_models/openAItoJSON.py
python3 getJSON/run_models/jsonOllama.py
python3 getJSON/run_models/localLLM.py
python3 getJSON/run_models/glinerJSON.py

# 3) Evaluation and aggregation
python3 getJSON/compareJSON.py

# 4) Figures + statistics exports
Rscript graphs/graphs.R
```

### Review Results

- Raw outputs: `getJSON/outJSON/`
- Document-level metrics: `graphs/Hospitalfinal.csv`
- Summary metrics (parse-rate + iTT + parsed-only means): `graphs/Hospitalfinal_summary.csv`
- Primary mixed-effects stats (BH-adjusted): `graphs/stats_primary_table.csv`
- MWU sensitivity stats (BH-adjusted): `graphs/stats_mwu_sensitivity_table.csv`
- Supplementary parsed-only accuracy figure: `graphs/Supplementary_ParsedOnly_Accuracy.png`
- Generated reports and OCR outputs: `output_pdfs/`

## Project Structure

```
PDF_benchmarking/
├── makeTemplatePDF/              # Report generation system
│   ├── data/                     # Configuration and gene data
│   ├── scripts/                  # R generation scripts
│   └── templates/                # LaTeX report templates
├── getJSON/                      # LLM processing and evaluation
│   ├── run_models/
│   │   ├── openAItoJSON.py       # OpenAI interface
│   │   ├── jsonOllama.py         # Ollama interface
│   │   ├── localLLM.py           # Local HuggingFace-style models
│   │   └── glinerJSON.py         # GLiNER baseline extraction
│   ├── pdfToText.py              # OCR + image extraction from PDFs
│   ├── compareJSON.py            # Performance evaluation (iTT + parse-rate)
│   └── outJSON/                  # LLM results
├── graphs/                       # Plotting and statistical analysis
│   └── graphs.R                  # Figures + mixed-effects + sensitivity stats
├── output_pdfs/                  # Generated reports
├── api_keys.txt                  # API credentials
├── requirements.txt              # Python dependencies
└── README.md                     # Documentation
```

## Results

The system generates:
- Intent-to-treat (iTT) document-level F1 (unparseable outputs scored as 0)
- Parsed-only secondary F1/accuracy summaries
- Parse success rates by model/prompt/input condition
- Mixed-effects primary inference tables with BH correction
- Separate MWU sensitivity analysis table

## Configuration

### Adding Models

Edit the respective Python files:
- `getJSON/run_models/jsonOllama.py` for Ollama models
- `getJSON/run_models/openAItoJSON.py` for OpenAI models
- `getJSON/run_models/localLLM.py` for local transformer models
- `getJSON/run_models/glinerJSON.py` for GLiNER variants

Note: OpenRouter models are no longer part of the active benchmark pipeline.

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

## Reviewer Feedback Change Log

The table below maps major reviewer feedback items to implemented code changes.

| Reviewer item | Requested revision | Implemented change | File(s) |
|---|---|---|---|
| #6 / Reviewer N #2 | Score unparseable outputs as failures in headline metric | Intent-to-treat scoring added (`F1score=0` when parse fails), with explicit `Parsed` flag and `RunStatus` | `getJSON/compareJSON.py` |
| #6 / Reviewer N #2 | Report parse reliability separately | Added parse-rate summaries (`Hospitalfinal_summary.csv`) including iTT and parsed-only means | `getJSON/compareJSON.py` |
| #7 | Clarify partial-match TP/FN/FP math, including hallucinated tokens | Added explicit partial scoring (`TP += alpha`, `FN += 1-alpha`) and hallucination FP penalty based on unmatched token fraction | `getJSON/compareJSON.py` |
| Reviewer M #3 | Improve statistics beyond MWU-only | Added mixed-effects primary analysis (`lme4`) + `emmeans` contrasts + BH correction | `graphs/graphs.R` |
| Reviewer M #3 | Keep MWU as sensitivity analysis only | Added separate MWU sensitivity export with BH correction; removed legacy hard-coded MWU plotting tests | `graphs/graphs.R` |
| Supplement request | Add non-iTT parsed-only figure | Added parsed-only accuracy summary + supplementary figure export | `graphs/graphs.R` |
| Pipeline update | OpenRouter removed from active run path | Updated docs and run script to exclude OpenRouter from active benchmark reruns | `README.md`, `run_all_llms.sh` |

## Run Entire Pipeline (`everything.sh`)

Use `everything.sh` for a full clean rerun (report generation + extraction + scoring + stats/figures):

```bash
cd PDF_benchmarking
chmod +x everything.sh generate_reports.sh run_all_llms.sh
./everything.sh
```

What `everything.sh` does:

1. Cleans prior outputs under `getJSON/outJSON/`, `makeTemplatePDF/out/`, and `output_pdfs/`
2. Runs `./generate_reports.sh` to regenerate mock data and PDFs
3. Runs `./run_all_llms.sh` to execute OCR, model extraction, scoring, and graph/stat generation

Important notes:
- Start in the repository root (`PDF_benchmarking`) or the script will exit.
