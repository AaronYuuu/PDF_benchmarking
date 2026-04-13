#!/bin/bash
set -euo pipefail

# Run from anywhere; script resolves repo root automatically.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "Starting full benchmarking pipeline..."
echo "======================================="

echo "1) OCR: PDF to text/images"
echo "--------------------------"
python3 getJSON/pdfToText.py
echo ""

echo "2) OpenAI extraction"
echo "--------------------"
python3 getJSON/run_models/openAItoJSON.py
echo ""

echo "3) Ollama extraction"
echo "--------------------"
python3 getJSON/run_models/jsonOllama.py
echo ""

echo "4) Local transformer extraction"
echo "-------------------------------"
python3 getJSON/run_models/localLLM.py
echo ""

echo "5) GLiNER extraction"
echo "--------------------"
python3 getJSON/run_models/glinerJSON.py
echo ""

echo "6) Aggregate + score (iTT + parse rate)"
echo "---------------------------------------"
python3 getJSON/compareJSON.py
echo ""

echo "7) Figures + statistics tables"
echo "------------------------------"
Rscript graphs/graphs.R
echo ""

echo "======================================="
echo "Pipeline completed."
echo "Key outputs:"
echo "- graphs/Hospitalfinal.csv"
echo "- graphs/Hospitalfinal_summary.csv"
echo "- graphs/stats_primary_table.csv"
echo "- graphs/stats_mwu_sensitivity_table.csv"
echo "- graphs/Supplementary_ParsedOnly_Accuracy.png"