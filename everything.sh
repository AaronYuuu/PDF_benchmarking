#!/bin/bash

echo "PDF Benchmarking - Complete Pipeline Runner"
echo "============================================"
echo "Must start in PDF_benchmarking directory"

# Check if we're in the right directory
if [ ! -d "getJSON" ] || [ ! -d "makeTemplatePDF" ]; then
    echo "Error: Not in the correct directory! Please run from PDF_benchmarking directory."
    exit 1
fi

echo "Cleaning up previous output folders..."
echo "--------------------------------------"

# Remove getJSON output folders
if [ -d "getJSON/outJSON" ]; then
    echo "Removing getJSON/outJSON/*"
    rm -rf getJSON/outJSON/OllamaOut/*
    rm -rf getJSON/outJSON/OllamaVisionOut/*
    rm -rf getJSON/outJSON/OpenAIOut/*
    rm -rf getJSON/outJSON/OpenAIVisionOut/*
    rm -rf getJSON/outJSON/OpenRouter/*
    rm -rf getJSON/outJSON/OpenRouterVisionOut/*
fi

# Remove makeTemplatePDF output folders
if [ -d "makeTemplatePDF/out" ]; then
    echo "Removing makeTemplatePDF/out/*"
    rm -rf makeTemplatePDF/out/*
fi
# Remove output_pdfs folders
if [ -d "output_pdfs" ]; then
    echo "Removing output_pdfs/*"
    rm -rf output_pdfs/images/*
    rm -rf output_pdfs/text/*
fi

# Remove visualization outputs
if [ -d "visualization_outputs" ]; then
    echo "Removing visualization_outputs/*"
    rm -rf visualization_outputs/*
fi

echo "Cleanup completed!"
echo ""

# Run script to generate mock data and PDFs
echo "Running generate_reports.sh..."
# Wrapper script to generate mock data and PDFs
# Run this from the main PDF_benchmarking directory

echo "PDF Benchmarking - Automatic Report Generator"
echo "=============================================="

# Check if we're in the right directory

# Change to scripts directory and run the pipeline
cd makeTemplatePDF
cd scripts

echo "Changing to scripts directory..."
echo "Running pipeline to generate sets of mock data and PDFs..."
echo ""

# Execute the main pipeline
./run.sh "$@"

# Return to original directory
cd ..

echo ""
echo "Pipeline complete! Check the scripts/out/ directory for generated PDFs."

cd ..
echo "=============================================="
echo "All tasks completed successfully!"
echo "You can now review the generated reports in the 'out' directory."
echo "=============================================="
echo "Generating text files for each PDF..."
# Generate text files for each PDF in the out directory
cd getJSON
python3 pdfToText.py
echo "Text files generated successfully!"
echo "=============================================="
echo ""

# Run script to process LLMs
echo "Running run_all_llms.sh..."
echo "Starting LLM processing pipeline..."
echo "======================================="

# Change to the getJSON directory
cd getJSON || { echo "Directory getJSON not found! Exiting. Must start in PDF_benchmarking"; exit 1; }

# Run jsonOllama.py
echo "1. Running Ollama models..."
echo "----------------------------"
python3 jsonOllama.py
echo ""

# Run openAItoJSON.py
echo "2. Running OpenAI models..."
echo "---------------------------"
python3 openAItoJSON.py

echo ""

# Run openRouterLLMs.py
echo "3. Running OpenRouter models..."
echo "-------------------------------"
#python3 openRouterLLMs.py
echo ""

echo "======================================="
echo "All LLM processing scripts completed!"
echo "Check the respective output directories:"
echo "- OllamaOut/ for Ollama results"
echo "- OpenAIOut/ for OpenAI results" 
echo "- JSONout/ for OpenRouter results"

echo "4. Checking accuracy"
echo "---------------------"
# Run accuracy check script
python3 compareJSON.py
echo "Accuracy check completed! Check the Hospital.csv file for results."

echo ""
echo "============================================"
echo "Complete pipeline finished successfully!"
echo "============================================"

echo "Graphing results into the pdf"
python3 genGraphs.py
echo "Graph and final report generated successfully!"
echo "============================================"


