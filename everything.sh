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
./generate_reports.sh

echo ""

# Run script to process LLMs
echo "Running run_all_llms.sh..."
./run_all_llms.sh

echo ""
echo "============================================"
echo "Complete pipeline finished successfully!"
echo "============================================"

echo "Graphing results into the pdf"
python3 genGraphs.py
echo "Graph and final report generated successfully!"
echo "============================================"


