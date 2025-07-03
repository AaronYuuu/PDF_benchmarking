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
    rm -rf getJSON/outJSON/localout/*
    rm -rf getJSON/outJSON/glinerOut/*
    rm -rf getJSON/outJSON/OllamaOutNP/*
    rm -rf getJSON/outJSON/*
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

echo "Cleanup completed!"
echo ""

# Run script to generate mock data and PDFs
echo "Running generate_reports.sh..."
echo "--------------------------------------"
./generate_reports.sh

echo ""
echo "Reports generated successfully!"
echo ""

# Run script to process LLMs
echo "Running run_all_llms.sh..."
echo "--------------------------------------"
./run_all_llms.sh



echo ""
echo "============================================"
echo "Complete pipeline finished successfully!"
echo "============================================"

#echo "Generating visualization report..."
#echo "--------------------------------------"
#cd getJSON
#python3 genGraphs.py
#echo "Visualization report generated successfully!"
#echo "============================================"


