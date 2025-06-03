#!/usr/bin/env bash

# Wrapper script to generate mock data and PDFs
# Run this from the main PDF_benchmarking directory

echo "PDF Benchmarking - Automatic Report Generator"
echo "=============================================="

# Check if we're in the right directory
if [[ ! -d "scripts" ]] || [[ ! -d "templates" ]]; then
    echo "ERROR: Must run from PDF_benchmarking root directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Change to scripts directory and run the pipeline
cd scripts

echo "Changing to scripts directory..."
echo "Running pipeline to generate 3 sets of mock data and PDFs..."
echo ""

# Execute the main pipeline
./run.sh "$@"

# Return to original directory
cd ..

echo ""
echo "Pipeline complete! Check the scripts/out/ directory for generated PDFs."