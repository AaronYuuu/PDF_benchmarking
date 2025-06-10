#!/bin/bash

# Script to run all LLM processing scripts
# This will run jsonOllama, openAItoJSON, and openRouterLLMs scripts

echo "Starting LLM processing pipeline..."
echo "======================================="

# Change to the getJSON directory
cd "$(dirname "$0")"

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
python3 openRouterLLMs.py
echo ""

echo "======================================="
echo "All LLM processing scripts completed!"
echo "Check the respective output directories:"
echo "- OllamaOut/ for Ollama results"
echo "- OpenAIOut/ for OpenAI results" 
echo "- JSONout/ for OpenRouter results"