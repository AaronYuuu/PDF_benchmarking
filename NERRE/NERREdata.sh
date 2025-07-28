#!/bin/bash
for i in {1..10}; do
    ./generate_reports.sh
    python3 /Users/ayu/PDF_benchmarking/getJSON/makeData2.py 
done