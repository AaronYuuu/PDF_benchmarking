echo "Must start in PDF_benchmarking directory"
# Check if we're in the right directory
if [ ! -d "getJSON" ]; then
    echo "Directory getJSON not found! Exiting. Must start in PDF_benchmarking"
    exit 1
fi
# Run script to generate mock data and PDFs
generate_reports.sh
# Run script to process LLMs
run_all_llms.sh
