#!/usr/bin/env bash
set -euo pipefail


#helper function to print usage information
usage () {
  cat << EOF

run.sh v0.0.1 

by Jochen Weile <jweile@oicr.on.ca> 2025

This script generates a mock report using R scripts and LaTeX.

Usage: run.sh [-a|--amount <INTEGER>] [-o|--outdir <DIR>] [-t|--templates <TEMPLATE1,TEMPLATE2,...>] [<TEMPLATE>]

<TEMPLATE>  : Single template file (for backwards compatibility)
-a|--amount : The number of mock data entries to generate (default: 10)
-o|--outdir : The output directory where the reports will be saved (default: out/)
-t|--templates : Comma-separated list of template files to process with the same mock data

Examples:
  run.sh -t ../templates/fakeHospital1.tex,../templates/fakeHospital2.tex
  run.sh ../templates/fakeHospital1.tex  # backwards compatible single template

EOF
 exit $1
}

#Parse Arguments
PARAMS=""
mkdir -p "../out/"
OUTDIR="../out/"
mkdir -p "$OUTDIR"
AMOUNT=1
TEMPLATES=""
while (( "$#" )); do
  case "$1" in
    -h|--help)
      usage 0
      shift
      ;;
    -a|--amount)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        AMOUNT="$2"
        shift 2
      else
        echo "ERROR: Argument for $1 is missing" >&2
        usage 1
      fi
      ;;
    -o|--outdir)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        OUTDIR="$2"
        shift 2
      else
        echo "ERROR: Argument for $1 is missing" >&2
        usage 1
      fi
      ;;
    -t|--templates)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        TEMPLATES="$2"
        shift 2
      else
        echo "ERROR: Argument for $1 is missing" >&2
        usage 1
      fi
      ;;
    --) # end of options indicates that the main command follows
      shift
      PARAMS="$PARAMS $@"
      eval set -- ""
      ;;
    -*|--*=) # unsupported flags
      echo "ERROR: Unsupported flag $1" >&2
      usage 1
      ;;
    *) # positional parameter
      PARAMS="$PARAMS $1"
      shift
      ;;
  esac
done

eval set -- "$PARAMS"

# Dynamically discover all .tex files in the templates directory
TEMPLATE_ARRAY=()
while IFS= read -r -d '' template; do
    TEMPLATE_ARRAY+=("$template")
done < <(find ../templates -name "*.tex" -print0)

echo "Output directory: $OUTDIR"
echo "Found ${#TEMPLATE_ARRAY[@]} templates to process"
DATA="$OUTDIR/mock_data.json"

# Generate mock data using the AMOUNT variable (only once)
echo "Generating mock data files..."
Rscript generate_mock_data.R --amount "$AMOUNT" --outfile "$DATA"

# Process all templates with the same JSON data using the --all_templates flag
echo "Interpolating JSON files with all templates..."
Rscript interpolate.R "$DATA" --all_templates --outprefix "$OUTDIR/report_"

cd "$OUTDIR"
# Compile all generated .tex files into PDFs and clean up
echo "Compiling LaTeX files into PDFs..."
for TEXFILE in report_*.tex; do
  echo "Compiling $TEXFILE"
  pdflatex -halt-on-error -interaction batchmode "$TEXFILE" && \
  rm "${TEXFILE%.tex}.aux" "${TEXFILE%.tex}.log" "${TEXFILE%.tex}.tex"
done
cd ../scripts
for PDF in "$OUTDIR"/report_*.pdf; do
  echo "Distressing $PDF"
  ./simulate_copier.sh "$PDF"
done
echo "All done."
