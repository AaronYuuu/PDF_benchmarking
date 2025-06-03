#!/usr/bin/env bash
set -euo pipefail

#helper function to print usage information
usage () {
  cat << EOF

run.sh v0.0.1 

by Jochen Weile <jweile@oicr.on.ca> 2025

This script generates mock reports using R scripts and LaTeX.

Usage: run.sh [-a|--amount <INTEGER>] [-o|--outdir <DIR>]

-a|--amount : The number of mock data entries to generate (default: 10)
-o|--outdir : The output directory where the reports will be saved (default: out/)

This script will generate reports using both fakeHospital1.tex and fakeHospital2.tex templates.

EOF
 exit $1
}

#Parse Arguments
PARAMS=""
OUTDIR="out/"
AMOUNT=1 # Default amount of mock data entries
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
#reset command arguments as only positional parameters
eval set -- "$PARAMS"

# Define templates
TEMPLATE1="../templates/fakeHospital1.tex"
TEMPLATE2="../templates/fakeHospital2.tex"

# Check if both template files exist
if [[ ! -f "$TEMPLATE1" ]]; then
  echo "Template file not found: $TEMPLATE1"
  exit 1
fi

if [[ ! -f "$TEMPLATE2" ]]; then
  echo "Template file not found: $TEMPLATE2"
  exit 1
fi

# Check if the output directory exists, if not create it
mkdir -p "$OUTDIR"

# Define location for mock data
DATA="${OUTDIR}mock_data.json"

# Generate the mock data
Rscript generate_mock_data.R --amount "$AMOUNT" --outfile "$DATA"

# Interpolate both templates with the mock data
echo "Generating reports from fakeHospital1.tex template..."
Rscript interpolate.R "$TEMPLATE1" "$DATA" --outprefix "${OUTDIR}/hospital1_report_"

echo "Generating reports from fakeHospital2.tex template..."
Rscript interpolate.R "$TEMPLATE2" "$DATA" --outprefix "${OUTDIR}/hospital2_report_"

# Compile the interpolated LaTeX files to PDF
cd "$OUTDIR"
for TEXFILE in *.tex; do
  echo " Compiling $TEXFILE to PDF..."
  pdflatex -halt-on-error -interaction batchmode "$TEXFILE" && 
    rm "${TEXFILE%.tex}.aux" "${TEXFILE%.tex}.log" "${TEXFILE%.tex}.tex"
done
cd -