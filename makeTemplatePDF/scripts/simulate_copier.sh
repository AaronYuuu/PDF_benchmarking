#!/usr/bin/env bash

INPDF="$1"
OUTPDF="${INPDF%.pdf}_distressed.pdf"

#check that imagemagick is installed and the input PDF file exists
if [[ -z $(which magick) ]]; then
  echo "Error: ImageMagick is not installed or not found in PATH."
  exit 1
fi
if [[ ! -f "${INPDF}" ]]; then
  echo "Error: Input PDF file '${INPDF}' does not exist."
  exit 1
fi

# Helper function to generate gaussian distributed random numbers
rnorm() {
  SD=$1
  Rscript -e "rnorm(1,0,${SD})|>format(digits=2)|>cat('\n')"
}
# Helper function to generate uniformly distributed random numbers
runif() {
  MIN=$1
  MAX=$2
  Rscript -e "runif(1,${MIN},${MAX})|>format(digits=2)|>cat('\n')"
}
# Helper function to generate random integers in a range
rint() {
  MIN=$1
  MAX=$2
  echo $RANDOM % $((MAX - MIN + 1)) + $MIN | bc
}

# Randomly pick parameters for the distressing effect
BLUR_RADIUS=$(rint 0 2)
BLUR_SIGMA=$(runif 0.5 1.5)
STRETCH_PERCENT=$(rint 1 7)
ROTATE_DEGREES=$(runif "-1" "1")
# ROTATE_DEGREES=$(rnorm 0.2)
NOISE_INTENSITY=$(runif 0.1 2)
# NOISE_INTENSITY=$(runif 1 10)

# Use ImageMagick to apply the distressing effect
magick -density 200 "${INPDF}" -colorspace gray \
  -linear-stretch "${STRETCH_PERCENT}%x10%" -rotate "${ROTATE_DEGREES}" \
  -repage +0 -blur ${BLUR_RADIUS}x${BLUR_SIGMA} \
  -attenuate ${NOISE_INTENSITY} +noise poisson "${OUTPDF}"

echo "Success!"
