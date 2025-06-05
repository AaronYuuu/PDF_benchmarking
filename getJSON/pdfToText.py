"""OHCRN-LEI - LLM-based Extraction of Information
Copyright (C) 2025 Ontario Institute for Cancer Research

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
import argparse
import glob
from pathlib import Path
from typing import List
import re

def convert_pdf_to_images(pdf_path: str, output_dir: str, dpi=300) -> List[str]:
    """Convert PDF to images and save them.
    
    Args:
        pdf_path: path to the pdf input file
        output_dir: directory to save images
        dpi: The scan resolution to use
        
    Returns:
        List of saved image file paths
    """
    from pdf2image import convert_from_path  # lazy import to speed up app boot time
    # Convert PDF pages to images
    try:
        pages = convert_from_path(pdf_path, dpi=dpi)
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
    # Save images
    pdf_name = Path(pdf_path).stem
    image_paths = []
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i, page in enumerate(pages):
        image_path = os.path.join(output_dir, f"{pdf_name}_page_{i+1}.png")
        page.save(image_path, 'PNG')
        image_paths.append(image_path)
        print(f" - Saved image: {image_path}")
    
    return image_paths


def convert_pdf_to_str_list(pdf_path: str, language_list=["en"], dpi=300) -> List[str]:
  """Use EasyOCR to convert the given PDF file into plain text but page by page.
  Returns a list of strings representing each page.

  Args:
    pdf_path: path to the pdf input file
    language_list: A list of languages to detect for OCR
    dpi: The scan resolution to use for OCR

  Returns:
    A list of text strings representing the document pages

  """
  # Initialize the EasyOCR reader.
  # lazy import to speed up app boot time
  from easyocr import Reader  # type: ignore

  reader = Reader(language_list)

  # Convert PDF pages to images.
  try:
    # lazy import to speed up app boot time
    from pdf2image import convert_from_path

    # You can adjust dpi if necessary.
    pages = convert_from_path(pdf_path, dpi=dpi)
  except Exception as e:
    print(f"Error converting PDF to images: {e}")
    return []

  # lazy import to speed up app boot time
  from numpy import array

  full_text = []
  for i, page in enumerate(pages):
    print(f" - Processing page {i + 1}...")
    # Convert PIL image to numpy array.
    image_np = array(page)

    # Use EasyOCR to extract text from the image.
    try:
      results = reader.readtext(image_np, detail=0, paragraph=True)
      # Join text from detected regions.
      page_text = "\n".join(str(item) for item in results if isinstance(item, str))
      full_text.append(page_text)
    except Exception as e:
      print(f"Failed processing page {i + 1}: {e}", os.EX_DATAERR)

  return full_text


def convert_pdf_to_text(pdf_path: str, language_list=["en"], dpi=300) -> str:
  """Use EasyOCR to convert the given PDF file into plain text.
  Returns a string, with pages separated by two newline characters.

  Args:
    pdf_path: path to the pdf input file
    language_list: A list of languages to detect for OCR
    dpi: The scan resolution to use for OCR

  Returns:
    A text string of the OCR result.

  """
  pages = convert_pdf_to_str_list(pdf_path, language_list, dpi)
  return "\n\n".join(pages)


def process_pdf_batch(input_dir: str, output_dir: str, save_text: bool = True, 
                     save_images: bool = False, language_list=["en"], dpi=300):
    """Process a batch of PDF files.
    
    Args:
        input_dir: Directory containing PDF files
        output_dir: Directory to save outputs
        save_text: Whether to save OCR text files
        save_images: Whether to save page images
        language_list: Languages for OCR
        dpi: Resolution for processing
    """
    # Find all PDF files
    pdf_pattern = os.path.join(input_dir, "*.pdf")
    pdf_files = glob.glob(pdf_pattern)
    
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Create output directories
    if save_text:
        text_output_dir = os.path.join(output_dir, "text")
        os.makedirs(text_output_dir, exist_ok=True)
    
    if save_images:
        image_output_dir = os.path.join(output_dir, "images")
        os.makedirs(image_output_dir, exist_ok=True)
    
    # Process each PDF
    for i, pdf_file in enumerate(pdf_files, 1):
        pdf_name = Path(pdf_file).stem
        print(f"\nProcessing {i}/{len(pdf_files)}: {pdf_name}")
        
        try:
            # Save images if requested
            if save_images:
                convert_pdf_to_images(pdf_file, image_output_dir, dpi)
            
            # Save text if requested
            if save_text:
                print(f" - Extracting text from {pdf_name}...")
                extracted_text = convert_pdf_to_text(pdf_file, language_list, dpi)
                
                output_file = os.path.join(text_output_dir, f"{pdf_name}.txt")
                with open(output_file, 'w', encoding='utf-8') as file:
                    file.write(extracted_text)
                print(f" - Text saved to {output_file}")
                
        except Exception as e:
            print(f" - Error processing {pdf_name}: {e}")
            continue
    
    print(f"\nBatch processing complete! Outputs saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert PDF files to text and/or images")
    parser.add_argument("input", help="Input PDF file or directory containing PDF files")
    parser.add_argument("-o", "--output", default="output", 
                       help="Output directory (default: output)")
    parser.add_argument("--text", action="store_true", default=True,
                       help="Save OCR text files (default: True)")
    parser.add_argument("--images", action="store_true", 
                       help="Save page images")
    parser.add_argument("--no-text", action="store_true",
                       help="Don't save text files")
    parser.add_argument("--dpi", type=int, default=300,
                       help="DPI for image processing (default: 300)")
    parser.add_argument("--languages", nargs="+", default=["en"],
                       help="Languages for OCR (default: en)")
    
    args = parser.parse_args()
    
    # Handle text/no-text flags
    save_text = args.text and not args.no_text
    save_images = args.images
    
    if not save_text and not save_images:
        print("Error: Must specify at least --text or --images")
        return
    
    # Check if input is a file or directory
    if os.path.isfile(args.input) and args.input.endswith('.pdf'):
        # Single file processing
        pdf_name = Path(args.input).stem
        os.makedirs(args.output, exist_ok=True)
        
        print(f"Processing single file: {pdf_name}")
        
        try:
            if save_images:
                image_dir = os.path.join(args.output, "images")
                convert_pdf_to_images(args.input, image_dir, args.dpi)
            
            if save_text:
                text = convert_pdf_to_text(args.input, args.languages, args.dpi)
                text_file = os.path.join(args.output, f"{pdf_name}.txt")
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(text)
                print(f"Text saved to {text_file}")
                
        except Exception as e:
            print(f"Error processing {args.input}: {e}")
            
    elif os.path.isdir(args.input):
        # Batch processing
        process_pdf_batch(args.input, args.output, save_text, save_images, 
                         args.languages, args.dpi)
    else:
        print(f"Error: {args.input} is not a valid PDF file or directory")

def main2():
    """Main function for testing purposes."""
    input_dir = "makeTemplatePDF/out/"  # Replace with your input directory
    # Ensure output directory exists
    if not os.path.exists("output_pdfs"):
        os.makedirs("output_pdfs")
    output_dir = "output_pdfs"  # Replace with your output directory
    process_pdf_batch(input_dir, output_dir, save_text=True, save_images=False, 
                     language_list=["en"], dpi=300)
    
main2()