import json
import os
import re
import base64
from collections import defaultdict

def read_text_file(file_path):
    """
    Read the content of a text file.
    Returns the file content as a string or None if failed.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def read_prompt_file(prompt_path="getJSON/prompt.txt"):
    """
    Read the prompt file content.
    Returns the prompt as a string or None if failed.
    """
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()

def encode_image_to_base64(image_path):
    """
    Encode a single image to base64.
    Returns the base64 encoded string.
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def group_images_by_source(directory="output_pdfs/images/"):
    """
    Group images by their source document (hospital report).
    Returns a dictionary where keys are source names and values are lists of image paths.
    """
    try:
        image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        grouped_images = defaultdict(list)
        
        for image_file in image_files:
            # Extract source name from filename (e.g., "fakeHospital1" from "report_fakeHospital1__...")
            if image_file.startswith("report_"):
                parts = image_file.split("_")
                if len(parts) >= 2:
                    source_name = parts[1]  # e.g., "fakeHospital1"
                    grouped_images[source_name].append(os.path.join(directory, image_file))
        
        # Sort images within each group by page number
        for source_name in grouped_images:
            grouped_images[source_name].sort(key=lambda x: int(x.split("_page_")[1].split(".")[0]) if "_page_" in x else 0)
        
        return dict(grouped_images)
    except Exception as e:
        print(f"Error grouping images from directory {directory}: {e}")
        return {}

def encode_image_group_to_base64(image_paths):
    """
    Encode multiple images to base64.
    Returns a list of base64 encoded strings.
    """
    encoded_images = []
    for image_path in image_paths:
        encoded = encode_image_to_base64(image_path)
        if encoded:
            encoded_images.append(encoded)
    return encoded_images

def fix_json_structure(json_str):
    """
    Fix common JSON structure issues, particularly with the tested_genes field.
    """
    # Fix the tested_genes structure - convert from object with duplicate keys to array
    # Look for the pattern: "tested_genes": { "gene_id": {...}, "gene_id": {...}, ... }
    tested_genes_pattern = r'"tested_genes":\s*\{([^}]+(?:\{[^}]*\}[^}]*)*)\}'
    
    def fix_tested_genes(match):
        content = match.group(1)
        
        # Extract all gene_id objects
        gene_objects = []
        gene_pattern = r'"gene_id":\s*(\{[^}]*\})'
        
        for gene_match in re.finditer(gene_pattern, content):
            gene_objects.append(gene_match.group(1))
        
        if gene_objects:
            # Convert to array format
            genes_array = ',\n      '.join(gene_objects)
            return f'"tested_genes": [\n      {genes_array}\n    ]'
        else:
            return match.group(0)
    
    # Apply the fix
    fixed_json = re.sub(tested_genes_pattern, fix_tested_genes, json_str, flags=re.DOTALL)
    
    return fixed_json

def extract_json_from_response(response):
    """
    Extract JSON from responses that may have extra text or markdown formatting.
    Returns the parsed JSON object or None if no valid JSON is found.
    """
    if not response:
        return None
    
    # First try the existing cleaning method
    cleaned = response.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    
    # Try to fix JSON structure issues
    try:
        fixed_cleaned = fix_json_structure(cleaned)
        return json.loads(fixed_cleaned)
    except json.JSONDecodeError:
        pass
    
    comment_pattern = r'^\s*//.*$'
    json_nocomment = re.sub(comment_pattern, '', cleaned, flags=re.MULTILINE)
    try:
        return json.loads(json_nocomment)
    except json.JSONDecodeError:
        pass
    # If that fails, try to find JSON within the text
    # Look for content between ```json and ``` or just between { and }
        
    # Try to find JSON block with just ``` markers
    

    return None

def save_model_response(model, response, source_file, output_dir):
    """
    Save the response from a specific model to a JSON file.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean the source filename to get template name
    clean_name = source_file.split("_")[1]
    print(clean_name)
    
    # Create a clean filename from model name
    model_name = model.replace("/", "_").replace(":", "_")
    filename = f"{output_dir}/{clean_name}_{model_name}__response.json"
    
    try:
        if response is None:
            # Save error info if model failed
            output_data = {
                "model": model,
                "status": "failed",
                "error": "Model returned None response",
                "timestamp": "2025-06-05",
                "source_file": source_file
            }
            
            with open(filename, "w") as f:
                json.dump(output_data, f, indent=2)
            
            print(f"Error response saved to {filename}")
            return
        
        # Try to extract JSON using the improved extraction function
        parsed_json = extract_json_from_response(response)
        
        if parsed_json is not None:
            # Successfully extracted JSON
            output_data = {
                "model": model,
                "status": "success",
                "data": parsed_json,
                "timestamp": "2025-06-05",
                "source_file": source_file
            }
            
            with open(filename, "w") as f:
                json.dump(output_data, f, indent=2)
            
            print(f"✓ JSON response saved to {filename}")
            
        else:
            # Could not extract valid JSON
            print(f"Could not extract valid JSON from {model}")
            
            # Save raw response as fallback
            output_data = {
                "model": model,
                "status": "json_extraction_failed",
                "error": "Could not extract valid JSON from response",
                "raw_response": response,
                "timestamp": "2025-06-05",
                "source_file": source_file
            }
            
            fallback_filename = f"{output_dir}/{clean_name}_{model_name}_fallback.json"
            with open(fallback_filename, "w") as f:
                json.dump(output_data, f, indent=2)
            
            print(f"Fallback response saved to {fallback_filename}")
            
    except Exception as e:
        print(f"Error saving response for {model}: {e}")

def get_text_files_from_directory(directory="output_pdfs/text/"):
    """
    Get all text files from the specified directory.
    Returns a list of text file names.
    """
    try:
        return [f for f in os.listdir(directory) if f.endswith('.txt')]
    except Exception as e:
        print(f"Error reading directory {directory}: {e}")
        return []
    
def get_images_from_directory(directory="output_pdfs/images/"):
    """
    Get all image files from the specified directory.
    Returns a list of image file names.
    """
    try:
        return [f for f in os.listdir(directory) if f.lower().endswith(('.png'))]
    except Exception as e:
        print(f"Error reading directory {directory}: {e}")
        return []

def process_text_files_with_models(models, output_dir,text_directory="../output_pdfs/text/", prompt_path="prompt.txt", llm_function=None ):
    """
    Process all text files with the given models using the provided LLM function.
    
    Args:
        models: List of model names/identifiers
        text_directory: Directory containing text files to process
        prompt_path: Path to the prompt file
        llm_function: Function to call LLM (should accept prompt, text, model as parameters)
    """
    if llm_function is None:
        raise ValueError("llm_function must be provided")
    
    # Read the prompt file content
    prompt = read_prompt_file(prompt_path)
    os.makedirs("outJSON", exist_ok=True)

    output_dir = "outJSON/"+output_dir
    # Find all text files in the directory
    text_files = get_text_files_from_directory(text_directory)
    
    print(f"Found {len(text_files)} text files to process")
    print(f"Running extraction with {len(models)} models...")
    
    # Process each text file
    for text_file in text_files:
        print(f"\n{'='*60}")
        print(f"Processing file: {text_file}")
        print(f"{'='*60}")
        
        # Read the text file content
        text = read_text_file(os.path.join(text_directory, text_file))
        if text is None:
            print(f"Failed to read {text_file}, skipping...")
            continue
        
        # Try each model for this text file
        for model in models:
            print(f"\nProcessing {text_file} with model: {model}")
            
            response = llm_function(prompt, text, model)
            save_model_response(model, response, text_file, output_dir)
    
    print(f"\n{'='*60}")
    print(f"All files and models completed. Check {output_dir} folder for results.")
    print(f"{'='*60}")

def process_image_with_models(models, outputdir, image_path = "../output_pdfs/images", prompt_path = "prompt.txt", llm_function=None):
    """
    Process a single image with the given models using the provided LLM function.
    
    Args:
        models: List of model names/identifiers
        image_path: Path to the image file
        prompt_path: Path to the prompt file
        llm_function: Function to call LLM (should accept prompt, text, model as parameters)
    """
    if llm_function is None:
        raise ValueError("llm_function must be provided")
    
    # Read the prompt file content
    prompt = read_prompt_file(prompt_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(outputdir, exist_ok=True)
    
    print(f"Processing image: {image_path}")
    
    # Read the image file content
    try:
        with open(image_path, "rb") as f:
            image_content = f.read()
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return
    
    # Try each model for this image
    for model in models:
        print(f"\nProcessing image with model: {model}")
        
        response = llm_function(prompt, image_content, model)
        save_model_response(model, response, os.path.basename(image_path), outputdir)
    
    print(f"\nAll models completed. Check {outputdir} folder for results.")
