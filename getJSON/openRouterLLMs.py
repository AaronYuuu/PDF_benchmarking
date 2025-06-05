from openai import OpenAI
import json
import os
import time

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
)

def cleanFilename(filename):
    """
    Clean filename to extract just the template name part.
    Examples:
    - report_fakeHospital1__ab5c2ad8-906f-4bda-ad56-208429da34a5b02f80c240696a9adc7998ba14ec0301.txt
      becomes: reportfakeHospital1
    - report_fakeHospital2__some-uuid.txt 
      becomes: reportfakeHospital2
    """
    # Remove file extension
    base_name = os.path.splitext(filename)[0]
    
    # Split by underscore and take the first two parts (report_fakeHospitalX)
    parts = base_name.split('_')
    if len(parts) >= 2 and parts[0] == 'report':
        # Join report and fakeHospitalX, removing the underscore
        return parts[0] + parts[1]  # e.g., "report" + "fakeHospital1" = "reportfakeHospital1"
    
    # Fallback: return the original filename without extension
    return base_name

def textToLLM(prompt, text, model, max_retries=3):
    """
    Function to send text to a specific LLM model and get response.
    Returns the response text or None if failed.
    """
    for retry in range(max_retries):
        try:
            print(f"Trying model: {model} (attempt {retry + 1})")
            
            completion = client.chat.completions.create(
                extra_body={},
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"{prompt}\n\nText to analyze:\n{text}"
                            }
                        ]
                    }
                ]
            )
            print(f"✓ Success with model: {model}")
            return completion.choices[0].message.content
            
        except Exception as e:
            error_str = str(e)
            print(f"✗ Error with model {model}: {error_str}")
            
            # Check if it's a 429 rate limit error
            if "429" in error_str or "rate limit" in error_str.lower():
                print(f"Rate limit hit on {model}")
                if retry < max_retries - 1:
                    wait_time = 2 ** retry  # Exponential backoff: 1s, 2s, 4s
                    print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"Max retries reached for {model}")
                    break
            
            # For context length or other errors, don't retry
            elif "context_length_exceeded" in error_str or "too long" in error_str.lower():
                print(f"Context length exceeded on {model}")
                break
            
            else:
                # For other errors, retry with same model
                if retry < max_retries - 1:
                    print(f"Retrying with same model in 1 second...")
                    time.sleep(1)
                    continue
                else:
                    print(f"Max retries reached for {model}")
                    break
    
    print(f"Model {model} failed")
    return None

def fix_json_structure(json_str):
    """
    Fix common JSON structure issues, particularly with the tested_genes field.
    """
    import re
    
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
    
    # If that fails, try to find JSON within the text
    # Look for content between ```json and ``` or just between { and }
    import re
    
    # Try to find JSON block with ```json markers
    json_pattern = r'```json\s*(\{.*?\})\s*```'
    match = re.search(json_pattern, response, re.DOTALL | re.IGNORECASE)
    
    if match:
        try:
            json_content = match.group(1)
            fixed_json = fix_json_structure(json_content)
            return json.loads(fixed_json)
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON block with just ``` markers
    json_pattern = r'```\s*(\{.*?\})\s*```'
    match = re.search(json_pattern, response, re.DOTALL)
    
    if match:
        try:
            json_content = match.group(1)
            fixed_json = fix_json_structure(json_content)
            return json.loads(fixed_json)
        except json.JSONDecodeError:
            pass

    return None

def save_model_response(model, response, source_file="hospital2.txt"):
    """
    Save the response from a specific model to a JSON file.
    """
    # Clean the source filename to get template name
    clean_name = cleanFilename(source_file)
    
    # Create a clean filename from model name
    model_name = model.replace("/", "_").replace(":", "_")
    filename = f"JSONout/{model_name}_{clean_name}_response.json"
    
    try:
        if response is None:
            # Save error info if model failed
            output_data = {
                "model": model,
                "status": "failed",
                "error": "Model returned None response",
                "timestamp": "2025-06-04",
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
                "timestamp": "2025-06-04",
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
                "timestamp": "2025-06-04",
                "source_file": source_file
            }
            
            fallback_filename = f"JSONout/{model_name}_{clean_name}_fallback.json"
            with open(fallback_filename, "w") as f:
                json.dump(output_data, f, indent=2)
            
            print(f"Fallback response saved to {fallback_filename}")
            
    except Exception as e:
        print(f"Error saving response for {model}: {e}")
    
def main():
    # List of models to try
    MODELS = [
        "google/gemini-2.0-flash-exp:free",
        "qwen/qwen2.5-vl-72b-instruct:free",
        "meta-llama/llama-4-scout:free",
        "mistralai/devstral-small:free", 
        "rekaai/reka-flash-3:free"
    ]
    
    # Read the prompt file content
    with open("getJSON/prompt.txt", "r") as f:
        prompt = f.read()
    
    # Create JSONout directory if it doesn't exist
    os.makedirs("JSONout", exist_ok=True)
    
    # Find all text files in output_pdfs/text/ directory
    text_dir = "output_pdfs/text/"
    text_files = [f for f in os.listdir(text_dir) if f.endswith('.txt')]
    
    print(f"Found {len(text_files)} text files to process")
    print(f"Running extraction with {len(MODELS)} models...")
    
    # Process each text file
    for text_file in text_files:
        print(f"\n{'='*60}")
        print(f"Processing file: {text_file}")
        print(f"{'='*60}")
        
        # Read the text file content
        with open(os.path.join(text_dir, text_file), "r") as f:
            text = f.read()
        
        # Try each model for this text file
        for model in MODELS:
            print(f"\nProcessing {text_file} with model: {model}")
            
            response = textToLLM(prompt, text, model)
            save_model_response(model, response, text_file)
    
    print(f"\n{'='*60}")
    print("All files and models completed. Check JSONout/ folder for results.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()