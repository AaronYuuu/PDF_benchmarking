import ollama 
import base64
import os

from jsonLLM import(
    save_model_response, 
    process_text_files_with_models,
    group_images_by_source,
    read_prompt_file
)

#global ollama for simplicity
client = ollama.Client(host='localhost:11437')

def textToLLM(prompt, text, model):
    try:
        print(f"Trying model: {model}")
        
        # Send request to Ollama
        response = client.chat(model=model, messages=[
            {"role": "user", "content": f"{prompt}\n\nText to analyze:\n{text}"}
        ], options={"temperature": 0.0})
        
        print(f"✓ Success with model: {model}")
        return response['message']['content']
        
    except Exception as e:
        error_str = str(e)
        print(f"Error with model {model}: {error_str}")
        return None

def ensure_model_exists(model):
    try:
        ollama.show(model)
    except Exception:
        print(f"Pulling model: {model}")
        ollama.pull(model)

def imageGroupToLLM(prompt, image_group, model):
    """
    Function to send a group of images (multiple pages) to a vision-capable LLM model using Ollama.
    Returns the response text or None if failed.
    """
    try:
        print(f"Trying model: {model} with {len(image_group)} images")
        
        # For Ollama, we need to encode images and pass them directly
        images = []
        for image_path in image_group:
            with open(image_path, 'rb') as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
                images.append(image_data)
        
        # Send request to Ollama with images
        response = client.chat(
            model=model, 
            messages=[
                {
                    "role": "user", 
                    "content": f"{prompt}\n\nUse these images in place of the input text:",
                    "images": images
                }
            ]
        )
        
        print(f"Success with model: {model}")
        return response['message']['content']
        
    except Exception as e:
        error_str = str(e)
        print(f"✗ Error with model {model}: {error_str}")
        return None

def process_grouped_images_with_models(models, output_dir, image_directory="../output_pdfs/images/", prompt_path="/Users/ayu/PDF_benchmarking/getJSON/run_models/prompt.txt"):
    """    
    Args:
        models: List of vision-capable model names
        output_dir: Directory to save results with common naming structure manual input
        image_directory: Directory containing images
        prompt_path: Path to the prompt file
    """
    # Read the prompt file content
    prompt = read_prompt_file(prompt_path)
    
    # Create output directory
    os.makedirs("outJSON", exist_ok=True)
    full_output_dir = "outJSON/" + output_dir
    os.makedirs(full_output_dir, exist_ok=True)
    
    # Group images by source document
    grouped_images = group_images_by_source(image_directory)
    
    print(f"Found {len(grouped_images)} source documents with images")
    print(f"Running extraction with {len(models)} vision models...")
    
    # Get list of already processed sources from both vision output directories
    processed_sources = set()
    vision_out_dirs = [
        "/Users/ayu/PDF_benchmarking/getJSON/outJSON/" + output_dir
    ]
    
    for vision_dir in vision_out_dirs:
        if os.path.exists(vision_dir):
            try:
                vision_files = os.listdir(vision_dir)
                for filename in vision_files:
                    if filename.endswith('.json'):
                        # Extract source ID (UUID) from filename
                        source_id = filename.split('_')[0]
                        if "distressed" in filename.lower():
                            source_id += "_distressed"
                        processed_sources.add(source_id)
            except Exception as e:
                print(f"Warning: Could not read {vision_dir}: {e}")
    
    print(f"Found {len(processed_sources)} already processed sources, will skip them.")
    
    # Process each group of images
    for source_name, image_paths in grouped_images.items():
        print(f"\n{'='*60}")
        source = image_paths[0].split("_page")[0].split("/images/")[1]  # Extract source name from image paths
        print(f"Processing source: {source} ({len(image_paths)} pages)")
        s = source.replace("report_", "")  # Clean up source name if needed
        # Check if this source has already been processed
        if s in processed_sources:
            print(f"⏭️  Skipping {source} - already processed in vision output directories")
            continue
        
        print(f"{'='*60}")
        
        # Try each vision model for this image group
        for model in models:
            print(f"\nProcessing {source_name} with model: {model}")
            
            response = imageGroupToLLM(prompt, image_paths, model)
            # Use source name as the "file" identifier for consistent naming
            save_model_response(model, response, source, full_output_dir)
    
    print(f"\n{'='*60}")
    print(f"All image groups and models completed. Check {full_output_dir} folder for results.")
    print(f"{'='*60}")

def main(): #if I ran this on an Ollama server would that 
    os.chdir("/Users/ayu/PDF_benchmarking/getJSON")
    models = [
        #==============HPC models========================
        "mistral-small3.1:latest", #24b
        "llama3.1:70b", #70b
        "gemma3:27b", #27b
    ]
    for model in models:
        ensure_model_exists(model) 

    print(f"Found {len(models)} models to process.")
    '''process_text_files_with_models(
            prompt_path="/Users/ayu/PDF_benchmarking/getJSON/run_models/OSprompt.txt",
            models=models, 
            output_dir="OllamaOut",
            llm_function=textToLLM
    )'''
    
#stop when at 210 files
    process_text_files_with_models(
        prompt_path="/Users/ayu/PDF_benchmarking/getJSON/run_models/OSprompt.txt",
        models=models, 
        output_dir="OllamaOutOS",
        llm_function=textToLLM
    )
    
    vision_models = [
       #HPC
       "mistral-small3.1:latest", #24b
       "gemma3:27b", #27b
    ]
    print(f"Found {len(vision_models)} vision models to process.")
    process_grouped_images_with_models(
        models=vision_models,
        output_dir="OllamaVisionOut"
    )
    
    process_grouped_images_with_models(
        models=vision_models,
        output_dir="OllamaVisionOutOS",
        image_directory="../output_pdfs/images/",
        prompt_path="/Users/ayu/PDF_benchmarking/getJSON/run_models/OSprompt.txt"
    )

if __name__ == "__main__":
    main()
