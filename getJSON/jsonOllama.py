import ollama 
import base64
import os

from jsonLLM import(
    extract_json_from_response, 
    save_model_response, 
    process_text_files_with_models,
    group_images_by_source,
    encode_image_group_to_base64,
    read_prompt_file
)

def textToLLM(prompt, text, model):
    try:
        print(f"Trying model: {model}")
        
        # Send request to Ollama
        response = ollama.chat(model=model, messages=[
            {"role": "user", "content": f"{prompt}\n\nText:\n{text}"}
        ], options={"temperature": 0.0})
        
        print(f"✓ Success with model: {model}")
        return response['message']['content']
        
    except Exception as e:
        error_str = str(e)
        print(f"✗ Error with model {model}: {error_str}")
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
        response = ollama.chat(
            model=model, 
            messages=[
                {
                    "role": "user", 
                    "content": f"{prompt}\n\nUse these pages:",
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

def process_grouped_images_with_models(models, output_dir, image_directory="../output_pdfs/images/", prompt_path="prompt.txt"):
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
    
    # Process each group of images
    for source_name, image_paths in grouped_images.items():
        print(f"\n{'='*60}")
        print(f"Processing source: {source_name} ({len(image_paths)} pages)")
        print(f"{'='*60}")
        
        # Try each vision model for this image group
        for model in models:
            print(f"\nProcessing {source_name} with model: {model}")
            
            response = imageGroupToLLM(prompt, image_paths, model)
            # Use source name as the "file" identifier for consistent naming
            save_model_response(model, response, f"report_{source_name}", full_output_dir)
    
    print(f"\n{'='*60}")
    print(f"All image groups and models completed. Check {full_output_dir} folder for results.")
    print(f"{'='*60}")

def main(): #if I ran this on an Ollama server would that 
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    models = [
        #"mistral:7b",     # Fast and capable
        #"phi3:mini",      # Very efficient
        #"gemma3:1b",
        #"gemma3:1b-it-qat",  #quantized 1b
        "gemma3n:e4b", #optimiazed for laptops
        "qwen2.5vl:3b", #vision model
        #"qwen3:4B", #good for laptops
        #"gemma3:12b",    # can also do images unblock when done this trial
        #"llama3.2:1b", 
        #"llama3.2:3b",
       #"llama3.2-vision",
        "granite3.2-vision:2b",#specialized for document tasks (vision model only)
    ]
    for model in models:
        ensure_model_exists(model)   
    print(f"Found {len(models)} models to process.")
    process_text_files_with_models(
            prompt_path="prompt.txt",
            models=models, 
            output_dir="OllamaOut",
            llm_function=textToLLM
    )

    process_text_files_with_models(
        prompt_path="NERprompt.txt",
        models=models, 
        output_dir="OllamaOutNP",
        llm_function=textToLLM
    )
    
    vision_models = [
        #"gemma3:12b",       # Can handle images
        "granite3.2-vision:2b",
        "qwen2.5vl:3b", #vision model
        #"llama3.2-vision", #specialized for document tasks (vision model only)
       #"gemma3:4b", 
    ]
    print(f"Found {len(vision_models)} vision models to process.")
    process_grouped_images_with_models(
        models=vision_models,
        output_dir="OllamaVisionOut"
    )

    process_grouped_images_with_models(
        models=vision_models,
        output_dir="OllamaVisionOutNP",
        image_directory="../output_pdfs/images/",
        prompt_path="NERprompt.txt"
    )
if __name__ == "__main__":
    main()