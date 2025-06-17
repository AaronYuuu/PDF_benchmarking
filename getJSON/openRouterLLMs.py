from openai import OpenAI
import time
import base64
from jsonLLM import ( 
    extract_json_from_response, 
    save_model_response, 
    process_text_files_with_models,
    group_images_by_source,
    read_prompt_file
)
import os 
from typing import List, Dict, Any

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.environ["OPENROUTER_API_KEY"]
)

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

def give_image_group_to_llm(prompt, image_group, model, max_retries=3):
    """
    Function to send a group of images (multiple pages) to a vision-capable LLM model.
    Returns the response text or None if failed.
    """
    import requests
    for retry in range(max_retries):
        try:
            print(f"Trying model: {model} with {len(image_group)} images (attempt {retry + 1})")
            
            # Prepare the message content with multiple images
            message_content: List[Dict[str, Any]] = [
                {"type": "text", "text": f"{prompt}\n\nPlease analyze all the pages of this medical report:"}
            ]
            
            # Add each image as base64 encoded data
            for i, image_path in enumerate(image_group):
                with open(image_path, 'rb') as image_file:
                    image_data = base64.b64encode(image_file.read()).decode('utf-8')
                    message_content.append({
                        "type": "image_url", 
                        "image_url": {
                            "url": f"data:image/png;base64,{image_data}"
                        }
                    })
            
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": message_content
                    }
                ]
            }
            completion = requests.post(json = payload, url = "https://openrouter.ai/api/v1/chat/completions", headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})
            completion = completion.json()
            print(f"✓ Success with model: {model}")
            return completion["choices"][0]["message"]["content"]
            
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

def process_grouped_images_with_openrouter_models(models, output_dir, image_directory="../output_pdfs/images/", prompt_path="prompt.txt"):
    """
    Process grouped images (by source document) with OpenRouter vision-capable models.
    
    Args:
        models: List of vision-capable model names
        output_dir: Directory to save results
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
        if source_name == "fakeHospital2":
            print(f"\n{'='*60}")
            print(f"Processing source: {source_name} ({len(image_paths)} pages)")
            print(f"Image files: {[os.path.basename(p) for p in image_paths]}")
            print(f"{'='*60}")
            
            # Try each vision model for this image group
            for model in models:
                print(f"\nProcessing {source_name} with model: {model}")
                
                response = give_image_group_to_llm(prompt, image_paths, model)
                # Use source name as the "file" identifier for consistent naming
                save_model_response(model, response, f"report_{source_name}", full_output_dir)
    
    print(f"\n{'='*60}")
    print(f"All image groups and models completed. Check {full_output_dir} folder for results.")
    print(f"{'='*60}")

def main_vision():
    """
    Main function to process images with vision-capable OpenRouter models.
    """
    # Vision-capable models from OpenRouter
    vision_models = [
        #"google/gemini-2.0-flash-exp:free",
        #"qwen/qwen2.5-vl-72b-instruct:free",
        "meta-llama/llama-4-scout:free",
    ]
    
    print(f"Found {len(vision_models)} vision models to process.")
    process_grouped_images_with_openrouter_models(
        models=vision_models,
        output_dir="OpenRouterVisionOut"
    )

def main():
    # List of models to try
    MODELS = [
        #"google/gemini-2.0-flash-exp:free",
        "qwen/qwen2.5-vl-72b-instruct:free",
       # "meta-llama/llama-4-scout:free",
        #"mistralai/devstral-small:free", 
    ]
    #ensure text files are cleaned
    
    # Use the shared processing function
    process_text_files_with_models(MODELS,output_dir="OpenRouter/", llm_function=textToLLM)

main_vision()
