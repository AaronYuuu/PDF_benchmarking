from jsonLLM import(
    extract_json_from_response, 
    save_model_response, 
    process_text_files_with_models,
    group_images_by_source,
    encode_image_group_to_base64,
    read_prompt_file
)

from openai import OpenAI
import os
import time
API = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=API)

def textToLLM(prompt, text, model):
    """
    Function to send text to a specific OpenAI model and get response.
    Returns the response text or None if failed.
    """
    print(f"Trying model: {model}")
            
    response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": f"{prompt}\n\nText to analyze:\n{text}"
                    }
                ],
                temperature=0.0,
            )
            
    print(f"✓ Success with model: {model}")
    return response.choices[0].message.content
            
def image_group_toLLM(prompt, image_group, model):
    """
    Function to send a group of images (multiple pages) to a vision-capable OpenAI model.
    Returns the response text or None if failed.
    """
    print(f"Trying model: {model} with {len(image_group)} images")
    
    # For OpenAI, we need to encode images and pass them directly
    images = []
    for image_path in image_group:
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
            images.append(image_data)
    
    # Send request to OpenAI with images
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": f"{prompt}\n\nImages to analyze:\n{images}"
            }
        ],
        temperature=0.0,
        max_tokens=2048,
    )
    
    print(f"✓ Success with model: {model}")
    return response.choices[0].message.content

def main():
    # List of OpenAI models to try
    OPENAI_MODELS = [
        "gpt-4o-mini"
        # Add more OpenAI models as needed
    ]
    
    print("Starting OpenAI model processing...")
    print(f"Available models: {OPENAI_MODELS}")
    
    # Use the shared processing function
    '''
    process_text_files_with_models(
        models=OPENAI_MODELS,
        output_dir="OpenAIOut/", 
        text_directory="../output_pdfs/text/", 
        prompt_path="prompt.txt", 
        llm_function=textToLLM
    )
    '''
    image_group_toLLM(
        prompt=read_prompt_file("prompt.txt"),
        image_group=encode_image_group_to_base64(group_images_by_source("../output_pdfs/images/")),
        model="gpt-4o-mini"
    )

if __name__ == "__main__":
    main()