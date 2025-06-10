from jsonLLM import ( 
    extract_json_from_response, 
    save_model_response, 
    process_text_files_with_models
)

from openai import OpenAI
import os
import time

client = OpenAI(api_key="da")

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
    process_text_files_with_models(
        models=OPENAI_MODELS,
        output_dir="OpenAIOut/", 
        text_directory="output_pdfs/text/", 
        prompt_path="getJSON/prompt.txt", 
        llm_function=textToLLM
    )

if __name__ == "__main__":
    main()