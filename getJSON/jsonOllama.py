import ollama 

from jsonLLM import(
    cleanFilename, 
    extract_json_from_response, 
    save_model_response, 
    process_text_files_with_models
)

def textToLLM(prompt, text, model):
    """
    Function to send text to a specific LLM model using Ollama and get response.
    Returns the response text or None if failed.
    """
    try:
        print(f"Trying model: {model}")
        
        # Send request to Ollama
        response = ollama.chat(model=model, messages=[
            {"role": "user", "content": f"{prompt}\n\nText to analyze:\n{text}"}
        ])
        
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

def main(): #if I ran this on an Ollama server would that 
    models = [
        #"mistral:7b",     # Fast and capable
        "phi3:mini",      # Very efficient
        "gemma:2b"        # Tiny and quick
    ]
    for model in models:
        ensure_model_exists(model)   
    print(f"Found {len(models)} models to process.")
    process_text_files_with_models(
            models=models, 
            output_dir="OllamaOut", 
            text_directory="output_pdfs/text/", 
            prompt_path="getJSON/prompt.txt", 
            llm_function=textToLLM
    )

main()