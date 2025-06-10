from jsonLLM import ( 
    extract_json_from_response, 
    save_model_response, 
    process_text_files_with_models
)
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Global variables to store loaded models and tokenizers
loaded_models = {}
loaded_tokenizers = {}

def load_model(model_name):
    """
    Load a GPT model and tokenizer from transformers library.
    Caches the model to avoid reloading.
    """
    if model_name in loaded_models:
        return loaded_models[model_name], loaded_tokenizers[model_name]
    
    try:
        print(f"Loading model: {model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with appropriate settings
        device_map = "auto" if torch.cuda.is_available() else None
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Cache the model and tokenizer
        loaded_models[model_name] = model
        loaded_tokenizers[model_name] = tokenizer
        
        print(f"✓ Successfully loaded {model_name}")
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None, None

def textToLLM(prompt, text, model):
    """
    Function to send text to a local GPT model using transformers and get response.
    Returns the response text or None if failed.
    """
    try:
        # Load the model and tokenizer
        gpt_model, tokenizer = load_model(model)
        print("Model loaded successfully")
        if gpt_model is None or tokenizer is None:
            print(f"Failed to load model {model}")
            return None
        
        # Combine prompt and text
        full_prompt = f"{prompt}\n\nText to analyze:\n{text}\n\nResponse:"
        print("Prompt has been prepared.")
        # Tokenize the input with proper truncation
        inputs = tokenizer(
            full_prompt, 
            return_tensors="pt", 
            truncation=True,  # Reduced to leave room for output
            padding=True
        )
        
        # Move to same device as model
        device = next(gpt_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = gpt_model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode the response (excluding the input tokens)
        input_length = inputs['input_ids'].shape[1]
        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        
        print(f"✓ Generated response with {model}")
        return response
        
    except Exception as e:
        print(f"Error with local LLM model {model}: {e}")
        return None

def main():
    # List of GPT models to try (starting with smaller, more reliable models)
    LOCAL_MODELS = [
        "microsoft/BioGPT-Large"              
        # Add more models as needed:
        # "microsoft/phi-2",            # Microsoft Phi-2 (good performance)
        # "stabilityai/stablelm-base-alpha-3b",  # StableLM
    ]
    
    print("Starting local GPT model processing...")
    print(f"Available models: {LOCAL_MODELS}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Use the shared processing function
    process_text_files_with_models(LOCAL_MODELS,output_dir="localout/", llm_function=textToLLM)
    

if __name__ == "__main__":
    main()
