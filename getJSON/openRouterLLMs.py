from openai import OpenAI
import time
from jsonLLM import ( 
    extract_json_from_response, 
    save_model_response, 
    process_text_files_with_models
)
import os 

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

def main():
    # List of models to try
    MODELS = [
        "google/gemini-2.0-flash-exp:free",
        "qwen/qwen2.5-vl-72b-instruct:free",
        "meta-llama/llama-4-scout:free",
        "mistralai/devstral-small:free", 
        "rekaai/reka-flash-3:free"
    ]
    #ensure text files are cleaned
    
    # Use the shared processing function
    process_text_files_with_models(MODELS,output_dir="JSONout/", llm_function=textToLLM)

if __name__ == "__main__":
    main()