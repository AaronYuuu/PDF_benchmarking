from jsonLLM import save_model_response
import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForVision2Seq, AutoProcessor

# Global cache for loaded models
model_cache = {}

# Standard template for genetic report extraction
EXTRACTION_TEMPLATE = {
    "report_id": "string",
    "date_collected": "date-time",
    "date_received": "date-time", 
    "date_verified": "date-time",
    "report_type": ["Pathology", "Molecular Genetics"],
    "testing_context": ["Clinical", "Research"],
    "ordering_clinic": "string",
    "testing_laboratory": "string",
    "sequencing_scope": ["Gene panel", "Targeted variant testing", "Whole exome sequencing (WES)", "Whole genome sequencing (WGS)", "Whole transcriptome sequencing (WTS)"],
    "tested_genes": [{"gene_symbol": "string", "refseq_mrna": "string"}],
    "num_tested_genes": "integer",
    "sample_type": ["Amplified DNA", "ctDNA", "Other DNA enrichments", "Other RNA fractions", "polyA+ RNA", "Ribo-Zero RNA", "Total DNA", "Total RNA"],
    "analysis_type": ["Variant analysis", "Microarray", "Repeat expansion analysis", "Karyotyping", "Fusion analysis", "Methylation analysis"],
    "variants": [{
        "gene_symbol": "string", "variant_id": "string", "chromosome": "string",
        "hgvsg": "string", "hgvsc": "string", "hgvsp": "string",
        "transcript_id": "string", "exon": "string",
        "zygosity": ["Homozygous", "Heterozygous", "Hemizygous", "Compound heterozygous"],
        "interpretation": ["Variant of clinical significance", "Variant of uncertain clinical significance", "Variant of no clinical significance"],
        "mafac": "number", "mafan": "integer", "mafaf": "number"
    }],
    "num_variants": "integer",
    "reference_genome": ["GRCh37", "GRCh38", "NCBI build 34", "hg19", "hg38"]
}

def get_device_config():
    """Get optimal device configuration."""
    if torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    elif torch.cuda.is_available():
        return "cuda", torch.bfloat16
    else:
        return "cpu", torch.float32

def load_model(model_name):
    """Load and cache NuExtract models."""
    print(f"Loading model: {model_name}")
    device, torch_dtype = get_device_config()
    print(f"Using {device} device")
    if model_name in model_cache:
        return model_cache[model_name]
    if model_name == "numind/NuExtract-2.0-2B":
        model = AutoModelForVision2Seq.from_pretrained(
            model_name, # <-- FIX: Load the correct model
            torch_dtype=torch_dtype,
            trust_remote_code=True
        ).to(device).eval()

        # <-- FIX: Load the necessary processor for multimodal inputs
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        
        # The 'processor' handles both tokenization and image processing.
        # We return it instead of a separate tokenizer.
        result = (model, processor)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch_dtype, 
            trust_remote_code=True
        ).to(device).eval()
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        result = (model, tokenizer)  # <-- FIX: Ensure result is assigned correctly
    model_cache[model_name] = result
    print(f"✓ Successfully loaded {model_name}")
    return result

def process_with_nuextract(text, model_name):
    """Process text with NuExtract using official template format."""
    model, tokenizer = load_model(model_name)
    
    # Format prompt with official NuExtract structure
    template = json.dumps(EXTRACTION_TEMPLATE, indent=4)
    prompt = f"""<|input|>\n### Template:\n{template}\n### Text:\n{text}\n\n<|output|>"""
    
    try:
        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=10000).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=4000,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode and extract response
        output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = output.split("<|output|>")[1].strip() if "<|output|>" in output else output.strip()
        
        # Validate JSON
        try:
            parsed_json = json.loads(response)
            return json.dumps(parsed_json, indent=2)
        except json.JSONDecodeError:
            return json.dumps({"raw_response": response, "error": "Could not parse as JSON"}, indent=2)
            
    except Exception as e:
        return json.dumps({"error": f"Model processing failed: {str(e)}"}, indent=2)

def process_with_nuextract_2_0_text(text, model_name):
    """Process text with NuExtract 2.0 using the specific chat template."""
    model, processor = load_model(model_name)
    
    template_str = json.dumps(EXTRACTION_TEMPLATE, indent=4)
    user_content = f"# Template:\n{template_str}\n# Context:\n{text}"
    messages = [{"role": "user", "content": user_content}]
    
    prompt = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt"
    ).to(model.device)
    
    generation_config = {"do_sample": False, "num_beams": 1, "max_new_tokens": 4000}
    
    try:
        with torch.no_grad():
            generated_ids = model.generate(**inputs, **generation_config)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        response = output_text[0] if output_text else ""
        
        try:
            parsed_json = json.loads(response)
            return json.dumps(parsed_json, indent=2)
        except json.JSONDecodeError:
            return json.dumps({"raw_response": response, "error": "Could not parse as JSON"}, indent=2)
            
    except Exception as e:
        return json.dumps({"error": f"Model processing failed: {str(e)}"}, indent=2)


def process_all_text_files(models, output_dir="localout/", text_directory="../output_pdfs/text/"):
    """Process all text files with NuExtract models."""
    # Setup output directory
    os.makedirs("outJSON", exist_ok=True)
    full_output_dir = f"outJSON/{output_dir}"
    os.makedirs(full_output_dir, exist_ok=True)
    
    # Get text files
    text_files = [f for f in os.listdir(text_directory) if f.endswith('.txt')]
    if not text_files:
        print(f"No text files found in {text_directory}")
        return
    
    print(f"Found {len(text_files)} text files")
    print(f"Processing with {len(models)} models...")
    
    # Process each file with each model
    for text_file in text_files:
        text_path = os.path.join(text_directory, text_file)
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
        except Exception as e:
            print(f"Error reading {text_file}: {e}")
            continue
        
        print(f"\nProcessing {text_file}...")
        
        for model in models:
            print(f"  Using model: {model}")
            filename = text_file.replace('.txt', '')
            if model == "numind/NuExtract-2.0-2B":
                response = process_with_nuextract_2_0_text(text_content, model)
                save_model_response(model, response, filename, full_output_dir)
            else:
                response = process_with_nuextract(text_content, model)
                save_model_response(model, response, filename, full_output_dir)
    
    print(f"\nCompleted! Results saved to {full_output_dir}")

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    models = ["numind/NuExtract-2.0-2B", "numind/NuExtract-1.5-tiny"]
    print(f"Starting NuExtract processing with models: {models}")
    
    process_all_text_files(models)

if __name__ == "__main__":
    main()
