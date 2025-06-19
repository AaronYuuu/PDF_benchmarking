from jsonLLM import ( 
    extract_json_from_response, 
    save_model_response, 
    process_text_files_with_models,
    process_image_with_models
)
from transformers import AutoModelForVision2Seq, AutoModelForCausalLM, AutoProcessor, AutoTokenizer
import torch
import json
from PIL import Image
import os

# Global cache for loaded models
model_cache = {}

def load_model(model_name):
    """Load and cache NuExtract models with automatic type detection."""
    if model_name in model_cache:
        return model_cache[model_name]
    
    print(f"Loading model: {model_name}")
    
    # Try vision model first, fall back to text model
    try:
        model = AutoModelForVision2Seq.from_pretrained(
            model_name, 
            trust_remote_code=True, 
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model_type = "vision"
        
    except ValueError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        processor = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if processor.pad_token is None:
            processor.pad_token = processor.eos_token
        model_type = "text"
    
    result = (model, processor, model_type)
    model_cache[model_name] = result
    print(f"✓ Successfully loaded {model_name} on CPU ({model_type} model)")
    return result

def create_extraction_template():
    """Create the JSON template for genetic report extraction."""
    return {
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

def generate_response(model, processor, model_type, formatted_text, image=None):
    """Generate response from model with proper input handling."""
    if model_type == "vision":
        inputs = processor(
            text=[formatted_text],
            images=[image] if image else None,
            padding=True,
            return_tensors="pt"
        )
        tokenizer = processor.tokenizer
    else:
        inputs = processor(
            formatted_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        tokenizer = processor
    
    # Move to device and generate
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    input_length = inputs['input_ids'].shape[1]
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    return response

def NuToLLM(prompt, text, model):
    """Process text with NuExtract models."""
    model_obj, processor, model_type = load_model(model)
    template = create_extraction_template()
    
    # Format input based on model type
    if model_type == "vision":
        messages = [{"role": "user", "content": text}]
        formatted_text = processor.tokenizer.apply_chat_template(
            messages, template=json.dumps(template), tokenize=False, add_generation_prompt=True
        )
    else:
        # For text models, use a simpler format
        formatted_text = f"Extract the following information as JSON:\n\nTemplate: {json.dumps(template)}\n\nText: {text}\n\nJSON:"
    
    response = generate_response(model_obj, processor, model_type, formatted_text)
    
    # Handle empty or invalid responses
    if not response:
        return json.dumps({"error": "Empty response from model"}, indent=2)
    
    # Try to parse as JSON, return raw response if parsing fails
    try:
        parsed_json = json.loads(response)
        return json.dumps(parsed_json, indent=2)
    except json.JSONDecodeError:
        # Return a structured response with the raw output
        return json.dumps({"raw_response": response, "error": "Could not parse as JSON"}, indent=2)

def NuImageToLLM(prompt, image_path, model):
    """Process images with NuExtract vision models."""
    model_obj, processor, model_type = load_model(model)
    
    if model_type != "vision":
        return json.dumps({"error": f"Model {model} is not a vision model"}, indent=2)
    
    if not os.path.exists(image_path):
        return json.dumps({"error": f"Image file not found: {image_path}"}, indent=2)
    
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return json.dumps({"error": f"Error loading image: {str(e)}"}, indent=2)
    
    template = create_extraction_template()
    message_content = prompt or "Extract structured information from this medical report image."
    messages = [{"role": "user", "content": message_content}]
    
    formatted_text = processor.tokenizer.apply_chat_template(
        messages, template=json.dumps(template), tokenize=False, add_generation_prompt=True
    )
    
    response = generate_response(model_obj, processor, model_type, formatted_text, image)
    
    if not response:
        return json.dumps({"error": "Empty response from model"}, indent=2)
    
    try:
        parsed_json = json.loads(response)
        return json.dumps(parsed_json, indent=2)
    except json.JSONDecodeError:
        return json.dumps({"raw_response": response, "error": "Could not parse as JSON"}, indent=2)

def process_grouped_images_with_models(models, output_dir, image_directory="../output_pdfs/images/", prompt_path="prompt.txt"):
    """
    Process grouped images (by source document) with NuExtract vision models.
    """
    from jsonLLM import read_prompt_file, group_images_by_source, save_model_response
    
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
            
            # Process each image in the group individually for now
            # (NuExtract models work better with single images)
            for i, image_path in enumerate(image_paths):
                page_prompt = f"{prompt}\n\nThis is page {i+1} of {len(image_paths)} from document {source_name}."
                response = NuImageToLLM(page_prompt, image_path, model)
                
                # Save with page number in filename
                filename = f"report_{source_name}_page_{i+1}"
                save_model_response(model, response, filename, full_output_dir)
    
    print(f"\n{'='*60}")
    print(f"All image groups and models completed. Check {full_output_dir} folder for results.")
    print(f"{'='*60}")

def main():
    LOCAL_MODELS = ["numind/NuExtract-2.0-2B", "numind/NuExtract-1.5-tiny"]
    
    print("Starting NuExtract model processing (CPU-only mode)...")
    print(f"Available models: {LOCAL_MODELS}")
    
    # Process text files
    print("\n=== Processing Text Files ===")
    process_text_files_with_models(LOCAL_MODELS, output_dir="localout/", llm_function=NuToLLM)
    
    # Process images with vision models only
    vision_models = ["numind/NuExtract-2.0-2B"]
    print("\n=== Processing Image Files ===")
    process_grouped_images_with_models(vision_models, output_dir="localout/")
main()
