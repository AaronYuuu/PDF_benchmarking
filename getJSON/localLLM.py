#run this on venv environment
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
    if model_name == "numind/NuExtract-2.0-2B" or model_name == "numind/NuExtract-2.0-4B":
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
            if model == "numind/NuExtract-2.0-2B" or model == "numind/NuExtract-2.0-4B":
                response = process_with_nuextract_2_0_text(text_content, model)
                save_model_response(model, response, filename, full_output_dir)
            else:
                response = process_with_nuextract(text_content, model)
                save_model_response(model, response, filename, full_output_dir)
    
    print(f"\nCompleted! Results saved to {full_output_dir}")

def process_all_vision_info(messages, examples=None, image_directory="../output_pdfs/images"):
    """
    Process vision information for Numind models.
    Returns a list of image file paths.
    """
    import os
    from PIL import Image

    def fetch_images_from_directory(directory):
        """Fetch all image paths from the specified directory."""
        try:
            image_files = []
            for f in os.listdir(directory):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(directory, f))
            return sorted(image_files)  # Sort for consistent ordering
        except Exception as e:
            print(f"Error accessing image directory {directory}: {e}")
            return []

    def extract_images_from_messages(messages):
        """Extract image paths from message content."""
        images = []
        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, dict) and 'content' in message:
                    content = message['content']
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict):
                                if item.get('type') == 'image' and 'image' in item:
                                    # Handle file:// URLs
                                    image_path = item['image']
                                    if image_path.startswith('file://'):
                                        image_path = image_path[7:]  # Remove file:// prefix
                                    images.append(image_path)
                                elif item.get('type') == 'image_url' and 'image_url' in item:
                                    # Handle base64 encoded images - we'll skip these for now
                                    # and rely on directory images instead
                                    pass
        return images

    # Get images from messages
    message_images = extract_images_from_messages(messages)
    
    # Get images from directory
    directory_images = fetch_images_from_directory(image_directory)
    
    # Combine all images
    all_images = message_images + directory_images
    
    return all_images if all_images else None


def process_hospital_images_and_extract(messages, model, processor, generation_config, image_directory="../output_pdfs/images"):
    """
    Process images sorted by hospital number and extract information using the provided model and processor.

    Args:
        messages: List of message dictionaries for processing.
        model: The model used for inference.
        processor: The processor used for tokenization and image processing.
        generation_config: Configuration for the model's generation process.
        image_directory: Path to the directory containing images to process.

    Returns:
        Extracted information for each hospital.
    """
    import os

    def sort_images_by_hospital(directory):
        """Sort images into hospital1 and hospital2 categories."""
        hospital1_images = []
        hospital2_images = []

        for filename in os.listdir(directory):
            if filename.startswith("report_fakeHospital1"):
                hospital1_images.append(os.path.join(directory, filename))
            elif filename.startswith("report_fakeHospital2"):
                hospital2_images.append(os.path.join(directory, filename))

        return hospital1_images, hospital2_images

    # Sort images by hospital
    hospital1_images, hospital2_images = sort_images_by_hospital(image_directory)


    def extract_information(images, hospital_name):
        """Extract information for a specific hospital."""
        document = [{"type": "image", "image": f"file://{image}"} for image in images]
        messages_with_images = [{"role": "user", "content": document}]

        text = processor.tokenizer.apply_chat_template(
            messages_with_images,
            template=EXTRACTION_TEMPLATE,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs = process_all_vision_info(messages_with_images)
        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        # Inference: Generation of the output
        generated_ids = model.generate(
            **inputs,
            **generation_config
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return {hospital_name: output_text}

    # Extract information for both hospitals
    hospital1_data = extract_information(hospital1_images, "hospital1")
    hospital2_data = extract_information(hospital2_images, "hospital2")

    return {**hospital1_data, **hospital2_data}


def process_all_hospital_images(models, output_dir="localout/", image_directory="../output_pdfs/images"):
    """
    Process all hospital images with NuExtract models and save the extracted data.

    Args:
        models: List of model names to use for processing.
        output_dir: Directory to save the extracted data.
        image_directory: Path to the directory containing images to process.

    Returns:
        None
    """
    import os
    import base64

    # Setup output directory
    os.makedirs("outJSON", exist_ok=True)
    full_output_dir = f"outJSON/{output_dir}"
    os.makedirs(full_output_dir, exist_ok=True)

    # Sort images by hospital
    def sort_images_by_hospital(directory):
        hospital1_images = []
        hospital2_images = []

        for filename in os.listdir(directory):
            if filename.startswith("report_fakeHospital1"):
                hospital1_images.append(os.path.join(directory, filename))
            elif filename.startswith("report_fakeHospital2"):
                hospital2_images.append(os.path.join(directory, filename))

        return hospital1_images, hospital2_images

    def encode_image(image_path):
        """Encode the image file to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    hospital1_images, hospital2_images = sort_images_by_hospital(image_directory)

    # Template for extraction
    template = """{"store": "verbatim-string"}"""

    def extract_and_save(images, hospital_name, model):
        """Extract information for a specific hospital and save the data."""
        encoded_images = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image)}"}}
            for image in images
        ]
        messages_with_images = [{"role": "user", "content": encoded_images}]

        model, processor = load_model(model)

        text = processor.tokenizer.apply_chat_template(
            messages_with_images,
            template=template,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs = process_all_vision_info(messages_with_images)
        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        generation_config = {"do_sample": False, "num_beams": 1, "max_new_tokens": 2048}

        # Inference: Generation of the output
        generated_ids = model.generate(
            **inputs,
            **generation_config
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # Save the extracted data
        filename = f"{hospital_name}_{model.replace('/', '_')}.json"
        output_path = os.path.join(full_output_dir, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_text, f, indent=2)

        print(f"Saved extracted data for {hospital_name} using {model} to {output_path}")

    # Process each hospital with each model
    for model in models:
        print(f"Processing images with model: {model}")
        extract_and_save(hospital1_images, "hospital1", model)
        extract_and_save(hospital2_images, "hospital2", model)

    print(f"\nCompleted! Results saved to {full_output_dir}")

def process_all_hospital_images_with_template(models, output_dir="localoutVision/", image_directory="../output_pdfs/images", max_images_per_batch=2, max_image_size=(1024, 1024)):
    """
    Process all hospital images with NuExtract models using the extraction template and save the extracted data.
    Implements memory-efficient batch processing to avoid buffer overflow errors.

    Args:
        models: List of model names to use for processing.
        output_dir: Directory to save the extracted data.
        image_directory: Path to the directory containing images to process.
        max_images_per_batch: Maximum number of images to process at once to manage memory.
        max_image_size: Maximum size (width, height) to resize images to save memory.

    Returns:
        None
    """
    import os
    from PIL import Image
    import gc
    import torch

    # Setup output directory
    os.makedirs("outJSON", exist_ok=True)
    full_output_dir = f"outJSON/{output_dir}"
    os.makedirs(full_output_dir, exist_ok=True)

    # Sort images by hospital
    def sort_images_by_hospital(directory):
        hospital1_images = []
        hospital2_images = []

        for filename in os.listdir(directory):
            if filename.startswith("report_fakeHospital1"):
                hospital1_images.append(os.path.join(directory, filename))
            elif filename.startswith("report_fakeHospital2"):
                hospital2_images.append(os.path.join(directory, filename))

        return sorted(hospital1_images), sorted(hospital2_images)

    def resize_image_if_needed(image, max_size):
        """Resize image if it's larger than max_size to save memory."""
        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
        return image

    def process_images_in_batches(images, hospital_name, model_name):
        """Process images in smaller batches to manage memory usage."""
        if not images:
            print(f"No images found for {hospital_name}")
            return

        try:
            # Load model and processor
            model, processor = load_model(model_name)
            device = model.device
            
            print(f"  Processing {len(images)} images in batches of {max_images_per_batch}")

            # Use the extraction template
            template_str = json.dumps(EXTRACTION_TEMPLATE, indent=4)
            
            all_responses = []

            # Process images in batches
            for i in range(0, len(images), max_images_per_batch):
                batch_images = images[i:i + max_images_per_batch]
                batch_num = (i // max_images_per_batch) + 1
                total_batches = (len(images) + max_images_per_batch - 1) // max_images_per_batch
                
                print(f"    Processing batch {batch_num}/{total_batches} ({len(batch_images)} images)")
                
                # Load and resize images for this batch
                pil_images = []
                for image_path in batch_images:
                    try:
                        img = Image.open(image_path).convert('RGB')
                        img = resize_image_if_needed(img, max_image_size)
                        pil_images.append(img)
                    except Exception as e:
                        print(f"      Error loading image {os.path.basename(image_path)}: {e}")
                        continue

                if not pil_images:
                    print(f"      No valid images in batch {batch_num}")
                    continue

                # Create user message with template for this batch
                user_content = f"# Template:\n{template_str}\n# Instructions: Extract information from these medical report images (batch {batch_num}/{total_batches}) according to the template structure."
                messages = [{"role": "user", "content": user_content}]

                try:
                    # Apply chat template
                    text = processor.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )

                    # Process inputs
                    inputs = processor(
                        text=[text],
                        images=pil_images,
                        padding=True,
                        return_tensors="pt",
                    ).to(device)

                    generation_config = {"do_sample": False, "num_beams": 1, "max_new_tokens": 4000}

                    # Inference: Generation of the output
                    with torch.no_grad():
                        generated_ids = model.generate(
                            **inputs,
                            **generation_config
                        )

                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    output_text = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )

                    # Get the response for this batch
                    batch_response = output_text[0] if output_text else ""
                    all_responses.append({
                        "batch": batch_num,
                        "images": [os.path.basename(img) for img in batch_images],
                        "response": batch_response
                    })

                    print(f"      ✓ Batch {batch_num} processed successfully")

                except Exception as e:
                    error_msg = f"Error processing batch {batch_num}: {str(e)}"
                    print(f"      ✗ {error_msg}")
                    all_responses.append({
                        "batch": batch_num,
                        "images": [os.path.basename(img) for img in batch_images],
                        "error": error_msg
                    })

                # Clear memory after each batch
                if 'inputs' in locals():
                    del inputs
                if 'generated_ids' in locals():
                    del generated_ids
                for img in pil_images:
                    img.close()
                gc.collect()
                if device.startswith('cuda'):
                    torch.cuda.empty_cache()
                elif device == 'mps':
                    torch.mps.empty_cache()

            # Combine all batch responses
            final_response = {
                "hospital": hospital_name,
                "model": model_name,
                "total_images": len(images),
                "total_batches": total_batches,
                "batches": all_responses,
                "combined_data": {}
            }

            # Try to extract and combine meaningful data from all batches
            try:
                combined_extraction = {}
                for batch_data in all_responses:
                    if "response" in batch_data and batch_data["response"]:
                        try:
                            batch_json = json.loads(batch_data["response"])
                            # Merge batch data into combined extraction
                            for key, value in batch_json.items():
                                if key not in combined_extraction:
                                    combined_extraction[key] = value
                                elif isinstance(value, list) and isinstance(combined_extraction[key], list):
                                    combined_extraction[key].extend(value)
                        except json.JSONDecodeError:
                            pass
                
                if combined_extraction:
                    final_response["combined_data"] = combined_extraction

            except Exception as e:
                print(f"      Warning: Could not combine batch data: {e}")

            # Save the extracted data
            filename = f"{hospital_name}_{model_name.replace('/', '_')}.json"
            output_path = os.path.join(full_output_dir, filename)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(final_response, f, indent=2)

            print(f"✓ Saved extracted data for {hospital_name} using {model_name} to {output_path}")

        except Exception as e:
            error_msg = f"Error processing {hospital_name} with {model_name}: {str(e)}"
            print(error_msg)
            
            # Save error information
            error_response = {
                "hospital": hospital_name,
                "model": model_name,
                "error": error_msg,
                "total_images": len(images) if images else 0
            }
            filename = f"{hospital_name}_{model_name.replace('/', '_')}_ERROR.json"
            output_path = os.path.join(full_output_dir, filename)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(error_response, f, indent=2)

    hospital1_images, hospital2_images = sort_images_by_hospital(image_directory)

    # Process each hospital with each model
    for model_name in models:
        print(f"Processing images with model: {model_name}")
        if hospital1_images:
            print(f"  Processing hospital1 ({len(hospital1_images)} images)")
            process_images_in_batches(hospital1_images, "hospital1", model_name)
        if hospital2_images:
            print(f"  Processing hospital2 ({len(hospital2_images)} images)")
            process_images_in_batches(hospital2_images, "hospital2", model_name)

        # Clear model cache between models to free memory
        if model_name in model_cache:
            del model_cache[model_name]
        gc.collect()
        device, _ = get_device_config()
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
        elif device == 'mps':
            torch.mps.empty_cache()

    print(f"\nCompleted! Results saved to {full_output_dir}")

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    #models = ["numind/NuExtract-2.0-2B", "numind/NuExtract-1.5-tiny","numind/NuExtract-2.0-4B" ]
    #print(f"Starting NuExtract processing with models: {models}")
    #process_all_text_files(models)
    vision_models = ["numind/NuExtract-2.0-2B", "numind/NuExtract-2.0-4B"]
    print(f"Starting vision processing with models: {vision_models}")
    process_all_hospital_images_with_template(vision_models)

if __name__ == "__main__":
    main()
