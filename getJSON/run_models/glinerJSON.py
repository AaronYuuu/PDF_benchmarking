#use PDF_benchmarking_py312 environment to run this code

from gliner import GLiNER # type: ignore
from jsonLLM import(
    read_text_file, 
    get_text_files_from_directory
)
import pprint
def merge_entities(entities, original_text, max_gap=10):
    """Merge entities with the same label that are close together."""
    if not entities:
        return []
    
    # Group by label
    groups = {}
    for entity in sorted(entities, key=lambda x: x['start']):
        label = entity['label']
        if label not in groups:
            groups[label] = []
        
        # Check if we can merge with the last entity in this label group
        if groups[label] and entity['start'] - groups[label][-1]['end'] <= max_gap:
            # Extend the last entity
            last = groups[label][-1]
            last['text'] = original_text[last['start']:entity['end']].strip()
            last['end'] = entity['end']
        else:
            groups[label].append(entity)
    
    # Flatten and sort by position
    return sorted([entity for group in groups.values() for entity in group], 
                  key=lambda x: x['start'])

def split_text(text):
    """
    Make text into sections of at most 834 characters
    """
    sections = []
    if len(text) <= 834:
        sections.append(text)
    else:
        words = text.split()
        current_section = ""
        for word in words:
            if len(current_section) + len(word) + 1 <= 834:
                if current_section:
                    current_section += " "
                current_section += word
            else:
                sections.append(current_section)
                current_section = word
        if current_section:
            sections.append(current_section)
    return sections

def main():
    import os
    os.chdir("/Users/ayu/PDF_benchmarking/getJSON")
    models = [
        "numind/NuNerZero", 
        #"Ihor/gliner-biomed-base-v1.0"
        ]
    
    labels = [
        "date_collected",
        "date_received", 
        "date_verified",
        "report_type",
        "testing_context",
        "ordering_clinic",
        "testing_laboratory",
        "sequencing_scope",
        "gene_symbol",
        "refseq_mrna",
        "num_tested_genes",
        "sample_type",
        "analysis_type",
        "chromosome",
        "hgvsg",
        "hgvsc",
        "hgvsp",
        "transcript_id",
        "exon",
        "zygosity",
        "interpretation",
        "mafac",
        "mafan",
        "mafaf",
        "num_variants",
        "reference_genome"
    ]
    
    # Use jsonLLM functions to read text files from output_pdfs/text directory
    text_directory = "../output_pdfs/text/"
    text_files = get_text_files_from_directory(text_directory)
    
    print(f"Found {len(text_files)} text files to process")
    
    # Create output directory
    os.makedirs("outJSON/glinerOut", exist_ok=True)
    
    # Process each model
    for model_name in models:
        print(f"\n{'='*50}")
        print(f"Processing with model: {model_name}")
        print(f"{'='*50}")
        
        # Load the current model
        model = GLiNER.from_pretrained(model_name)
        
        for text_file in text_files:
            print(f"\nProcessing file: {text_file} with {model_name}")
            
            # Read text content using jsonLLM function
            text_content = read_text_file(os.path.join(text_directory, text_file))
            if text_content is None:
                print(f"Failed to read {text_file}, skipping...")
                continue
            
            # Split text into manageable sections
            texts = split_text(text_content)
            results = {}
            
            for text_section in texts:
                entities = model.predict_entities(text_section, labels, threshold=0.25)
                entities = merge_entities(entities, text_section)
                
                for entity in entities:
                    if entity['label'] in results.keys():
                        results[entity['label']] += " " + entity['text']
                    else:
                        results[entity['label']] = entity['text']
            
            # Create output data with proper structure
            output_data = {
                "model": model_name,
                "status": "success",
                "data": results,
                "timestamp": "2025-06-19",
                "source_file": text_file
            }
            
            # Generate filename using same pattern as other models
            # Extract clean name from text file (e.g., "fakeHospital1" from "report_fakeHospital1__060f50fd...txt")
            clean_name = text_file.split("_")[1]  # Gets "fakeHospital1" or "fakeHospital2"
            append = "distressed" if "distressed" in text_file else ""
            # Convert model name to safe filename format
            safe_model_name = model_name.replace("/", "_").replace("-", "_")
            filename = f"outJSON/glinerOut/{clean_name}_{safe_model_name}__{append}__response.json"
            
            with open(filename, "w") as f:
                import json
                json.dump(output_data, f, indent=2)
            
            print(f"✓ GLiNER response saved to {filename}")
    
    print(f"\nAll files processed with all models. Check outJSON/glinerOut/ folder for results.")
main()