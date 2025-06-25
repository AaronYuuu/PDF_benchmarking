import json
import os
import pprint
import copy as c
import pandas as pd  # Ensure pandas is imported with alias 'pd'

def compare(json1, json2):
    """
    Compare two JSON objects for equality.
    """
    return json1 == json2


def filter_template(template, reportName):
    """
    Return a copy of `template` with only those entries whose key
    appears in reportName.txt (nested dicts/lists pruned similarly).
    """
    reportName = reportName.lower()
    
    # Apply filtering to both hospitals based on their respective content files
    report_file = f"{reportName}.txt"
    
    # Check if the report file exists
    if not os.path.exists(report_file):
        print(f"Warning: {report_file} not found, using full template")
        return template

    # read the report text once
    with open(report_file, "r", encoding="utf-8") as f:
        report_data = f.read()

    filtered = c.deepcopy(template)

    def recurse(obj):
        if isinstance(obj, dict):
            for k in list(obj):
                v = obj[k]
                if isinstance(v, (dict, list)):
                    recurse(v)
                    if not v:
                        obj[k] = "" #make the not listed keys an empty string
                else:
                    # drop any leaf whose key is not in the report text
                    if k not in report_data:
                        obj[k] = ""

        elif isinstance(obj, list):
            for item in list(obj):
                if isinstance(item, (dict, list)):
                    recurse(item)
                    if not item:
                        index = obj.index(item)
                        obj[index] = ""  # make the not listed items an empty string

    recurse(filtered)
    return filtered

def key_num(d):
    return sum(len(d) for d in d.values() if isinstance(d, dict))

def template_to_string(template):
    for a,v in template.items():
        if isinstance(v, int):
            template[a] = str(v).lower()
        if isinstance(v, list):
            for variantsdic in v:
                for key, value in variantsdic.items():
                    if isinstance(value, int) or isinstance(value, float):
                        variantsdic[key] = str(value).lower()
    #print("Template values converted to strings where applicable.")
    return template

def dict_to_lowercase(obj):
    """
    Recursively convert all string values in a dictionary/list structure to lowercase.
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, str):
                obj[key] = value.lower()
            elif isinstance(value, (dict, list)):
                dict_to_lowercase(value)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, str):
                obj[i] = item.lower()
            elif isinstance(item, (dict, list)):
                dict_to_lowercase(item)
    return obj

def count_all_template_values(template):
    """
    Count ALL string and numeric values in the template, including empty ones and those in nested structures.
    This gives us the true total that should be used as the denominator for accuracy.
    """
    def count_recursive(obj):
        count = 0
        if isinstance(obj, dict):
            for value in obj.values():
                if isinstance(value, (str, int, float)):
                    count += 1  # Count all strings, numbers, even empty ones
                elif isinstance(value, (dict, list)):
                    count += count_recursive(value)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (str, int, float)):
                    count += 1
                elif isinstance(item, (dict, list)):
                    count += count_recursive(item)
        return count
    
    total = count_recursive(template)
    print(f"Total template values (including empty): {total}")
    return total

def compare_dict_keys_and_values(dict1, dict2, path=""):
    """
    Compare two dictionaries by checking key matches and recursively comparing values.
    - If both values are strings: exact match comparison
    - If both values are dictionaries: recursive comparison
    - If types don't match: count as error
    """
    differences = []
    if not (isinstance(dict1, dict) and isinstance(dict2, dict)):
        differences.append(f"Type mismatch at {path}: expected dictionaries")
        return differences
    
    # Get all keys from both dictionaries
    all_keys = set(dict1.keys()) | set(dict2.keys())
    
    for key in all_keys:
        current_path = f"{path}.{key}" if path else key
        
        # Check if key exists in both dictionaries
        if key not in dict1:
            differences.append(f"Key missing in template at {current_path}")
            continue
        elif key not in dict2:
            differences.append(f"Key missing in extracted values at {current_path}")
            continue
        
        val1 = dict1[key]
        val2 = dict2[key]
        
        # Both values are strings - exact comparison
        if isinstance(val1, str) and isinstance(val2, str):
            differences.extend(compare_string(val1, val2, current_path))
        
        # Both values are dictionaries - recursive comparison
        elif isinstance(val1, dict) and isinstance(val2, dict):
            differences.extend(compare_dict_keys_and_values(val1, val2, current_path))
        
        # Both values are lists - handle list comparison
        elif isinstance(val1, list) and isinstance(val2, list):
            differences.extend(compare_list_values(val1, val2, current_path))
        
        # Type mismatch - count as error
        else:
            differences.append(f"Type mismatch at {current_path}: {type(val1).__name__} vs {type(val2).__name__}")
    
    return differences

def normalizeNames(x):
    import re
    '''
    Normalize names by removing special characters and converting to lowercase. 
    bAsed off of ohcrn_lei eval compare_json.py same function
    '''
    # normalize hgvs by removing prefixes and brackets
    x = re.sub(r"Chr.+:g\.", "", x)
    x = re.sub(r"^g\.|^c\.|^p\.", "", x)
    x = re.sub(r"^\(|\)$", "", x)
    if re.match(r"^\d+-\d+$", x):
      x = re.sub(r"-\d+$", "", x)
    # normalize omim, clinvar, dbsnp
    x = re.sub(r"^OMIM\D+", "", x)
    x = re.sub(r"^Clinvar[^V]*", "", x, flags=re.IGNORECASE)
    x = re.sub(r"^dbSNP[^r]*", "", x, flags=re.IGNORECASE)
    # normalize chromosomes
    if re.match(r"^ChrX$|^ChrY$|^Chr\d$", x, flags=re.IGNORECASE):
      x = re.sub(r"Chr", "", x, flags=re.IGNORECASE)
    # remove location tags
    x = re.sub(
      r" ?\(Toronto$| ?\(Kingston$| ?\(Ottawa| ?\(London| ?\(Orillia.*| ?\(Mississauga",
      "",
      x,
      flags=re.IGNORECASE,
    )
    # convert everything to uppercase for case insensitive matching
    x = x.lower()
    # remove extra spaces
    x = re.sub(r"\s+", " ", x).strip()
        
    return x


def compare_string(str1, str2, path=""):
    """
    Compare two strings for equality, ignoring case and scientific notation.
    Returns differences list with FP/FN information.
    """
    differences = []
    
    # Normalize strings for comparison
    norm_str1 = normalizeNames(str1.strip()) if str1 else ""
    norm_str2 = normalizeNames(str2.strip()) if str2 else ""
    
    # Perfect match
    if norm_str1 == norm_str2:
        return differences
    
    # Handle scientific notation comparison
    if ('e-' in str1.lower() or 'e-' in str2.lower()):
        try:
            num1 = float(str1)
            num2 = float(str2)
            if abs(num1 - num2) < 1e-10:  # Handle floating point precision
                return differences
        except ValueError:
            pass  # If conversion fails, continue with string comparison
    
    # Add difference based on comparison with FP/FN labels
    if not str1 and str2:  # Template empty, but prediction has value
        differences.append(f"FALSE POSITIVE at {path}: template empty but got '{str2}'")
    elif str1 and not str2:  # Template has value, but prediction empty
        differences.append(f"FALSE NEGATIVE at {path}: expected '{str1}' but got empty")
    #elif str2 in str1 or str1 in str2:  # One string is a substring of the other
        #pass
    else:  # Both have values but don't match
        differences.append(f"Value mismatch at {path}: expected '{str1}' but got '{str2}'")
    
    return differences
    
def compare_list_values(list1, list2, path=""):
    """
    Compare two lists by comparing values at corresponding positions.
    """
    differences = []
    
    if len(list1) != len(list2):
        differences.append(f"List length mismatch at {path}: {len(list1)} vs {len(list2)}")
    
    # Compare corresponding elements
    min_len = min(len(list1), len(list2))
    for i in range(min_len):
        current_path = f"{path}[{i}]"
        val1 = list1[i]
        val2 = list2[i]
        
        if isinstance(val1, str) and isinstance(val2, str):
            differences.extend(compare_string(val1, val2, current_path))
        elif isinstance(val1, dict) and isinstance(val2, dict):
            differences.extend(compare_dict_keys_and_values(val1, val2, current_path))
        elif isinstance(val1, list) and isinstance(val2, list):
            differences.extend(compare_list_values(val1, val2, current_path))
        else:
            differences.append(f"Type mismatch at {current_path}: {type(val1).__name__} vs {type(val2).__name__}")
    
    return differences

def compare_values_with_template(template, data):
    """
    Compare data with template using strict dictionary key matching.
    Returns count of matching vs mismatching values with type mismatch detection.
    Uses actual template size instead of hard-coded values.
    """
    differences = compare_dict_keys_and_values(template, data)
    
    # Calculate actual template values instead of hard-coding
    total_values = count_all_template_values(template)
    
    if not differences:
        return True, 0, total_values, []
    else:
        return False, len(differences), total_values, differences

def compare_gliner_output(template, data, hospital, source, model_name):
    """
    Fixed GLiNER comparison function with consistent mathematics.
    """
    # Create flattened template for GLiNER comparison
    flat_template = flatten_template_for_gliner(template)
    
    # CRITICAL FIX: Use the flattened template size, not the original template
    total_values = len(flat_template)  # Count flattened fields, not original nested structure
    print(f"Flattened template values for GLiNER: {total_values}")
    
    # Compare values with flattened template
    is_equal, num_differences, _, differences, partial_matches, perfect_matches = compare_gliner_with_template(flat_template, data)
    
    if is_equal and partial_matches == 0 and perfect_matches == total_values:
        print(f"Perfect match - all {total_values} values match!")
        fp, fn, ic = 0, 0, 0
        correct_matches = total_values
        accuracy = 100.0
        precision = 100.0
        recall = 100.0
        f1score = 100.0
    else:
        # Count each type of difference with proper logic
        fp = 0  # False positives: GLiNER extracts fields not in template
        fn = 0  # False negatives: template fields missing from GLiNER
        ic = 0  # Incorrect extractions: wrong values
        
        for diff in differences:
            if "Key missing in template" in diff or "FALSE POSITIVE" in diff:
                fp += 1  # GLiNER extracted something not in template
            elif "Key missing in extracted" in diff or "FALSE NEGATIVE" in diff:
                fn += 1  # Template has field but GLiNER missed it
            elif "PARTIAL MATCH" in diff:
                # Partial matches are already counted in partial_matches
                continue
            elif "Value mismatch" in diff or "Type mismatch" in diff:
                ic += 1  # Wrong value extracted
        
        # Calculate correct matches: perfect + partial matches
        correct_matches = perfect_matches + partial_matches
        
        # MATHEMATICAL CONSISTENCY CHECK: 
        # correct_matches + fn + ic should equal total_values (template fields)
        # fp can be extra (GLiNER extracting more than template)
        expected_total = correct_matches + fn + ic
        if expected_total > total_values:
            print(f"WARNING: Math inconsistency - adjusting counts")
            print(f"Expected: {expected_total}, Actual: {total_values}")
            # Proportionally reduce to maintain ratios
            scale_factor = total_values / expected_total
            correct_matches = int(correct_matches * scale_factor)
            fn = int(fn * scale_factor)
            ic = int(ic * scale_factor)
        
        # Calculate metrics using standard formulas
        accuracy = (correct_matches / total_values * 100) if total_values > 0 else 0
        precision = (correct_matches / (correct_matches + fp) * 100) if (correct_matches + fp) > 0 else 0
        recall = (correct_matches / (correct_matches + fn) * 100) if (correct_matches + fn) > 0 else 0
        f1score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"FP: {fp}, FN: {fn}, IC: {ic}, Correct: {correct_matches}/{total_values}")
        print(f"Perfect: {perfect_matches}, Partial: {partial_matches}")
        print(f"Accuracy: {accuracy:.1f}%, Precision: {precision:.1f}%, Recall: {recall:.1f}%, F1: {f1score:.1f}%")
    
    return {
        "LLM": model_name,
        "False Positives": fp,
        "False Negatives": fn,
        "Incorrect Extractions": ic,
        "Correct Matches": correct_matches,
        "Precision": precision,
        "Recall": recall,
        "F1score": f1score,
        "Accuracy": accuracy,
        "Source": source,
        "Hospital": "hospital1" if hospital == "fakeHospital1" else "hospital2"
    }

def is_partial_match(template_val, gliner_val):
    """
    Check if there's a partial match between template and GLiNER values.
    Returns True if any single word from template is found in GLiNER output.
    """
    if not template_val or not gliner_val:
        return False
    
    # Split template into individual words and check if any exist in GLiNER output
    template_words = template_val.lower().split()
    gliner_text = gliner_val.lower()
    
    # If any template word is found in the GLiNER output, count as partial match
    for word in template_words:
        if len(word) > 2 and word in gliner_text:  # Only count words longer than 2 characters
            return True
    
    return False

def flatten_template_for_gliner(template):
    """
    Recursively flatten the nested template structure to match GLiNER's flat output format.
    Maps all nested fields to a flat dictionary with meaningful key names.
    """
    flattened = {}
    
    def flatten_recursive(obj, prefix=""):
        """Recursively flatten nested dictionaries and lists"""
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_key = f"{prefix}_{key}" if prefix else key
                
                if isinstance(value, (dict, list)):
                    flatten_recursive(value, new_key)
                else:
                    # Convert to string and store
                    flattened[new_key] = str(value) if value is not None else ""
        
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                new_prefix = f"{prefix}_{i}" if prefix else str(i)
                flatten_recursive(item, new_prefix)
    
    # Start the recursive flattening
    flatten_recursive(template)
    
    # Also create some simplified mappings that GLiNER might extract
    # Direct field mappings
    direct_fields = [
        "date_collected", "date_received", "date_verified", "report_type",
        "testing_context", "ordering_clinic", "testing_laboratory", 
        "sequencing_scope", "num_tested_genes", "sample_type", 
        "analysis_type", "num_variants", "reference_genome"
    ]
    
    for field in direct_fields:
        if field in template:
            flattened[field] = str(template[field]) if template[field] is not None else ""
    
    # Extract gene symbols and refseq_mrna from tested_genes
    if "tested_genes" in template and isinstance(template["tested_genes"], dict):
        for gene_name, gene_data in template["tested_genes"].items():
            if isinstance(gene_data, dict):
                if "gene_symbol" in gene_data:
                    flattened[f"gene_symbol_{gene_name}"] = str(gene_data["gene_symbol"])
                if "refseq_mrna" in gene_data:
                    flattened[f"refseq_mrna_{gene_name}"] = str(gene_data["refseq_mrna"])
    
    # Extract variant information
    if "variants" in template and isinstance(template["variants"], list):
        for i, variant in enumerate(template["variants"]):
            if isinstance(variant, dict):
                for key, value in variant.items():
                    flattened[f"variant_{i}_{key}"] = str(value) if value is not None else ""
    
    return flattened

def compare_gliner_with_template(template, gliner_data):
    """
    Simplified GLiNER comparison - single word matches count as correct.
    Uses actual template size instead of hard-coded values.
    """
    differences = []
    # Calculate actual template values instead of hard-coding
    total_template_values = len(template)
    partial_matches = 0
    perfect_matches = 0
    
    # Check for matches and mismatches
    all_keys = set(template.keys()) | set(gliner_data.keys())
    
    for key in all_keys:
        if key not in template:
            if key in gliner_data:
                differences.append(f"FALSE POSITIVE at {key}: template missing but got '{gliner_data[key]}'")
        elif key not in gliner_data:
            if template[key].strip():
                differences.append(f"FALSE NEGATIVE at {key}: expected '{template[key]}' but missing in extraction")
        else:
            # Both have the key, compare values
            template_val = template[key].strip() if template[key] else ""
            gliner_val = gliner_data[key].strip() if gliner_data[key] else ""
            
            if template_val and not gliner_val:
                differences.append(f"FALSE NEGATIVE at {key}: expected '{template_val}' but got empty")
            elif not template_val and gliner_val:
                differences.append(f"FALSE POSITIVE at {key}: template empty but got '{gliner_val}'")
            elif template_val and gliner_val:
                # Normalize and compare
                norm_template = normalizeNames(template_val)
                norm_gliner = normalizeNames(gliner_val)
                
                if norm_template == norm_gliner:
                    # Perfect match
                    perfect_matches += 1
                    continue
                elif is_partial_match(norm_template, norm_gliner):
                    # Single word match - count as correct
                    partial_matches += 1
                    differences.append(f"PARTIAL MATCH at {key}: expected '{template_val}' but got '{gliner_val}' (single word match)")
                else:
                    # No match
                    differences.append(f"Value mismatch at {key}: expected '{template_val}' but got '{gliner_val}'")
    
    num_differences = len(differences)
    is_equal = num_differences == 0
    
    return is_equal, num_differences, total_template_values, differences, partial_matches, perfect_matches

def main():
    import pandas as pd
    import os
    os.chdir("/Users/ayu/PDF_benchmarking/getJSON")
    with open("../makeTemplatePDF/out/mock_data.json", "r") as f:
        temp = json.load(f)
    k = list(temp.keys())[0]
    template = temp[k]
    print("Template loaded successfully.")
    template = template_to_string(template)  

    if os.path.exists("Hospital.csv"):
        ovr = pd.read_csv("Hospital.csv") 
    else:
        ovr = pd.DataFrame(columns = ["LLM","False Positives","False Negatives","Incorrect Extractions","Correct Matches","Precision","Recall","F1score","Accuracy","Source","Hospital"])

    # Expand to all available folders in outJSON with their corresponding sources
    json_direcs = [
        "localout",
        "glinerOut", 
        "OllamaOut",
        "OllamaVisionOut",
        "OpenAIOut", 
        "OpenAIVisionOut",
        "OpenRouter",
        "OpenRouterVisionOut"
    ]
    
    hospitals = ["fakeHospital1", "fakeHospital2"]
    
    # Map each directory to its corresponding source
    sources = {
        "localout": "huggingface",
        "glinerOut": "gliner",
        "OllamaOut": "ollama",
        "OllamaVisionOut": "ollama_vision", 
        "OpenAIOut": "openai",
        "OpenAIVisionOut": "openai_vision",
        "OpenRouter": "openrouter",
        "OpenRouterVisionOut": "openrouter_vision"
    }
    
    for direc in json_direcs:
        source = sources[direc]
        direc_path = "outJSON/" + direc
        
        # Check if directory exists and has files
        if not os.path.exists(direc_path):
            print(f"Directory {direc_path} does not exist, skipping...")
            continue
            
        all_files = [f for f in os.listdir(direc_path) if f.endswith('.json')]
        if not all_files:
            print(f"No JSON files found in {direc_path}, skipping...")
            continue
        
        print(f"\n=== Processing directory: {direc} (source: {source}) ===")
        
        for hospital in hospitals: 
            # For now, focus on numind models, but make it flexible for other models
            if direc == "glinerOut":
                # GLiNER uses different naming convention
                json_files = [f for f in all_files if hospital in f and "numind" in f]
            else:
                # Standard naming convention for other sources
                json_files = [f for f in all_files if hospital in f and "numind" in f]
            
            if not json_files:
                print(f"No matching JSON files found for {hospital} in {direc_path}")
                continue
                
            # Filter template for this hospital
            copy = template_to_string(filter_template(template, hospital))
            copy = dict_to_lowercase(copy)
            
            print(f"\n--- Processing {hospital} in {direc} ---")
            
            for json_file in json_files:
                file_path = os.path.join(direc_path, json_file)
                try:
                    with open(file_path, "r") as f:
                        dtemp = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    print(f"Error reading {json_file}: {e}")
                    continue
                    
                print(f"\n--- {json_file} ---")
                
                # Handle different file structures based on source
                if direc == "glinerOut":
                    # GLiNER has different structure - use the GLiNER comparison function
                    model_name = determine_model_name(direc, dtemp, json_file)
                    result = compare_gliner_output(copy, dtemp, hospital, source, model_name)
                    ovr = pd.concat([ovr, pd.DataFrame([result])], ignore_index=True)
                else:
                    # Use the determine_model_name function for consistent naming
                    model_name = determine_model_name(direc, dtemp, json_file)
                    
                    # Check for failed JSON parsing
                    if "error" in dtemp.get("data", {}) and dtemp["data"].get("error") == "Could not parse as JSON":
                        print(f"Model failed to parse JSON - treating as 0% accuracy")
                        temp_row = {
                            "LLM": model_name,
                            "False Positives": 0,
                            "False Negatives": count_all_template_values(copy),
                            "Incorrect Extractions": 0,
                            "Correct Matches": 0,
                            "Precision": 0,
                            "Recall": 0,
                            "F1score": 0,
                            "Accuracy": 0,
                            "Source": source,
                            "Hospital": "hospital1" if hospital == "fakeHospital1" else "hospital2"
                        }
                        ovr = pd.concat([ovr, pd.DataFrame([temp_row])], ignore_index=True)
                        continue
                    
                    # Use simplified comparison for valid extractions
                    data = dict_to_lowercase(dtemp["data"])
                    is_equal, num_differences, total_values, differences = compare_values_with_template(copy, data)
                    
                    # Properly count each type of difference
                    fp = 0  # False positives: extra fields in extraction
                    fn = 0  # False negatives: missing fields from extraction  
                    ic = 0  # Incorrect extractions: wrong values
                    correct_matches = 0  # Count actual correct matches
                    
                    for diff in differences:
                        if "Key missing in template" in diff:
                            fp += 1  # Extra field in extraction
                        elif "Key missing in extracted values" in diff:
                            fn += 1  # Missing field from extraction
                        elif "FALSE POSITIVE" in diff:
                            fp += 1  # Template empty but extraction has value
                        elif "FALSE NEGATIVE" in diff:
                            fn += 1  # Template has value but extraction empty
                        elif "Value mismatch" in diff or "Type mismatch" in diff:
                            ic += 1  # Wrong value (incorrect extraction)
                    
                    # Calculate correct matches properly - total minus all error types
                    correct_matches = max(0, total_values - fn - ic)
                    
                    # Ensure we don't exceed total template values
                    if fp + fn + ic + correct_matches > total_values:
                        # If we have overcounting, prioritize errors and adjust correct matches
                        correct_matches = max(0, total_values - fn - ic)
                    
                    # Calculate metrics using standard formulas
                    accuracy = (correct_matches / total_values * 100) if total_values > 0 else 0
                    precision = (correct_matches / (correct_matches + fp) * 100) if (correct_matches + fp) > 0 else 0
                    recall = (correct_matches / (correct_matches + fn) * 100) if (correct_matches + fn) > 0 else 0
                    f1score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    print(f"FP: {fp}, FN: {fn}, IC: {ic}, Correct: {correct_matches}/{total_values}")
                    print(f"Accuracy: {accuracy:.1f}%, Precision: {precision:.1f}%, Recall: {recall:.1f}%, F1: {f1score:.1f}%")
                    
                    temp_row = {
                        "LLM": model_name,
                        "False Positives": fp,
                        "False Negatives": fn,
                        "Incorrect Extractions": ic,
                        "Correct Matches": correct_matches,
                        "Precision": precision,
                        "Recall": recall,
                        "F1score": f1score,
                        "Accuracy": accuracy,
                        "Source": source,
                        "Hospital": "hospital1" if hospital == "fakeHospital1" else "hospital2"
                    }
                    ovr = pd.concat([ovr, pd.DataFrame([temp_row])], ignore_index=True)

    # Save results
    #ovr.to_csv("Hospital.csv")
    #print("Comparison complete. Results saved to Hospital.csv")


# Add a test function to verify order independenc

def determine_model_name(directory, json_data, filename=""):
    """
    Determine the appropriate model name based on directory, JSON content, and filename.
    Handles vision detection and different model types across various sources.
    
    Args:
        directory (str): The directory name (e.g., "OpenAIVisionOut", "localout")
        json_data (dict): The JSON data containing model information
        filename (str): The filename for additional context
    
    Returns:
        str: The formatted model name with vision indicator if applicable
    """
    model_name = json_data.get("model", "Unknown")
    
    # Model name normalization - handle colon separated names
    if ":" in model_name:
        t = model_name.split(":")
        model_name = t[0] + t[1]
    
    # Specific model mappings
    if "NuExtract-1.5-tiny" in model_name:
        model_name = "NuExtract:0.5B"
    elif "NuExtract-2.0-2B" in model_name:
        model_name = "NuExtract:2B"
    elif "qwen/qwen2.5-vl-72b-instruct" in model_name:
        model_name = "qwen2.5:72b"
    elif "meta-llama/llama-4-scout" in model_name:
        model_name = "llama-4:17B"
    elif "google/gemini-2.0-flash-exp" in model_name:
        model_name = "gemini-2.0"
    elif "devstral-small" in model_name:
        model_name = "mistral-3.1-24b"
    elif "mistral-small-3.1-24b-instruct" in model_name:
        model_name = "mistral-3.1-24b"
    elif "granite3.2-vision" in model_name:
        model_name = "granite3.2"
    elif "llama3.2_1b" in model_name:
        model_name = "llama3.2:1b"
    elif "llama3.2_3b" in model_name:
        model_name = "llama3.2:3b"
    
    # Add vision indicator if this is a vision-enabled directory
    if "Vision" in directory:
        model_name = model_name + "*ImageInput*"
    
    # Handle GLiNER models
    if "gliner" in directory.lower() or "numind" in model_name.lower():
        model_name = "GLiNER:NuNerZero"
    
    # Extract final model name (remove path prefixes)
    model_name = model_name.split("/")[-1]
    
    return model_name
 
main()