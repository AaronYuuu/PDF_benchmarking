from deepdiff import DeepDiff
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

def findexact(json1, json2):
    result = DeepDiff(json1, json2, ignore_order=True, ignore_numeric_type_changes=True,
                      ignore_string_case=True)
    return result


#TODO make this work
def filter_template(template, reportName):
    """
    Return a copy of `template` with only those entries whose key
    appears in reportName.txt (nested dicts/lists pruned similarly).
    """
    reportName = reportName.lower()
    if reportName == "fakehospital2":
        return template

    # read the report text once
    report_file = f"{reportName}.txt"
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
                        obj[k] = "" #make the not listend keys an empty string
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

def count_string_values(obj):
    """
    Count the total number of string values in the JSON structure.
    """
    count = 0
    
    if isinstance(obj, dict):
        for value in obj.values():
            if isinstance(value, str):
                count += 1
                #print(value)
            elif isinstance(value, (dict, list)):
                count += count_string_values(value)
    
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, str):
                count += 1
                #print(item)
            elif isinstance(item, (dict, list)):
                count += count_string_values(item)
    
    return count

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
    """
    differences = compare_dict_keys_and_values(template, data)
    
    # Count total string values in template for comparison base
    total_values = count_string_values(template)
    
    if not differences:
        return True, 0, total_values, []
    else:
        return False, len(differences), total_values, differences
    

def main():
    import pandas as pd
    with open("../makeTemplatePDF/out/mock_data.json", "r") as f:
        temp = json.load(f)
    k = list(temp.keys())[0]
    template = temp[k]
    print("Template loaded successfully.")
    template = template_to_string(template)  

    if os.path.exists("Hospital.csv"):
        ovr = pd.read_csv("Hospital.csv") 
    else:
        ovr = pd.DataFrame(columns = ["LLM","False Positives","False Negatives","Incorrect Extractions","Correct Matches","Precision","Recall","F1score","Accuracy","Source","Hospital"])  # Load existing CSV if it exists to append results

    json_direcs = ["OllamaOut", "OpenAIOut", "OpenRouter", "OpenRouterVisionOut", "OllamaVisionOut", "OpenAiVisionOut"]
    hospitals = ["fakeHospital1", "fakeHospital2"] ##update according to latex templates generated
    sources = {"OllamaOut": "Ollama", "OpenAIOut": "OpenAI", "OpenRouter": "OpenRouter",
               "OpenRouterVisionOut": "OpenRouter", "OllamaVisionOut": "Ollama", "OpenAiVisionOut": "OpenAi"}
    #json_direcs = ["OpenAIOut"]
    for direc in json_direcs:
        source = sources[direc]
        direc = "outJSON/" + direc
        for hospital in hospitals: 
            json_files = [f for f in os.listdir(direc) if f.endswith('.json') and f.__contains__(hospital)]
            copy = template_to_string(filter_template(template, hospital))
            # Convert template to lowercase
            copy = dict_to_lowercase(copy)
            
            for json_file in json_files:
                with open(os.path.join(direc, json_file), "r") as f:
                    dtemp = json.load(f)
                #print(f"\n--- {json_file} ---")
                if ":" in dtemp["model"]:
                        t = dtemp["model"].split(":")
                        dtemp["model"] = t[0] + t[1]
                if "Vision" in direc:
                        dtemp["model"] = dtemp["model"] + "*ImageInput*"
                if "qwen/qwen2.5-vl-72b-instruct" in dtemp["model"] :
                        dtemp["model"] = "qwen2.5:72b"
                if "meta-llama/llama-4-scout" in dtemp["model"]:
                        dtemp["model"] = "llama-4-scout"
                if "google/gemini-2.0-flash-exp" in dtemp["model"]:
                        dtemp["model"] = "gemini-2.0"
                if "devstral-small" in dtemp["model"]:
                        dtemp["model"] = "devstral-small"
                dtemp["model"] = dtemp["model"].split("/")[-1] 

                if dtemp["status"] != "success":
                    #print(f"Error status: {dtemp['status']}. Skipping comparison.")
                    if hospital == "fakeHospital1":
                        temp = {
                            "LLM": dtemp["model"],
                            "False Positives": 88,
                            "False Negatives": 88,
                            "Incorrect Extractions": 88,
                            "Correct Matches": 0,
                            "Precision": 0,
                            "Recall": 0,
                            "F1score": 0,
                            "Accuracy": 0,
                            "Source": source, 
                            "Hospital": "hospital1"
                        }
                        ovr = pd.concat([ovr, pd.DataFrame([temp])], ignore_index=True)
                    elif hospital == "fakeHospital2":
                        temp = {
                            "LLM": dtemp["model"],
                            "False Positives": 88,
                            "False Negatives": 88,
                            "Incorrect Extractions": 88,
                            "Correct Matches": 0,
                            "Precision": 0,
                            "Recall": 0,
                            "F1score": 0,
                            "Accuracy": 0,
                            "Source": source,
                            "Hospital": "hospital2"
                        }
                        ovr = pd.concat([ovr, pd.DataFrame([temp])], ignore_index=True)
                    continue
                else:  
                    try:
                        data = dtemp["data"]["report_id"]
                    except KeyError:
                        try:
                            data = dtemp["data"]["report"]
                        except KeyError:
                            #print(f"Error: No valid report data found in {json_file}. Skipping comparison.")
                            continue
                    
                    # Convert data to lowercase before comparison
                    data = dict_to_lowercase(data)
                    
                    # Compare values with template using strict dictionary matching
                    is_equal, num_differences, total_values, differences = compare_values_with_template(copy, data)
                    
                    if is_equal:
                        print(f"✓ Perfect match - all {total_values} values match!")
                    else:
                        matching_values = total_values - num_differences
                        accuracy = (matching_values / total_values) * 100
                        #print(f"✗ {matching_values}/{total_values} values match")
                        print(f"Accuracy: {accuracy:.1f}%")
                        fn = 0 
                        fp = 0
                        ic = 0
                        for diff in differences:
                            if "FALSE POSITIVE" in diff or "Key missing in template" in diff:
                                fp += 1
                            elif "FALSE NEGATIVE" in diff or "Key missing in extracted values" in diff:
                                fn += 1
                            elif "Type mismatch" in diff or "Value mismatch" in diff:
                                ic += 1
                                
                        #print(f"False Positives: {fp}, False Negatives: {fn}, Incorrect Extraction: {ic}")
                        fp += ic
                        fn += ic
                        precision = (matching_values / (matching_values + fp)) * 100 if (matching_values + fp) > 0 else 0
                        recall = (matching_values / (matching_values + fn)) * 100 if (matching_values + fn) > 0 else 0
                        print(f"Precision: {precision:.1f}%, Recall: {recall:.1f}%")
                        f1score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                        print(f"F1 Score: {f1score:.1f}\n")
                    
                    if hospital == "fakeHospital1":
                        temp = {
                            "LLM": dtemp["model"],
                            "False Positives": fp,
                            "False Negatives": fn,
                            "Incorrect Extractions": ic,
                            "Correct Matches": matching_values,
                            "Precision": precision,
                            "Recall": recall,
                            "F1score": f1score,
                            "Accuracy": accuracy,
                            "Source": source, 
                            "Hospital": "hospital1"
                        }
                        ovr = pd.concat([ovr, pd.DataFrame([temp])], ignore_index=True)
                    elif hospital == "fakeHospital2":
                        temp = {
                            "LLM": dtemp["model"],
                            "False Positives": fp,
                            "False Negatives": fn,
                            "Incorrect Extractions": ic,
                            "Correct Matches": matching_values,
                            "Precision": precision,
                            "Recall": recall,
                            "F1score": f1score, 
                            "Accuracy": accuracy,
                            "Source": source,
                            "Hospital": "hospital2"
                        }
                        ovr = pd.concat([ovr, pd.DataFrame([temp])], ignore_index=True)
                    continue

    # Save as pandas dataframe and export to csv
    ovr.to_csv("Hospital.csv")
    print("Visualization complete.")

    
if __name__ == "__main__":
    main()