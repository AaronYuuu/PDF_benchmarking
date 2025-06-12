from deepdiff import DeepDiff
import json
import os
import pprint
import copy as c

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
            template[a] = str(v)
        if isinstance(v, list):
            for variantsdic in v:
                for key, value in variantsdic.items():
                    if isinstance(value, int) or isinstance(value, float):
                        variantsdic[key] = str(value)
    print("Template values converted to strings where applicable.")
    return template

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
    fp = 0
    fn = 0
    if not (isinstance(dict1, dict) and isinstance(dict2, dict)):
        differences.append(f"Type mismatch at {path}: expected dictionaries")
        return differences
    
    # Get all keys from both dictionaries
    all_keys = set(dict1.keys()) | set(dict2.keys())
    
    for key in all_keys:
        current_path = f"{path}.{key}" if path else key
        
        # Check if key exists in both dictionaries
        if key not in dict1:
            differences.append(f"Key missing in dict1 at {current_path}")
            continue
        elif key not in dict2:
            differences.append(f"Key missing in dict2 at {current_path}")
            continue
        
        val1 = dict1[key]
        val2 = dict2[key]
        
        # Both values are strings - exact comparison
        if isinstance(val1, str) and isinstance(val2, str):
            temp = compare_string(val1, val2, current_path)
            differences.extend(temp[0])
            fp += temp[1]
            fn += temp[2]
        
        # Both values are dictionaries - recursive comparison
        elif isinstance(val1, dict) and isinstance(val2, dict):
            differences.extend(compare_dict_keys_and_values(val1, val2, current_path)[0])
        
        # Both values are lists - handle list comparison
        elif isinstance(val1, list) and isinstance(val2, list):
            differences.extend(compare_list_values(val1, val2, current_path)[0])
        
        # Type mismatch - count as error
        else:
            differences.append(f"Type mismatch at {current_path}: {type(val1).__name__} vs {type(val2).__name__}")
    
    return [differences, fp, fn]

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
    """
    if str1.lower() == str2.lower():
        return [[],0,0]
    fn = 0 
    fp = 0
    # Handle scientific notation
    if 'e-' in str1.lower() or 'e-' in str2.lower():
        try:
            num1 = float(str1)
            num2 = float(str2)
            if num1 == num2:
                return []
        except ValueError:
            pass  # If conversion fails, treat as mismatch
    if str1 == "" and str2 !="":
        fp += 1
    return [[f"Extra value at {path}: expected '{str1}' but got {str2}"], fp, fn]
    
def compare_list_values(list1, list2, path=""):
    """
    Compare two lists by comparing values at corresponding positions.
    """
    differences = []
    
    if len(list1) != len(list2):
        differences.append(f"List length mismatch at {path}: {len(list1)} vs {len(list2)}")
    fp = 0
    fn = 0
    # Compare corresponding elements
    min_len = min(len(list1), len(list2))
    for i in range(min_len):
        current_path = f"{path}[{i}]"
        val1 = list1[i]
        val2 = list2[i]
        
        if isinstance(val1, str) and isinstance(val2, str):
            temp = compare_string(val1, val2, current_path)
            differences.extend(temp[0])
            fp += temp[1]
            fn += temp[2]
        elif isinstance(val1, dict) and isinstance(val2, dict):
            differences.extend(compare_dict_keys_and_values(val1, val2, current_path)[0])
        elif isinstance(val1, list) and isinstance(val2, list):
            differences.extend(compare_list_values(val1, val2, current_path)[0])
        else:
            differences.append(f"Type mismatch at {current_path}: {type(val1).__name__} vs {type(val2).__name__}")
    
    return differences, fp, fn

def compare_values_with_template(template, data):
    """
    Compare data with template using strict dictionary key matching.
    Returns count of matching vs mismatching values with type mismatch detection.
    """
    differences = compare_dict_keys_and_values(template, data)[0]
    
    # Count total string values in template for comparison base
    total_values = count_string_values(template)
    
    if not differences:
        return True, 0, total_values, []
    else:
        return False, len(differences), total_values, differences

def main():
    with open("../makeTemplatePDF/out/mock_data.json", "r") as f:
        temp = json.load(f)
    k = list(temp.keys())[0]
    template = temp[k]
    print("Template loaded successfully.")
    template = template_to_string(template)  

    # Show template value count
    

    json_direcs = ["JSONout", "OllamaOut", "OpenAIOut", "OpenRouter"]
    hospitals = ["fakeHospital1", "fakeHospital2"] ##update according to latex templates generated
    #json_direcs = ["OpenAIOut"]
    for direc in json_direcs:
        direc = "outJSON/" + direc
        for hospital in hospitals: 
            json_files = [f for f in os.listdir(direc) if f.endswith('.json') and f.__contains__(hospital)]
            copy = template_to_string(filter_template(template, hospital)) 
            total_template_values = count_string_values(copy)
            print(f"Template has {total_template_values} string values to compare.")
            print(f"Found {len(json_files)} JSON files to compare.")
            
            for json_file in json_files:
                with open(os.path.join(direc, json_file), "r") as f:
                    dtemp = json.load(f)
                #print(f"\n--- {json_file} ---")
                
                
                if dtemp["status"] != "success":
                    #print(f"Error status: {dtemp['status']}. Skipping comparison.")
                    continue
                else: 
                    if ":" in dtemp["model"]:
                        dtemp["model"] = dtemp["model"].split(":")[0]
                    print(f"\nModel: {dtemp['model']}")
                    try:
                        data = dtemp["data"]["report_id"]
                    except KeyError:
                        try:
                            data = dtemp["data"]["report"]
                        except KeyError:
                            #print(f"Error: No valid report data found in {json_file}. Skipping comparison.")
                            continue
                    
                    # Compare values with template using strict dictionary matching
                    is_equal, num_differences, total_values, differences = compare_values_with_template(copy, data)
                    
                    if is_equal:
                        print(f"✓ Perfect match - all {total_values} values match!")
                    else:
                        matching_values = total_values - num_differences
                        accuracy = (matching_values / total_values) * 100
                        print(f"✗ {matching_values}/{total_values} values match")
                        print(f"Accuracy: {accuracy:.1f}% \n")
                        print(f"Differences found: [{differences}]")
                    
                    continue

if __name__ == "__main__":
    main()