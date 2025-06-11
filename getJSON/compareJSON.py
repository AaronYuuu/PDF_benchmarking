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
                        del obj[k]
                else:
                    # drop any leaf whose key is not in the report text
                    if k not in report_data:
                        del obj[k]

        elif isinstance(obj, list):
            for item in list(obj):
                if isinstance(item, (dict, list)):
                    recurse(item)
                    if not item:
                        obj.remove(item)
                # leave primitive list items untouched

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

def compare_string(str1, str2, path=""):
    """
    Compare two strings for equality, ignoring case and scientific notation.
    """
    if str1.lower() == str2.lower():
        return []
    
    # Handle scientific notation
    if 'e-' in str1.lower() or 'e-' in str2.lower():
        try:
            num1 = float(str1)
            num2 = float(str2)
            if num1 == num2:
                return []
        except ValueError:
            pass  # If conversion fails, treat as mismatch
    
    return [f"Value mismatch at {path}: '{str1}' vs '{str2}'"]

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
    with open("../makeTemplatePDF/out/mock_data.json", "r") as f:
        temp = json.load(f)
    k = list(temp.keys())[0]
    template = temp[k]
    print("Template loaded successfully.")
    template = template_to_string(template)  

    # Show template value count
    

    json_direcs = ["JSONout", "OllamaOut", "OpenAIOut"]
    hospitals = ["fakeHospital1", "fakeHospital2"] ##update according to latex templates generated
    #json_direcs = ["OpenAIOut"]
    for direc in json_direcs:
        direc = "outJSON/" + direc
        for hospital in hospitals: 
            json_files = [f for f in os.listdir(direc) if f.endswith('.json') and f.__contains__(hospital)]
            copy = filter_template(template, hospital)
            copy = template_to_string(copy)  
            # Ensure all values are strings for comparison
            #copy = c.deepcopy(template)
            total_template_values = count_string_values(copy)
            print(f"Template has {total_template_values} string values to compare.")
            print(f"Found {len(json_files)} JSON files to compare.")
            
            for json_file in json_files:
                with open(os.path.join(direc, json_file), "r") as f:
                    dtemp = json.load(f)
                print(f"\n--- {json_file} ---")
                print(f"Model: {dtemp['model']}")
                print(f"Status: {dtemp['status']}")
                
                if dtemp["status"] == "success":
                    try:
                        data = dtemp["data"]["report_id"]
                    except KeyError:
                        try:
                            data = dtemp["data"]["report"]
                        except KeyError:
                            print(f"Error: No valid report data found in {json_file}. Skipping comparison.")
                            continue
                    
                    # Compare values with template using strict dictionary matching
                    is_equal, num_differences, total_values, differences = compare_values_with_template(copy, data)
                    
                    if is_equal:
                        print(f"✓ Perfect match - all {total_values} values match!")
                    else:
                        matching_values = total_values - num_differences
                        accuracy = (matching_values / total_values) * 100
                        print(f"✗ {matching_values}/{total_values} values match")
                        print(f"Accuracy: {accuracy:.1f}%")
                        
                            
                else:
                    print(f"Error status: {dtemp['status']}. Skipping comparison.")

if __name__ == "__main__":
    main()