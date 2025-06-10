from deepdiff import DeepDiff
import json
import os
import pprint

def compare(json1, json2):
    """
    Compare two JSON objects for equality.
    """
    return json1 == json2

def findexact(json1, json2):
    result = DeepDiff(json1, json2, ignore_order=True, ignore_numeric_type_changes=True,
                      ignore_string_case=True)
    return result

def compare_keys_only(obj1, obj2, path=""):
    """
    Compare two JSON objects focusing only on key-value pairs where values are strings.
    For lists, recursively check keys within list items without caring about list structure.
    """
    differences = []
    
    # Handle both objects being dictionaries
    if isinstance(obj1, dict) and isinstance(obj2, dict):
        # Check all keys from both objects
        all_keys = set(obj1.keys()) | set(obj2.keys())
        
        for key in all_keys:
            current_path = f"{path}.{key}" if path else key
            
            if key not in obj1:
                differences.append(f"Key missing in obj1: {current_path}")
            elif key not in obj2:
                differences.append(f"Key missing in obj2: {current_path}")
            else:
                # Recursively compare values
                differences.extend(compare_keys_only(obj1[key], obj2[key], current_path))
    
    # Handle lists - check keys within list items without caring about order/structure
    elif isinstance(obj1, list) and isinstance(obj2, list):
        # For lists, we'll check if all keys from obj1 items exist somewhere in obj2 items
        obj1_keys = set()
        obj2_keys = set()
        
        # Collect all keys from all items in both lists
        for item in obj1:
            if isinstance(item, dict):
                obj1_keys.update(get_all_keys_recursive(item))
        
        for item in obj2:
            if isinstance(item, dict):
                obj2_keys.update(get_all_keys_recursive(item))
        
        # Check for missing keys
        missing_in_obj2 = obj1_keys - obj2_keys
        missing_in_obj1 = obj2_keys - obj1_keys
        
        for key in missing_in_obj2:
            differences.append(f"Key missing in obj2 list items: {path}[*].{key}")
        for key in missing_in_obj1:
            differences.append(f"Key missing in obj1 list items: {path}[*].{key}")
        
        # For string values, compare them if they exist in both
        for item1 in obj1:
            if isinstance(item1, dict):
                # Find matching item in obj2 based on shared keys
                best_match = find_best_matching_item(item1, obj2)
                if best_match:
                    differences.extend(compare_keys_only(item1, best_match, f"{path}[item]"))
    
    # Handle string values - this is where we actually compare values
    elif isinstance(obj1, str) and isinstance(obj2, str):
        if obj1.lower() != obj2.lower():
            differences.append(f"Value difference at {path}: '{obj1}' vs '{obj2}'")
        if "e-" in obj1 or "e-" in obj2:
            from decimal import Decimal
            v1 = Decimal(obj1)
            v2 = Decimal(obj2)
            if v1 != v2:
                differences.append(f"Value difference at {path}: '{v1}' vs '{v2}'")
    
    return differences

def get_all_keys_recursive(obj, prefix=""):
    """
    Get all keys from a nested dictionary structure.
    """
    keys = set()
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            current_key = f"{prefix}.{key}" if prefix else key
            keys.add(current_key)
            
            if isinstance(value, (dict, list)):
                keys.update(get_all_keys_recursive(value, current_key))
    
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, (dict, list)):
                keys.update(get_all_keys_recursive(item, f"{prefix}[{i}]"))
    
    return keys

def find_best_matching_item(target_item, item_list):
    """
    Find the item in item_list that has the most keys in common with target_item.
    """
    if not isinstance(target_item, dict):
        return None
    
    best_match = None
    max_common_keys = 0
    
    target_keys = set(target_item.keys())
    
    for item in item_list:
        if isinstance(item, dict):
            item_keys = set(item.keys())
            common_keys = len(target_keys & item_keys)
            
            if common_keys > max_common_keys:
                max_common_keys = common_keys
                best_match = item
    
    return best_match

def compare_json_key_structure(json1, json2):
    """
    Main function to compare JSON structures focusing on keys and string values only.
    """
    differences = compare_keys_only(json1, json2)
    
    if not differences:
        return True, []
    else:
        return False, differences

def count_total_keys(obj, path=""):
    """
    Count the total number of keys in a JSON object, including nested keys.
    Returns the total count of all keys at all levels.
    """
    total_keys = 0
    
    if isinstance(obj, dict):
        # Count keys at this level
        total_keys += len(obj.keys())
        
        # Recursively count keys in nested objects
        for key, value in obj.items():
            current_path = f"{path}.{key}" if path else key
            if isinstance(value, (dict, list)):
                total_keys += count_total_keys(value, current_path)
    
    elif isinstance(obj, list):
        # For lists, count keys in all dictionary items
        for i, item in enumerate(obj):
            if isinstance(item, (dict, list)):
                total_keys += count_total_keys(item, f"{path}[{i}]")
    
    return total_keys

def count_string_value_keys(obj, path=""):
    """
    Count only the keys that have string values (leaf nodes).
    This gives you the number of actual key-value pairs that are compared.
    """
    string_keys = 0
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            current_path = f"{path}.{key}" if path else key
            
            if isinstance(value, str):
                string_keys += 1
            elif isinstance(value, (dict, list)):
                string_keys += count_string_value_keys(value, current_path)
    
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, (dict, list)):
                string_keys += count_string_value_keys(item, f"{path}[{i}]")
    
    return string_keys

def compare_with_template_keys_only(template, data):
    """
    Compare data with template focusing only on key structure and string values.
    Now includes key counting for better reporting.
    """
    is_equal, differences = compare_json_key_structure(template, data)
    
    # Count total keys and string value keys
    template_total_keys = count_total_keys(template)
    template_string_keys = count_string_value_keys(template)
    data_total_keys = count_total_keys(data)
    data_string_keys = count_string_value_keys(data)
    
    key_counts = {
        'template_total_keys': template_total_keys,
        'template_string_keys': template_string_keys,
        'data_total_keys': data_total_keys,
        'data_string_keys': data_string_keys
    }
    
    if is_equal:
        return True, 0, [], key_counts
    else:
        return False, len(differences), differences, key_counts

def finddiff(diff,  json1, json2):
    print("Differences found:")
    for key, value in diff.items():
        if value is None:
            print(f"Key: {key} is missing in one of the JSON objects.")
        else:
            if isinstance(value, dict):
                print(f"Key: {key} has different values in the two JSON objects.")
        print(f"Key: {key}, Value in JSON1: {json1[key]}, Value in JSON2: {json2[key]}")

def filter_template(template, reportNum):
    """
    Filter the template JSON to only include keys that are present in the corresponding reports. 
    """
    if reportNum == 1:
        with open("blurb_hospital1.txt", "r") as f:
            blurb = f.read()
    elif reportNum == 2:
        return
    elif reportNum == 3:
        with open("blurb_hospital3.txt", "r") as f:
            blurb = f.read()

def key_num(d):
    return sum(len(d) for d in d.values() if isinstance(d, dict))

def template_to_string(template):
    for a,v in template.items():
        if isinstance(v, int):
            template[a] = str(v)
        if isinstance(v, list):
            variantsdic = v[0]
            for key, value in variantsdic.items():
                if isinstance(value, int) or isinstance(value, float):
                    variantsdic[key] = str(value)
            template[a] = [variantsdic]
    print("Template keys converted to strings where applicable.")

    return template

    
def main():
    with open("../makeTemplatePDF/out/mock_data.json", "r") as f:
        temp = json.load(f)
    k = list(temp.keys())[0]
    template = temp[k]
    print("Template loaded successfully.")
    template = template_to_string(template)  
    
    # Show template key counts
    template_total = count_total_keys(template)
    template_strings = count_string_value_keys(template)
    print(f"Template has {template_total} total keys and {template_strings} string value keys")

    json_direcs = ["JSONout", "OllamaOut", "OpenAIOut"]
    #json_direcs = ["OpenAIOut"]
    for direc in json_direcs:
        direc = "outJSON/" + direc
        json_files = [f for f in os.listdir(direc) if f.endswith('.json') and f.__contains__('fakeHospital2')]
        print(f"Found {len(json_files)} JSON files to compare.")
        
        for json_file in json_files:
            with open(os.path.join(direc, json_file), "r") as f:
                dtemp = json.load(f)
            print(json_file)
            print(dtemp["model"])
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
                # Convert data keys to strings if necessary
                
                # Use the new key-focused comparison with key counting
                is_equal, num_differences, differences, key_counts = compare_with_template_keys_only(template, data)
                
                print(f"\n--- {json_file} ---")
                print(f"Template: {key_counts['template_total_keys']} total keys, {key_counts['template_string_keys']} string keys")
                print(f"Data:     {key_counts['data_total_keys']} total keys, {key_counts['data_string_keys']} string keys")
                
                if is_equal:
                    print(f"✓ Perfect match - all keys and values match!")
                else:
                    if key_counts["template_string_keys"] < key_counts["data_total_keys"]:
                        missed = key_counts["data_total_keys"] - key_counts["template_string_keys"]
                        num_differences += missed
                    print(f"✗ Found {num_differences} differences out of {key_counts['template_string_keys']} comparable string values")
                    accuracy = ((key_counts['data_string_keys'] - num_differences) / key_counts['template_total_keys']) * 100
                    print(f"Accuracy: {accuracy:.1f}%")
                    
                    # Show first few differences
                    #for i, difference in enumerate(differences[:3]):
                     #   print(f"  {i+1}. {difference}")
                    
                    #if len(differences) > 3:
                    #    print(f"  ... and {len(differences) - 3} more differences")
                        
            else:
                print(f"{json_file} has an error status: {dtemp['status']}. Skipping comparison.")
    
main()