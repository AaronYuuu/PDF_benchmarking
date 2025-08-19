def compare_dict_keys_and_values(dict1, dict2, path=""):
    """
    Compare two dictionaries by checking key matches and recursively comparing values.
    Enhanced to better handle missing keys, null/None values, and comprehensive error categorization.
    """
    differences = []
    if not (isinstance(dict1, dict) and isinstance(dict2, dict)):
        differences.append(f"Type mismatch at {path}: expected dictionaries")
        return differences
    
    # Get all keys from both dictionaries
    template_keys = set(dict1.keys())
    extracted_keys = set(dict2.keys())
    all_keys = template_keys | extracted_keys
    
    # First, identify missing keys explicitly
    missing_in_extraction = template_keys - extracted_keys
    extra_in_extraction = extracted_keys - template_keys
    
    # Handle missing keys in extraction (these are always false negatives)
    for key in missing_in_extraction:
        current_path = f"{path}.{key}" if path else key
        template_value = dict1[key]
        differences.append(f"FALSE NEGATIVE at {current_path}: missing key '{key}' with template value '{template_value}'")
    
    # Handle extra keys in extraction (these are potential false positives)
    for key in extra_in_extraction:
        if key == "report_id":
            # Skip report_id as it's not a meaningful extraction key
            continue
        current_path = f"{path}.{key}" if path else key
        differences.append(f"FALSE POSITIVE at {current_path}: extra key with structure")
    
    # Now compare keys that exist in both dictionaries
    common_keys = template_keys & extracted_keys
    
    for key in common_keys:
        current_path = f"{path}.{key}" if path else key
        val1 = dict1[key]
        val2 = dict2[key]
        
        # Handle null/None values consistently
        val1 = val1 if val1 is not None else ""
        val2 = val2 if val2 is not None else ""
        
        # Both values are strings (or converted from null) - exact comparison
        differences = compare_vals(val1, val2, current_path, differences)
    
    return differences

def compare_string(str1, str2, path=""):
    """
    Compare two strings for equality with better handling of empty values and placeholders.
    Treats null/None values as equivalent to empty strings.
    """
    differences = []
    
    # Handle null/None values - treat as empty strings
    str1 = str1 if str1 is not None else ""
    str2 = str2 if str2 is not None else ""
    
    # Normalize strings for comparison
    norm_str1 = normalizeNames(str1.strip()) if str1 else ""
    norm_str2 = normalizeNames(str2.strip()) if str2 else ""
    
    # Perfect match (including both empty/null)
    if norm_str1 == norm_str2:
        differences.append(f"EXACT MATCH at {path}: both are '{str1}'")
        return differences
    
    # Handle scientific notation comparison
    try:
        if norm_str1 and norm_str2 and abs(float(norm_str1) - float(norm_str2)) < 1e-10:
            differences.append(f"EXACT MATCH at {path}: both are '{str1}' (scientific notation match)")
            return differences
    except ValueError:
        pass
    
    # Categorize mismatches more precisely
    if not norm_str1 and not norm_str2:
        # Both empty/null - this is a match
        differences.append(f"EXACT MATCH at {path}: both empty/null")
    elif not norm_str1 and norm_str2:
        # Template empty/null but extraction has value
       differences.append(f"FALSE POSITIVE at {path}: expected empty/null but got '{str2}'")    
    elif norm_str1 and not norm_str2:
        # Template has value but extraction empty/null
        differences.append(f"FALSE NEGATIVE at {path}: expected '{str1}' but got empty/null")
    elif norm_str1 in norm_str2:
        differences.append(f"{len(norm_str1)/len(norm_str2):.2f} PARTIAL MATCH at {path}: '{norm_str2}' found in '{norm_str1}'")
    else:
        differences.append(f"VALUE MISMATCH at {path}: expected '{str1}' but got '{str2}'")
    
    return differences
    
def compare_list_values(list1, list2, path=""):
    """
    Compare two lists by comparing values at corresponding positions.
    """
    differences = []

    
    # Compare corresponding elements
    min_len = min(len(list1), len(list2))
    for i in range(min_len):
        current_path = f"{path}[{i}]"
        val1 = list1[i]
        val2 = list2[i]
        differences = compare_vals(val1, val2, current_path, differences)
    
    return differences

def compare_vals(val1, val2, current_path, differences):
    if isinstance(val1, (str,int,float)) and isinstance(val2, (str, int, float)):
        val1 = str(val1).strip() if val1 is not None else ""
        val2 = str(val2).strip() if val2 is not None else ""
        differences.extend(compare_string(val1, val2, current_path))
    elif isinstance(val1, dict) and isinstance(val2, dict):
        differences.extend(compare_dict_keys_and_values(val1, val2, current_path))
    elif isinstance(val1, list) and isinstance(val2, list):
        differences.extend(compare_list_values(val1, val2, current_path))
    elif isinstance(val1, list) and isinstance(val2, dict):
        vals = []
        for k,v in val2.items():
            if isinstance(v,dict):
                vals.append(v)
            differences.extend(compare_list_values(val1, vals, current_path))
    elif isinstance(val1, dict) and isinstance(val2, list): 
        vals = []
        for k,v in val1.values():
            if isinstance(v,dict):
                vals.append(v)
            differences.extend(compare_list_values(vals, val2, current_path))
    elif isinstance(val1, (str, int, float)) and isinstance(val2, list):
        if len(val2) == 1 and isinstance(val2[0], (str, int, float)):
            # If list has one item, compare directly
            differences.extend(compare_string(str(val1), str(val2[0]), current_path))
        elif len(val2) > 1:
            if val1 in val2:
                temp = 1/len(val2)
                differences.append(f"{temp} PARTIAL MATCH at {current_path}: '{val1}' found in list")
            else:
                differences.append(f"VALUE MISMATCH at {current_path}: expected '{val1}' but got list containing {val2}")
    else:
        differences.append(f"Type mismatch at {current_path}: {type(val1).__name__} vs {type(val2).__name__}")
    return differences

def compare_values_with_template(template, data):
    """
    Compare data with template using strict dictionary key matching.
    Returns count of matching vs mismatching values with proper categorization.
    Only EXACT matches count as correct extractions - missing keys are always false negatives.
    """
    differences = compare_dict_keys_and_values(template, data)
    
    # Calculate actual template values
    total_template_values = count_all_template_values(template)
    
    # Properly categorize differences - ONLY exact matches count as correct
    fp = 0  # False positives: extra fields, placeholder values, or wrong values where template is empty
    fn = 0  # False negatives: missing fields or empty values where template has content
    ic = 0  # Incorrect extractions: wrong values where both template and extraction have content
    correct_matches = 0  # ONLY exact matches of non-empty values
    count = 0.0
    for diff in differences:
        diff_lower = diff.lower()
        
        # Only count as correct if it's an exact match of meaningful content
        if "exact match" in diff_lower:
            correct_matches += 1
            count += 1
        elif "false positive" in diff_lower:
            fp += 1
        elif "false negative" in diff_lower:
            fn += 1
            count += 1
        elif any(term in diff_lower for term in ["value mismatch", "type mismatch"]):
            ic += 1
            count += 1
        elif "partial match" in diff_lower:
            #print(diff_lower)
            #print(diff.lower().split(" "))
            temp = float(diff_lower.split(" ")[0])
            correct_matches += temp
            ic += 1 - temp
            count += temp
    # Verify counts make sense
    
    # Debug information
    #print(f"  Categorization: Correct={correct_matches}, FP={fp}, FN={fn}, IC={ic}")
    #print(f"  Total accounted: {count}/{total_template_values}")
    
    # Any unaccounted template values are missing fields (false negatives)
    if count < total_template_values:
        missing_fields = total_template_values - count
        fn += missing_fields
        #print(f"  Added {missing_fields} missing fields as false negatives")
    
    return correct_matches, fp, fn, ic, total_template_values, differences
