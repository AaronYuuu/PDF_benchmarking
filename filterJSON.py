templateJSON = "../lei_mockup_generator/mock_data.json"
templateTex = "../lei_mockup_generator/templates/fakeHospital2.tex"

def labelsJSON(templateJSON):
    import json
    import os
    # Check if the file exists before reading
    if not os.path.exists(templateJSON):
        print(f"Error: JSON file not found at {templateJSON}")
        return []

    try:
        with open(templateJSON, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error reading JSON file: {e}")
        return []
    layer1  = next(iter(data.values()), {})
    print(layer1)


def filterTemplate(templateJSON, templateTEX):
    import json
    import re
    import os
    
    # Check if files exist before reading
    if not os.path.exists(templateJSON):
        print(f"Error: JSON file not found at {templateJSON}")
        return []
    
    if not os.path.exists(templateTEX):
        print(f"Error: TEX file not found at {templateTEX}")
        return []

    # Read JSON file
    try:
        with open(templateJSON, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error reading JSON file: {e}")
        return []

    # Read TEX file
    try:
        with open(templateTEX, 'r', encoding='utf-8') as f:
            template = f.read()
    except UnicodeDecodeError:
        # Try with different encoding if UTF-8 fails
        with open(templateTEX, 'r', encoding='latin-1') as f:
            template = f.read()

    # Extract second-layer keys from JSON
    all_keys = set()
    
    # If data is a dictionary, extract keys from nested dictionaries
    if isinstance(data, dict):
        for key, value in data.items():
            # Add first-layer keys
            all_keys.add(key)
            
            # Add second-layer keys if value is a dictionary
            if isinstance(value, dict):
                for nested_key in value.keys():
                    all_keys.add(nested_key)
            
            # If value is a list, check if list items are dictionaries
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        for nested_key in item.keys():
                            all_keys.add(nested_key)

    print(f"Found {len(all_keys)} total keys in JSON")
    print("Keys found:", sorted(all_keys))

    # Updated regex patterns to match different LaTeX command formats
    patterns = [r'\\data\{([a-zA-Z0-9_]+)\}']
    # Find all commands in the template using multiple patterns
    matches = re.findall(patterns[0], template)
    
    print(f"Found {len(matches)} commands in template")
    print("Commands found:", matches)

    return matches, all_keys

# Execute the function
labelsJSON(templateJSON)
matches, allkeys = filterTemplate(templateJSON, templateTex)
print(allkeys)