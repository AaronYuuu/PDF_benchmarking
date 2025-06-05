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
    Filter the template JSON to only include keys that are present in the first 'num' items.
    """

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
    with open("makeTemplatePDF/out/mock_data.json", "r") as f:
        temp = json.load(f)
    k = list(temp.keys())[0]
    template = temp[k]
    print("Template loaded successfully.")
    template = template_to_string(template)  
    print(template)
    print(type(template['variants']))

    json_files = [f for f in os.listdir("JSONout") if f.endswith('.json') and f.__contains__('reportfakeHospital2')]
    print(f"Found {len(json_files)} JSON files to compare.")
    
    for json_file in json_files:
        with open(os.path.join("JSONout", json_file), "r") as f:
            dtemp = json.load(f)
        if dtemp["status"] == "success":
            data = dtemp["data"]["report_id"]
            if compare(template, data):
                print(f"{json_file} is equal to the template.")
            else:
                print(f"{json_file} is not equal to the template.")
                modelname = json_file.split('__')[0]
                diff = findexact(template, data)
                numWrong = len(diff["values_changed"])
                ##pprint.pprint(diff)
                print(f"Number of differences found in {modelname}'s JSON: {numWrong} out of {len(template)} keys.")
        else:
            print(f"{json_file} has an error status: {dtemp['status']}. Skipping comparison.")
    
main()