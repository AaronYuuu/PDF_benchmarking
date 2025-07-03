import os
import json
from compareJSON import filter_template, template_to_string

def get_text_files_from_directory(directory):
    """
    Get all text file names from the specified directory.
    """
    return [f for f in os.listdir(directory) if f.endswith('.txt')]

def get_template(boo):
    """
    Returns the appropriate JSON template based on hospital.
    """
    with open("../makeTemplatePDF/out/mock_data.json", "r") as f:
            temp = json.load(f)
    k = list(temp.keys())[0]
    template = temp[k]
    print("Template loaded successfully.")
    template = template_to_string(template) 
    if boo:
        filtered = filter_template(template, "fakeHospital1")
    else:
        filtered = template
    return filtered

def main():
    input_dir = "../output_pdfs/text"

    output_file = "dataset.jsonl"
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    with open(output_file, "a") as out_f:
        for filename in get_text_files_from_directory(input_dir):
            file_path = os.path.join(input_dir, filename)
            boo = True if "Hospital1" in filename else False
            with open(file_path, "r") as tf:
                input_text = tf.read()

            # Construct the prompt
            prompt = input_text.strip() + "\n\n###\n\n"

                    # Get structured output (mocked here as a dictionary or list — replace as needed)
            structured_output = get_template(boo)

                    # Ensure it's a JSON-encoded string
            completion = " " + json.dumps(structured_output, ensure_ascii=False, indent=2) + "\n\nEND\n\n"

                    # Write the full JSONL entry
            entry = {"prompt": prompt, "completion": completion}
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
main()
