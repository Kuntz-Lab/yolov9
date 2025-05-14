import json
import argparse
import os
from typing import List, Dict, Union

def extract_image_names(json_data: Union[List, Dict], key_name: str = "file_name") -> List[str]:
    """Extract image names from JSON data based on the specified key"""
    image_names = []
    
    if isinstance(json_data, list):
        for item in json_data:
            if isinstance(item, dict):
                if key_name in item:
                    image_names.append(item[key_name])
                else:
                    # Recursively search in nested dictionaries
                    image_names.extend(extract_image_names(item, key_name))
            elif isinstance(item, list):
                image_names.extend(extract_image_names(item, key_name))
    elif isinstance(json_data, dict):
        if key_name in json_data:
            image_names.append(json_data[key_name])
        else:
            # Search all dictionary values
            for value in json_data.values():
                if isinstance(value, (dict, list)):
                    image_names.extend(extract_image_names(value, key_name))
    
    return image_names

def main():
    # parser = argparse.ArgumentParser(description='Extract image names from JSON file to text file')
    # parser.add_argument('json_tag', default='train', help='Path to the JSON file')
    # parser.add_argument('json_dir', default='./med_data', help='Path to the JSON directory')
    # parser.add_argument('output_tag', default='train', help='Path to the output text file')
    # parser.add_argument('output_dir', default='./data/dataset', help='Path to the output text file')
    # parser.add_argument('--key', default='file_name', help='JSON key containing image names (default: file_name)')

    json_tag = 'val'
    json_dir = './med_data'
    output_tag = 'test'
    output_dir = './data/dataset'
    key = 'file_name'
    
    # args = parser.parse_args()

    json_filename = 'medtool_' + json_tag + '_anns.json'
    json_file = os.path.join(json_dir, json_filename)

    save_filename = output_tag + '_medvision.txt'
    save_file = os.path.join(output_dir, save_filename)
    
    if not os.path.exists(json_file):
        print(f"Error: JSON file '{json_file}' not found")
        return 1
        
    try:
        with open(json_file, 'r') as f:
            json_data = json.load(f)
        
        image_names = extract_image_names(json_data, key)
        
        if not image_names:
            print(f"Warning: No image names found with key '{key}'")

        os.makedirs(os.path.dirname(save_file), exist_ok=True)

        with open(save_file, 'w') as f:
            for name in image_names:
                image = os.path.join(output_dir, 'images', output_tag, name)
                f.write(f"{image}\n")
        
        print(f"Successfully extracted {len(image_names)} image names to {save_file}")
        
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON file '{json_file}'")
        return 1
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())