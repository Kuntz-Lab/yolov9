import json
from collections import defaultdict

def merge_annotation_files(source_file, target_file, output_file=None):
    """
    Merge annotations from source_file into target_file for images that don't exist in target_file.
    
    Args:
        source_file (str): Path to the source annotations JSON (annotations_singles_updated.json)
        target_file (str): Path to the target annotations JSON (medtool_anns.json)
        output_file (str, optional): Path to save the merged result. If None, overwrites target_file.
    
    Returns:
        dict: Statistics about the merge operation
    """
    # Load the JSON files
    with open(source_file, 'r') as f:
        source_data = json.load(f)
    
    with open(target_file, 'r') as f:
        target_data = json.load(f)
    
    # Extract image IDs from target to identify what's already there
    # Usually would use 'id' field but we'll create a fingerprint based on available data
    target_image_fingerprints = set()
    for img in target_data['images']:
        # Create a fingerprint for each image (adjust with additional fields if available)
        fingerprint = (img.get('height', None), img.get('width', None), 
                      img.get('file_name', None))
        target_image_fingerprints.add(fingerprint)
    
    # Track which source images to add
    new_images = []
    new_image_ids = set()
    
    # Find unique images in source that aren't in target
    for img in source_data['images']:
        fingerprint = (img.get('height', None), img.get('width', None),
                     img.get('file_name', None))
        
        if fingerprint not in target_image_fingerprints:
            # If source has an id field, track it for annotations
            if 'id' in img:
                new_image_ids.add(img['id'])
            new_images.append(img)
    
    # If we don't have image IDs, we need another way to associate annotations with images
    # In this case, we assume the order of annotations matches the order of images
    # This is a fallback and may not work for all datasets
    
    # Find annotations for new images
    new_annotations = []
    if new_image_ids:
        # If we have image IDs, use them to filter annotations
        for anno in source_data['annotations']:
            if 'image_id' in anno and anno['image_id'] in new_image_ids:
                new_annotations.append(anno)
    else:
        # Fallback: If we added N new images, add the first N annotations
        # This is an approximation and may not be accurate
        new_annotations = source_data['annotations'][:len(new_images)]
    
    # Add new data to target
    target_data['images'].extend(new_images)
    target_data['annotations'].extend(new_annotations)
    
    # Save the merged result
    if output_file is None:
        output_file = target_file
    
    with open(output_file, 'w') as f:
        json.dump(target_data, f, indent=2)
    
    return {
        "new_images_added": len(new_images),
        "new_annotations_added": len(new_annotations),
        "total_images": len(target_data['images']),
        "total_annotations": len(target_data['annotations'])
    }

# Example usage
result = merge_annotation_files(
    "/home/joe/git_repos/yolov9/data/dataset/medtool_train_anns_polygon.json",
    "/home/joe/git_repos/yolov9/data/dataset/medtool_val_anns_polygon.json",
    "/home/joe/git_repos/yolov9/data/dataset/medtool_anns_polygon.json"
)
print(f"Merged annotations: {result}")