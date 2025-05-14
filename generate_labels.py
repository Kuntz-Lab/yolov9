#!/usr/bin/env python3

import json
import os
import numpy as np
import cv2
from pycocotools import mask as mask_utils
from tqdm import tqdm

def rle_to_yolo_polygon(rle, img_height, img_width, simplify=True, epsilon=1.0):
    """
    Convert RLE mask to YOLO polygon format
    
    Args:
        rle: RLE encoded mask
        img_height: Image height
        img_width: Image width
        simplify: Whether to simplify contours
        epsilon: Approximation accuracy parameter for simplification
        
    Returns:
        List of normalized polygon coordinates [x1, y1, x2, y2, ...]
    """
    # Decode RLE to binary mask
    if isinstance(rle, list):
        rle = {'counts': rle, 'size': [img_height, img_width]}
    binary_mask = mask_utils.decode(rle)
    
    # Find contours
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), 
                                   cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    # Get the largest contour (main object)
    if not contours:
        return []  # No contours found
        
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Simplify the contour if requested
    if simplify:
        largest_contour = cv2.approxPolyDP(largest_contour, 
                                          epsilon, 
                                          closed=True)
    
    # Convert to format needed for YOLO
    flattened = largest_contour.flatten().tolist()
    
    # Normalize coordinates to 0-1 range
    normalized = []
    for i in range(0, len(flattened), 2):
        normalized.append(flattened[i] / img_width)      # x
        normalized.append(flattened[i+1] / img_height)   # y
    
    return normalized, flattened

def normalize_bb(bbox, img_h, img_w):
    """Get bounding box from bounding box"""
    x_min, y_min, width, height = bbox
    x_max = x_min + width
    y_max = y_min + height

    # normalized centers and sizes
    x_center = (x_min + x_max) / 2 / img_w
    y_center = (y_min + y_max) / 2 / img_h
    bb_width = width / img_w
    bb_height = height / img_h

    return [x_center, y_center, bb_width, bb_height]

def convert_json_to_yolo_labels(json_path, output_dir, dataset_type='train'):#, img_source_dir=None):
    """
    Convert annotations from JSON with RLE masks to YOLO segmentation format
    
    Args:
        json_path: Path to JSON annotation file
        output_dir: Directory to save YOLO labels
        dataset_type: train, val, or test
        img_source_dir: Directory containing source images (if copying)
    """
    # Create output directories
    images_dir = os.path.join(output_dir, dataset_type, 'images')
    labels_dir = os.path.join(output_dir, dataset_type, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Load JSON file
    print(f"Loading annotations from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Create image id to file name mapping
    image_id_to_info = {}
    for image in data['images']:
        image_id_to_info[image['id']] = {
            'file_name': image['file_name'],
            'width': image['width'],
            'height': image['height']
        }
    
    # Create category id to index mapping (YOLO uses indices starting from 0)
    if 'categories' in data:
        categories = data['categories']
        category_mapping = {cat['id']: idx for idx, cat in enumerate(categories)}
    else:
        # If no categories provided, create a default mapping
        print("Warning: No categories found in JSON. Using index as category ID.")
        # Find all unique category IDs in annotations
        unique_cats = set()
        for ann in data['annotations']:
            if 'category_id' in ann:
                unique_cats.add(ann['category_id'])
        category_mapping = {cat_id: idx for idx, cat_id in enumerate(sorted(unique_cats))}
        
    # Process annotations and create new annotations with polygon segmentation
    new_annotations = []
    image_annotations = {}
    
    print(f"Processing annotations for {dataset_type} set...")
    for ann in tqdm(data['annotations']):
        # Check if this is a valid annotation with required fields
        if not all(key in ann for key in ['id', 'image_id', 'category_id', 'segmentation']):
            # Check if this seems to be a truncated annotation (just ID)
            if len(ann) == 1 and 'id' in ann:
                print(f"Warning: Annotation {ann['id']} appears to be truncated or incomplete. Skipping.")
            else:
                print(f"Warning: Annotation missing required fields: {ann}")
            continue
        
        image_id = ann['image_id']
        if image_id not in image_id_to_info:
            print(f"Warning: Image ID {image_id} not found in images list. Skipping annotation {ann['id']}.")
            continue
            
        img_info = image_id_to_info[image_id]
        img_width, img_height = img_info['width'], img_info['height']
        
        # Initialize image annotations entry if not exists
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        
        category_id = ann['category_id']
        class_idx = category_mapping[category_id]
        
        # Process segmentation based on format
        # For RLE format (usually has 'counts' and 'size')
        if 'counts' in ann['segmentation']:
            rle = ann['segmentation']
            yolo_polygon, raw_polygon = rle_to_yolo_polygon(rle, img_height, img_width)
            
            # Skip if no contours found
            if not yolo_polygon:
                print(f"Warning: No contours found for annotation {ann['id']}. Skipping.")
                continue
                
            # For YOLO format label file
            image_annotations[image_id].append((class_idx, yolo_polygon))
            
            # For updated JSON output - replace RLE with polygon format
            new_ann = ann.copy()
            new_ann['segmentation'] = [raw_polygon]  # COCO polygon format: [[x1, y1, x2, y2, ...]]
            new_annotations.append(new_ann)
            
        # For polygon format (list or list of lists of coordinates)
        elif isinstance(ann['segmentation'], list):
            # Already in polygon format, just normalize for YOLO
            poly_list = ann['segmentation']
            
            # Flatten if needed (COCO can have multiple polygons per object)
            if isinstance(poly_list[0], list):
                # Take the largest polygon if there are multiple
                largest_poly = max(poly_list, key=len)
                poly_points = largest_poly
            else:
                poly_points = poly_list
            
            # Normalize polygon coordinates
            yolo_polygon = []
            for i in range(0, len(poly_points), 2):
                yolo_polygon.append(poly_points[i] / img_width)
                yolo_polygon.append(poly_points[i+1] / img_height)
            
            # Add to image annotations for YOLO label file
            image_annotations[image_id].append((class_idx, yolo_polygon))
            
            # Keep original annotation for JSON output
            new_annotations.append(ann)
        else:
            print(f"Warning: Unknown segmentation format in annotation {ann['id']}. Skipping.")
            continue
    
    # Write YOLO label files
    print("Writing YOLO label files...")
    for image_id, annotations in tqdm(image_annotations.items()):
        img_info = image_id_to_info[image_id]
        file_name = img_info['file_name']
        base_name = os.path.splitext(file_name)[0]
        
        # Write label file
        label_path = os.path.join(labels_dir, f"{base_name}.txt")
        with open(label_path, 'w') as f:
            for class_idx, polygon in annotations:
                polygon_str = ' '.join([f"{coord:.6f}" for coord in polygon])
                f.write(f"{class_idx} {polygon_str}\n")
        
        # # Copy image if source directory provided
        # if img_source_dir:
        #     src_path = os.path.join(img_source_dir, file_name)
        #     dst_path = os.path.join(images_dir, file_name)
        #     if os.path.exists(src_path):
        #         shutil.copy2(src_path, dst_path)
        #     else:
        #         print(f"Warning: Source image {src_path} not found.")
    
    # Save updated JSON with polygon segmentations
    base_name = os.path.basename(json_path)
    name_parts = os.path.splitext(base_name)
    new_json_path = os.path.join(output_dir, f"{name_parts[0]}_polygon{name_parts[1]}")
    
    # Update annotations in the data
    data['annotations'] = new_annotations
    
    print(f"Writing updated JSON with polygon segmentations to {new_json_path}...")
    with open(new_json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Finished processing {dataset_type} dataset.")
    print(f"YOLO labels saved to {labels_dir}")
    print(f"New JSON with polygon segmentations saved to {new_json_path}")
    
    return new_json_path

if __name__ == "__main__":
    # Example usage
    json_path = "./med_data/medtool_train_anns.json"
    output_dir = "./data/dataset"
    # img_source_dir = "./med_data/images"  # Directory containing source images

    # Check if the source JSON exists
    if not os.path.exists(json_path):
        print(f"Error: Source JSON {json_path} not found.")
        print("Please update the paths in the script.")
        exit(1)
    os.makedirs(output_dir, exist_ok=True)

    # Convert training annotations
    print("\n=== Processing Training Data ===")
    convert_json_to_yolo_labels(json_path, output_dir, 'train')
    
    # Convert validation annotations
    val_json_path = "./med_data/medtool_val_anns.json"
    if os.path.exists(val_json_path):
        print("\n=== Processing Validation Data ===")
        convert_json_to_yolo_labels(val_json_path, output_dir, 'val')#, img_source_dir)
    else:
        print(f"\nWarning: Validation JSON {val_json_path} not found. Skipping.")
    
    # Convert test annotations (if available)
    test_json_path = "./med_data/medtool_val_anns.json"  # Often reusing validation set
    if os.path.exists(test_json_path):
        print("\n=== Processing Test Data ===")
        convert_json_to_yolo_labels(test_json_path, output_dir, 'test') #, img_source_dir)
    else:
        print(f"\nWarning: Test JSON {test_json_path} not found. Skipping.")

    print("\nProcessing complete! YOLO format dataset created at:", output_dir)

