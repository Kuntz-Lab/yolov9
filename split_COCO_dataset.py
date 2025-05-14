import json
import random
import os
import shutil
from collections import defaultdict

def split_coco_dataset_by_category(json_file_path, old_img_dir_path, train_dir, val_dir, train_ratio=0.8):
    # Load the original COCO dataset
    with open(json_file_path, 'r') as file:
        coco_data = json.load(file)
    
    # Extract image and annotation information
    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = coco_data['categories']
    
    # Create directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Create dictionaries to store the split datasets
    train_coco = {
        'images': [],
        'annotations': [],
        'categories': categories
    }
    val_coco = {
        'images': [],
        'annotations': [],
        'categories': categories
    }
    
    # Create a mapping from image ID to image info and file path
    image_id_to_info = {img['id']: img for img in images}
    image_id_to_path = {img['id']: old_img_dir_path + img['file_name'] for img in images}
    
    # Group annotations by category
    category_to_annotations = defaultdict(list)
    for ann in annotations:
        category_to_annotations[ann['category_id']].append(ann)
    
    # Split annotations for each category and add to the respective datasets
    for category_id, anns in category_to_annotations.items():
        random.shuffle(anns)
        split_idx = int(len(anns) * train_ratio)
        train_anns = anns[:split_idx]
        val_anns = anns[split_idx:]
        
        train_coco['annotations'].extend(train_anns)
        val_coco['annotations'].extend(val_anns)
    
    # Add images to the respective datasets based on the annotations
    train_image_ids = {ann['image_id'] for ann in train_coco['annotations']}
    val_image_ids = {ann['image_id'] for ann in val_coco['annotations']}
    
    for img_id in train_image_ids:
        train_coco['images'].append(image_id_to_info[img_id])
        shutil.copy(image_id_to_path[img_id], train_dir)
    
    for img_id in val_image_ids:
        val_coco['images'].append(image_id_to_info[img_id])
        shutil.copy(image_id_to_path[img_id], val_dir)
    
    # Save the new JSON files
    with open(os.path.join('medtool_train_anns.json'), 'w') as train_file:
        json.dump(train_coco, train_file)
    with open(os.path.join('medtool_val_anns.json'), 'w') as val_file:
        json.dump(val_coco, val_file)

if __name__=="__main__":
    """
    When we did our first round of annotations we had all of the images in one folder with no split between training and validation. This script seperates
    the images into training and validation by category of object in the image so that we get an equal split accross training and validataion and don't
    skew our training or validation towards one object category more than the others. The default ratio is 80% in training and 20% in validation.
    It would be best to do this before we create the dataset and simply make two datasets but this is an ok backup if you need to split an existing dataset
    based on category.
    """

    # Path to your original COCO dataset JSON file
    json_file_path = './med_data/medtool_anns.json'

    old_img_dir_path = './med_data/medvision_images/'

    # Directories to save the split datasets
    train_dir = 'med_data/train'
    val_dir = 'med_data/val'

    # Split the dataset by category and move images
    split_coco_dataset_by_category(json_file_path, old_img_dir_path, train_dir, val_dir)



