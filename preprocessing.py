"""
Preprocessing Pipeline

Usage:
    python preprocessing.py path\to\coco.json path\to\images desired\output\path
"""

import os
import shutil
import argparse
from utils import COCO_to_YOLO, resize_images, train_test_val

def main():
    # Parse necessary arguments
    parser = argparse.ArgumentParser(description='Preprocessing script for image dataset')
    parser.add_argument('coco_annotations_path', type=str, help='Path to the COCO annotations')
    parser.add_argument('input_image_dir', type=str, help='Path to the folder of input images')
    parser.add_argument('output_base_dir', type=str, help='Path to save the results')

    args = parser.parse_args()

    coco_annotation_path = args.coco_annotations_path
    input_image_dir = args.input_image_dir
    output_base_dir = os.path.abspath(args.output_base_dir)


    # Execution of preprocessing tasks
    COCO_to_YOLO.convert(coco_annotation_path, os.path.join(output_base_dir, 'annotations_yolo'))

    resize_images.resize(input_image_dir, os.path.join(output_base_dir, 'images_resized'))

    train_test_val.split_dataset(os.path.join(output_base_dir, 'images_resized'), os.path.join(output_base_dir, 'annotations_yolo'), output_base_dir)


    delete_auxiliary_folders = True # Set to False to keep auxiliary generated folders
    if delete_auxiliary_folders:
        shutil.rmtree(os.path.join(output_base_dir, 'annotations_yolo'))
        shutil.rmtree(os.path.join(output_base_dir, 'images_resized'))

if __name__ == "__main__":
    main()
