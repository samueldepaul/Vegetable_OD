import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(input_image_dir, input_annotation_dir, output_dir, test_size=0.2, val_size=0.1, random_seed=42):
    """
    Splits a dataset into train, test, and val sets and organizes them into the specified directory structure.

    Args:
        input_image_dir (str): Path to the directory containing input images.
        input_annotation_dir (str): Path to the directory containing annotation files.
        output_dir (str): Path to the directory where the split dataset will be saved.
        test_size (float): Size of the test set. Range 0 - 1.
        val_size (float): Size of the test set. Range 0 - 1.
        random_seed (int): Seed for random splitting.
    """

    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = [f for f in os.listdir(input_image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Split into train/test/val sets
    train_images, test_val_images = train_test_split(image_files, test_size=(test_size + val_size), random_state=random_seed)
    test_images, val_images = train_test_split(test_val_images, test_size=val_size/(test_size + val_size), random_state=random_seed)

    # Save images/labels accordingly
    for dataset in ['train', 'test', 'val']:
        output_image_dataset_dir = os.path.join(output_dir, 'images', dataset)
        output_annotation_dataset_dir = os.path.join(output_dir, 'labels', dataset)

        os.makedirs(output_image_dataset_dir, exist_ok=True)
        os.makedirs(output_annotation_dataset_dir, exist_ok=True)
    
    print("Splitting into train/val/test sets...")
    for dataset, images in zip(['train', 'test', 'val'], [train_images, test_images, val_images]):
        for image_file in images:

            src_image_path = os.path.join(input_image_dir, image_file)
            dest_image_path = os.path.join(output_dir, 'images', dataset, image_file)
            shutil.copy(src_image_path, dest_image_path)

            annotation_file = os.path.splitext(image_file)[0] + '.txt'
            src_annotation_path = os.path.join(input_annotation_dir, annotation_file)
            dest_annotation_path = os.path.join(output_dir, 'labels', dataset, annotation_file)
            shutil.copy(src_annotation_path, dest_annotation_path)
