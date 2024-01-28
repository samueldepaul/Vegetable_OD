import cv2
import os
from tqdm import tqdm


def resize(input_image_dir, output_image_dir, target_width=1024, target_height=768):
    """
    Resizes images in a directory to a specified width and height.

    Args:
        input_image_dir (str): Path to the directory containing input images.
        output_image_dir (str): Path to the directory where resized images will be saved.
        target_width (int): Target width for resized images.
        target_height (int): Target height for resized images.
    """    

    # Create output directory
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    image_files = [f for f in os.listdir(input_image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Load and resize images
    for image_file in tqdm(image_files, desc="Resizing images", unit="image"): # tqdm creates a progress bar
        
        image_path = os.path.join(input_image_dir, image_file)
        image = cv2.imread(image_path)

        resized_image = cv2.resize(image, (target_width, target_height))

        output_image_path = os.path.join(output_image_dir, image_file)
        cv2.imwrite(output_image_path, resized_image)