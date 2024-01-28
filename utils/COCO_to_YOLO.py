import json
import os
import yaml
from tqdm import tqdm

def convert(coco_file, output_folder):
    """
    Converts COCO format annotations to YOLO format and generates labelmap and data.yaml

    Args:
        coco_file (str): Path to the COCO format annotation file.
        output_folder (str): Path to the output folder where converted files will be saved.
    """

    # Create output directory
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read annotation file
    with open(coco_file, 'r') as infile:
        coco_data = json.load(infile)

    # Create labelmap
    print("Creating labelmap...")
    labelmap = {0: None}

    for category in coco_data["categories"]:
        label_id = category["id"]
        label_name = category["name"]
        labelmap[label_id] = label_name
    labelmap = dict(sorted(labelmap.items()))

    labelmap_file_path = os.path.join(output_folder, "..", "labelmap.txt")
    with open(labelmap_file_path, 'w') as labelmap_file:
        for label_id, label_name in labelmap.items():
            labelmap_file.write(f"{label_name}\n")

    # Create data.yaml
    print("Creating data.yaml...")
    data_yaml_path = os.path.join(output_folder, "..", "data.yaml")
    data_yaml = {
        "path": os.path.join(output_folder, ".."),  
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",  # optional
        "names": dict(enumerate(labelmap.values()))
}
    

    with open(data_yaml_path, 'w') as data_yaml_file:
        yaml.dump(data_yaml, data_yaml_file, default_flow_style=False)

    # Create YOLO format label files from coco.json file
    for image in tqdm(coco_data["images"], desc="Converting labels to YOLO format", unit="label"): # tqdm creates a progress bar
        image_id = image["id"]
        image_file_name = image["file_name"]

        image_annotations = [anno for anno in coco_data["annotations"] if anno["image_id"] == image_id]

        yolo_file_path = os.path.join(output_folder, os.path.splitext(image_file_name)[0] + ".txt")

        with open(yolo_file_path, 'w') as yolo_file:
            for annotation in image_annotations:
                category_id = annotation["category_id"]
                bbox = annotation["bbox"]

                image_width = image["width"]
                image_height = image["height"]

                x_center = (bbox[0] + bbox[2] / 2) / image_width
                y_center = (bbox[1] + bbox[3] / 2) / image_height
                width = bbox[2] / image_width
                height = bbox[3] / image_height

                yolo_line = f"{category_id} {x_center} {y_center} {width} {height}\n"
                yolo_file.write(yolo_line)
