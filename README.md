# Trifork ML Developer Technical Assignment

## Converting, Resizing, and Splitting Vegetable Image Dataset for Object Detection Training.

To solve the given task,  the preprocessing.py script has been created. It sequentially transforms the format of provided labels from COCO to YOLO, resizes the images to the specified dimensions, and ultimately performs the split into train, val and test sets. Additionally, an example illustrates how to use the output of this task in the development of a simple Object Detection model based on YOLOv8.

The output of the script, as specified in the instructions, follows the structure:

```bash
Vegetable_OD
├── ...
├── dataset                    
│   ├── raw                           # provided data
│   │   ├── coco.json
│   │   ├── images              
│   ├── preprocessed_dataset_v1       # output 
│   │   ├── images
│   │   │   ├── train
│   │   │   ├── test
│   │   │   ├── val
│   │   ├── labels 
│   │   │   ├── train
│   │   │   ├── test
│   │   │   ├── val
└── ...
```

### User instructions:
1. Clone this project or download the necessary files (*preprocessing.py* and *utils* folder)
2. Install *requirements.txt*:

    `pip install -r requirements.txt`
3. Run 
    
    `python preprocessing.py path\to\coco.json path\to\images desired\output\path`

### Comments on the code:
#### General comments
* Throughout the code, progress messages are printed to the console to indicate the process status. A `verbose` parameter could be implemented to regulate the display of these messages.
* In general, code for this task is lightly commented but could be further detailed if required.
* Code has been developed on Windows. Issues related to file paths and similar functionalities may arise when executing it on Linux.
* Pydocs have been added to the functions in the `utils` folder to facilitate understanding, maintainability and collaborative development.

#### *preprocessing.py*
* This is the task's main script, where the preprocessing pipeline is put together. 
* A modular approach is used in designing this pipeline. Each specific subtask is executed through dedicated scripts located in the `utils` folder. This modularity makes the code maintainable and organized, allowing for easier updates or additions to individual subtasks without affecting the overall functionality.
* The `delete_auxiliary_folders` parameter alows the user to decide whether to keep the intermediate folders created during the execution.

#### *COCO_to_YOLO.py*
* Both a `labelmap` and a `data.yaml` are generated, as they serve distinct purposes. The `labelmap` is useful for integrating data into tools like Roboflow, while the `data.yaml` is essential for training and validating models.

#### *resize_images.py*
* Since the YOLO annotations are relative to the corresponding image size, there is no need to resize the bounding boxes.
* Additional parameters `target_width`, `target_height` enable the user to decide the output image dimensions. These are defaulted to the dimensions specified in the instructions.

#### *train_test_val.py*
* The size of the train/test/val sets is determined through the parameters, `test_size`, `val_size`, the remaining images will be included in the training set. Also the `random_seed` parameter can be used to ensure reproducibility if a specific value is provided. The default distribution of the train/test/val sets is 70%-20%-10%.
* For better accessibility and ease of checking and visualizing train/test/val images, directories have been created and the corresponding images moved, despite the computational inefficiency, instead of using `.txt` files to associate images with train/test/val sets.
* Despite the initial instruction to name the folder containing annotations as `annotations`, for YOLOv8 model training convenience, it has been renamed to `labels`. However, this can be easily adjusted as needed.

### (ADDITIONAL) Using the result in the development of a simple OD model:

Once the `preprocessing.py` script is executed, we have the raw dataset in a YOLO-compatible format, ready to be used for model training. To demonstrate the application of the preprocessed dataset in training, validation, and prediction of an Object Detection model, the following example is provided:

First of all, to execute any YOLO-related code the `ultralytics` library must be installed:    
 ```bash
 pip install ultralytics```

To **train** the model using pretrained weights run:    
 `yolo detect train data=dataset\preprocessed_dataset_v1\data.yaml model=models\yolov8n.pt epochs=20 imgsz=512 project=runs\detect name=yolo_train_example`

This is a straightforward example utilizing the YOLOv8 Nano pretrained model, training for only 20 epochs without adjusting any of the more complex parameters.

To check the model's performance on the validation set run:    
 `yolo detect val data=dataset\preprocessed_dataset_v1\data.yaml model=runs\detect\yolo_train_example\weights\best.pt project=runs\detect name=yolo_val_example`

The validation metrics obtained from the previous instruction are the following:
```bash
                Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 
                   all         14        104      0.949      0.837      0.948      0.734
               lettuce         14          4      0.885          1      0.995      0.809
                potato         14         13          1      0.959      0.995      0.884
                carrot         14         42       0.97      0.774      0.917      0.696
                 onion         14         11      0.993      0.909       0.96      0.782
                garlic         14         22          1      0.327      0.867      0.626
                  leek         14          9      0.898      0.889      0.908      0.641
              broccoli         14          3      0.898          1      0.995      0.697
```
These metrics indicate a fairly promising performance despite the modest training process that was conducted. 

Finally, the adjusted model can be used on unseen images to detect and classify the specified vegetables. Therefore, to apply the model on the **test** set, run:   
`yolo detect predict source=dataset\preprocessed_dataset_v1\images\test model=runs\detect\yolo_train_example\weights\best.pt project=runs\detect name=yolo_pred_example`

Let's now illustrate some of the model's detections on unseen images:

![Prediction grid](runs/detect/yolo_pred_example/pred_grid.png)
