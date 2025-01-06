# Description

The the number of the images from every sensors (visible light, infrared, and depth) is 332, and their size, and the deviation of each camera are different.

## Meaning of the files

### Directories

* **crop_images**：containing the images after executing "adjust_and_crop_images.py", and sorting the original images into bright images and dark images, so the directories "bright_images" and "dark_images" are from original images which sorting manually. "label_original" is the json file from labeling original images. "generate_img_depth.txt", "generate_img_infrared.txt", and "generate_img_original.txt" are used to judge which images should be augment.

* **json_to_image_result**：containing the images after executing "adjust_and_crop_images.py". All the txt files are the list of corresponding images. "test_img_iou.txt", "test_img_iou_heat.txt", "test_img_iou_depth.txt", "test_img_iou_original.txt" are the crack IoU result when testing.

* **raw_images**：containing all the images before doing anything. "test_img_iou" just for some previous selection to decide which image should be augment.

### Files

* **split_dataset.py**：split the dataset into training set and testing set.

* **json_to_image.py** and **label.txt**：To convert json file to image file, it will generate the directory "json_to_image_result".

* **align_sample.py**：An example of calibrating the images (visible light and depth) from camera REALSense.

* **depth_camera_utilization.py**：To take pictures from camera REALSense, both visible light and depth images.

* **adjust_and_crop_images.py**：To calibrate the deviation between visible light images and infrared images, and crop the images to 400x400 to fit the input of the model.

* **generate_original_img.py**：To generate more images, but doesn't represent augmentation.

* **generate_augment_img.py**：To generate augment images.

* **adjust_infrared_threshold.py**：To adjust the scale of infrared images and change to gray scale.

* **adjust_infrared_more_aug.py**：To generate more augment images only for infrared images, like adding noise.

![dataset_information](https://github.com/teng2023/Crack-segmentation-on-customized-dataset-with-visible-ight-infrared-depth-images/blob/main/customized%20dataset/dataset_information.png)
