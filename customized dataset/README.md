# Description

The the number of the images from every sensors (visible light, infrared, and depth) is 332, and their size, and the deviation of each camera are different.

## Meaning of the files

**crop images**：containing the images after executing "adjust_and_crop_images.py", and sorting the original images into bright images and dark images, so the directories "bright_images" and "dark_images" are from original images and sorting manually. "label_original" is the json file from labeling original images. "generate_img_depth.txt", "generate_img_infrared.txt", and "generate_img_original.txt" are used to judge which images should be augment.


![dataset_information](https://github.com/teng2023/Crack-segmentation-on-customized-dataset-with-visible-ight-infrared-depth-images/blob/main/customized%20dataset/dataset_information.png)
