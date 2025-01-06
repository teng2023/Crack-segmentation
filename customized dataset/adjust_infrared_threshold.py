import cv2
import os

heat_img_path='depth_dataset/crop_images/heat_images'
heat_img_list=os.listdir(heat_img_path)

new_heat_path='depth_dataset/crop_images/new_heat_images'   
if not os.path.exists(new_heat_path):
    os.makedirs(new_heat_path)                 

for img_name in heat_img_list:
    img=cv2.imread(f'{heat_img_path}/{img_name}',cv2.IMREAD_GRAYSCALE)

    for x in range(400):
        for y in range(400):
            if img[x][y]>=160:
                img[x][y]=160
            if img[x][y]<=90:
                img[x][y]=90
            img[x][y]=img[x][y]*255/70-2295/7
    
    cv2.imwrite(f'{new_heat_path}/{img_name}',img)


# print(heat_img_gray)
# print(heat_img_gray.max())
# print(heat_img_gray.min())
# cv2.imshow('a',heat_img_gray)
# cv2.waitKey(0)                                        