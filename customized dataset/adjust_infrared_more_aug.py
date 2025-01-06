import cv2
import numpy as np
import os

heat_list_raw=open('depth_dataset/json_to_image_result/data_train1.txt','r')
heat_list=heat_list_raw.readlines()
noise_heat_path='depth_dataset/crop_images/noise_heat_images'
label_img_path='depth_dataset/json_to_image_result/SegmentationClassPNG'
new_label_path='depth_dataset/crop_images/noise_heat_images_label'

#############################################################################################
# add noise
ratio_1=0.09
ratio_2=0.045
ratio_3=0.0225

# the function to create the noise images
def random_white_noise(image1,ratio):
    # img=np.copy(image)
    size=image1.size
    white_point_num=np.ceil(ratio*size).astype('int')
    row,column,_=image1.shape

    x=np.random.randint(0,column-1,white_point_num)
    y=np.random.randint(0,row-1,white_point_num)
    image1[y,x]=255

#############################################################################################
# for img_name in heat_list:
#     heat_img=cv2.imread(f'depth_dataset/crop_images/heat_images/{img_name}'.replace('original','heat').replace('\n',''))
#     label_img=cv2.imread(os.path.join(label_img_path,img_name.replace('jpg\n','png')))

#     if 'combine' in img_name:
#         continue
#     else:
        # random_white_noise(heat_img,ratio_1)
        # cv2.imwrite(os.path.join(noise_heat_path,f'noise1_{img_name}'.replace('original','heat').replace('\n','')),heat_img)
        # cv2.imwrite(os.path.join(new_label_path,f'noise1_{img_name}'.replace('jpg\n','png')),label_img)
        # random_white_noise(heat_img,ratio_2)
        # cv2.imwrite(os.path.join(noise_heat_path,f'noise2_{img_name}'.replace('original','heat').replace('\n','')),heat_img)
        # cv2.imwrite(os.path.join(new_label_path,f'noise2_{img_name}'.replace('jpg\n','png')),label_img)
        # random_white_noise(heat_img,ratio_3)
        # cv2.imwrite(os.path.join(noise_heat_path,f'noise3_{img_name}'.replace('original','heat').replace('\n','')),heat_img)
        # cv2.imwrite(os.path.join(new_label_path,f'noise3_{img_name}'.replace('jpg\n','png')),label_img)

        # with open('depth_dataset/json_to_image_result/noise_heat.txt','a') as f:
        #     f.write(f'noise1_{img_name}')
        #     f.write(f'noise2_{img_name}')
        #     f.write(f'noise3_{img_name}')
noise_heat_txt='depth_dataset/json_to_image_result/noise_heat.txt'
train_data_txt=f'depth_dataset/json_to_image_result/data_train1.txt'
file1=open(train_data_txt,'r')
data1=file1.readlines()
file2=open(noise_heat_txt,'r')
data2=file2.readlines()

data=data1+data2
with open('depth_dataset/json_to_image_result/data_train_noise_heat1.txt','w') as f:
    for img in data:
        f.write(f'{img}')