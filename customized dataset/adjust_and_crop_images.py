import cv2
import os
import numpy as np

###############################################################################################################################
#decide where to crop the image

# heat_img=cv2.imread('depth_dataset/heat_images/heat_1.jpg')
# # heat_img=heat_img[73:479,4:556]
# original_img=cv2.imread('depth_dataset/depth_images/depth_1.jpg')
# # original_img=original_img[13:895,175:1390]

# original_img = cv2.resize(original_img, (853, 480))
# original_img = original_img[19:453, 75:715]
# heat_img = heat_img[46:, :]
# print('heat_img.shape: ', heat_img.shape)
# print('original_img.shape: ', original_img.shape)

# # cv2.rectangle(heat_img, (4, 73), (556, 479),(0,0,255),3)
# # cv2.rectangle(original_img, (175, 13), (1390, 895),(0,0,255),3)

# def click_a(event, x, y, flags, param):
#     if flags == 1:
#         if event == 1:
#             print('a_x: ', x)
#             print('a_y: ', y)

# def click_b(event, x, y, flags, param):
#     if flags == 1:
#         if event == 1:
#             print('b_x: ', x)
#             print('b_y: ', y)


# # cv2.imshow('a',heat_img)
# cv2.imshow('a',original_img)
# cv2.imshow('b',heat_img)
# cv2.setMouseCallback('a', click_a)
# cv2.setMouseCallback('b', click_b)
# cv2.waitKey(0)
###############################################################################################################################

original_img_path='depth_dataset/raw_images/original_images'
depth_img_path='depth_dataset/raw_images/depth_images'
heat_img_path='depth_dataset/raw_images/heat_images'

crop_depth_img_path='depth_dataset/crop_images/depth_images'
crop_heat_img_path='depth_dataset/crop_images/heat_images'
crop_original_img_path='depth_dataset/crop_images/original_images'

if not os.path.exists(crop_depth_img_path):
    os.makedirs(crop_depth_img_path)
if not os.path.exists(crop_heat_img_path):
    os.makedirs(crop_heat_img_path)
if not os.path.exists(crop_original_img_path):
    os.makedirs(crop_original_img_path)

original_img_list=os.listdir(original_img_path)
depth_img_list=os.listdir(depth_img_path)
heat_img_list=os.listdir(heat_img_path)

# adjust sorting method numerically (data_all)
original_img_list.sort(key=lambda x:int(x[9:-4]))

if len(original_img_list)==len(depth_img_list) and len(original_img_list)==len(heat_img_list):
    #change the heat images to wright file name
    original_number_list=[]
    for img in original_img_list:
        original_number_list.append(img[9:-4])
    
    heat_number_list=[]
    for img in heat_img_list:
        if 'heat' in img:
            heat_number_list.append(img[5:-4])
    heat_number_list.sort(key=lambda x:int(x))

    rename_number_list=list(set(original_number_list)-set(heat_number_list))
    rename_number_list.sort(key=lambda x:int(x))

    if len(rename_number_list)!=0:
        for i in range(len(rename_number_list)):
            old_name=os.path.join(heat_img_path,heat_img_list[i])
            new_name=os.path.join(heat_img_path,f'heat_{rename_number_list[i]}.jpg')
            os.rename(old_name,new_name)

    #crop the images and resize to (400,400)
    for i in range(len(heat_img_list)):
        heat_img=cv2.imread(os.path.join(heat_img_path,heat_img_list[i]))
        original_img=cv2.imread(os.path.join(original_img_path,original_img_list[i]))
        depth_img=cv2.imread(os.path.join(depth_img_path,depth_img_list[i]))

        # resize the original images and depth images
        original_img=cv2.resize(original_img, (853, 480))
        depth_img=cv2.resize(depth_img, (853, 480))

        #crop the images
        heat_img=heat_img[46:,:]
        original_img=original_img[19:453,75:715]
        depth_img=depth_img[19:453,75:715]

        #resize to (h,w)=(400,400)
        heat_img=cv2.resize(heat_img,(400,400))
        original_img=cv2.resize(original_img,(400,400))
        depth_img=cv2.resize(depth_img,(400,400))

        #save the images
        cv2.imwrite(os.path.join(crop_heat_img_path,heat_img_list[i]),heat_img)
        cv2.imwrite(os.path.join(crop_original_img_path,original_img_list[i]),original_img)
        cv2.imwrite(os.path.join(crop_depth_img_path,depth_img_list[i]),depth_img)

###############################################################################################################################
#brighter the dark images, adjust the brightness and label

# label_img_path='depth_dataset/crop_images/label_original'
# label_img_list=os.listdir(label_img_path)

# # sort the list in correct way
# label_img_list.sort(key=lambda x:int(x[9:-5]))

# original_img_list_2=original_img_list

# # remove the file type in file name
# for i in range(len(original_img_list_2)):
#     original_img_list_2[i]=original_img_list_2[i][:-4]
# for i in range(len(label_img_list)):
#     label_img_list[i]=label_img_list[i][:-5]

# # check whether the images should be brighter
# brighter_img_list=list(set(original_img_list_2)-set(label_img_list))

# # # sort the list in correct way
# brighter_img_list.sort(key=lambda x:int(x[9:]))

# only run this code when some dark images can not be labeled
# file=open('depth_dataset/raw_images/brighter_list_final.txt','r')
# images=file.readlines()


## for img in images:
#     brighter_img=cv2.imread(f'depth_dataset/crop_images/original_images/{img}.jpg'.replace('\n',''))
#     contrast=250
#     brightness=250
#     output=brighter_img*(contrast/127+1)-contrast+brightness
#     output=np.clip(output, 0, 255)
#     output=np.uint8(output)
#     cv2.imwrite(os.path.join(crop_original_img_path,f'{img}.jpg'.replace('\n','')),output)

# with open('depth_dataset/crop_images/brighter_list.txt','w') as f:
#     for img in brighter_img_list:
#         brighter_img=cv2.imread(f'depth_dataset/crop_images/original_images/{img}.jpg')
#         contrast=250
#         brightness=250
#         output=brighter_img*(contrast/127+1)-contrast+brightness
#         # 轉換公式參考 https://stackoverflow.com/questions/50474302/how-do-i-adjust-brightness-contrast-and-vibrance-with-opencv-python
#         # 調整後的數值大多為浮點數，且可能會小於 0 或大於 255
#         # 為了保持像素色彩區間為 0～255 的整數，所以再使用 np.clip() 和 np.uint8() 進行轉換
#         output=np.clip(output, 0, 255)
#         output=np.uint8(output)
#         cv2.imwrite(os.path.join(crop_original_img_path,f'{img}.jpg'),output)
#         f.write(f'{img}\n')
