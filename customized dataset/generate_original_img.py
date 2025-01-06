import cv2
import numpy as np
import os
import random

ori_iou_list_raw=open('depth_dataset/crop_images/generate_img_original.txt','r')
ori_iou_list_raw=ori_iou_list_raw.readlines()

ori_iou_list=[]
for i in ori_iou_list_raw:
    img_name=i.split(sep='\t')
    ori_iou_list.append(img_name[0])

heat_iou_list_raw=open('depth_dataset/crop_images/generate_img_heat.txt','r')
heat_iou_list_raw=heat_iou_list_raw.readlines()

heat_iou_list=[]
for i in heat_iou_list_raw:
    img_name=i.split(sep='\t')
    heat_iou_list.append(img_name[0])

depth_iou_list_raw=open('depth_dataset/crop_images/generate_img_depth.txt','r')
depth_iou_list_raw=depth_iou_list_raw.readlines()

depth_iou_list=[]
for i in depth_iou_list_raw:
    img_name=i.split(sep='\t')
    depth_iou_list.append(img_name[0])

# print(len(ori_iou_list))
# print(len(depth_iou_list))
# print(len(heat_iou_list))
# print(len(set(ori_iou_list+depth_iou_list+heat_iou_list)))

generate_list=set(ori_iou_list+depth_iou_list+heat_iou_list)

# the images to augment
original_img_path='depth_dataset/crop_images/original_images'
depth_img_path='depth_dataset/crop_images/depth_images'
heat_img_path='depth_dataset/crop_images/heat_images'
label_img_path='depth_dataset/json_to_image_result/SegmentationClassPNG'

#############################################################################################
# add noise
ratio_1=0.09
ratio_2=0.045
ratio_3=0.0225

# the function to create the noise images
def random_white_noise(image1,image2,image3,ratio):
    # img=np.copy(image)
    size=image1.size
    white_point_num=np.ceil(ratio*size).astype('int')
    row,column,_=image1.shape

    x=np.random.randint(0,column-1,white_point_num)
    y=np.random.randint(0,row-1,white_point_num)
    image1[y,x]=255
    image2[y,x]=255
    image3[y,x]=255
#############################################################################################
# rotate the images

angle_1=65
angle_2=130
angle_3=195
angle_4=260
angle_5=325

# original part
# angle_1=45
# angle_2=90
# angle_3=135
# angle_4=180
# angle_5=225
# angle_6=270
# angle_7=315

# heat part
# angle_11=37.5
# angle_12=75
# angle_13=112.5
# angle_14=150
# angle_15=187.5
# angle_16=262.5
# angle_17=300
# angle_18=337.5

def rotate_img(img1,img2,img3,img4,angle):
    (h,w,d)=img1.shape
    center=(w//2,h//2)

    M=cv2.getRotationMatrix2D(center,angle,1.5)
    img1=cv2.warpAffine(img1,M,(w,h))
    img2=cv2.warpAffine(img2,M,(w,h))
    img3=cv2.warpAffine(img3,M,(w,h))
    img4=cv2.warpAffine(img4,M,(w,h))

    return img1,img2,img3,img4

############################################################################################
# i=1
# with open('depth_dataset/raw_images/dark_images(noise).txt','w') as f:
#     for ratio in [ratio_1,ratio_2,ratio_3]:
#         for img in dark_img_list:
#             original_img=cv2.imread(os.path.join(original_img_path,img).replace('\n',''))
#             heat_img=cv2.imread(os.path.join(heat_img_path,img).replace('\n','').replace('original','heat'))
#             depth_img=cv2.imread(os.path.join(depth_img_path,img).replace('\n','').replace('original','depth'))
#             label_img=cv2.imread(os.path.join(label_img_path,img).replace('\n','').replace('jpg','png'))

#             random_white_noise(original_img,heat_img,depth_img,ratio)

#             cv2.imwrite(os.path.join(original_img_path,img.replace('original',f'n{i}_original')).replace('\n',''),original_img)
#             cv2.imwrite(os.path.join(heat_img_path,img).replace('\n','').replace('original',f'n{i}_heat'),heat_img)
#             cv2.imwrite(os.path.join(depth_img_path,img).replace('\n','').replace('original',f'n{i}_depth'),depth_img)
#             cv2.imwrite(os.path.join(label_img_path,img).replace('\n','').replace('jpg','png').replace('original',f'n{i}_original'),label_img)

#             f.write(f'{img}'.replace('original',f'n{i}_original'))

#         i+=1

# dark_img_noise_list=open('depth_dataset/raw_images/dark_images(noise).txt','r')
# dark_img_noise_list=dark_img_noise_list.readlines()

# final_list=dark_img_list+dark_img_noise_list

###########################################################################################
# the number start from
img_number=362

# generate the low iou part from original images

# with open('depth_dataset/raw_images/dark_images(rotate).txt','w') as f:
for angle in [angle_1,angle_2,angle_3,angle_4,angle_5]:
    for img in generate_list:
        if 'c_'in img:
            continue
        elif 'cc_' in img:
            continue
        elif 'v_' in img:
            continue
        elif 'h_' in img:
            continue
        elif 'vh_' in img:
            continue

        original_img=cv2.imread(os.path.join(original_img_path,img).replace('\n',''))
        heat_img=cv2.imread(os.path.join(heat_img_path,img).replace('\n','').replace('original','heat'))
        depth_img=cv2.imread(os.path.join(depth_img_path,img).replace('\n','').replace('original','depth'))
        label_img=cv2.imread(os.path.join(label_img_path,img).replace('\n','').replace('jpg','png'))

        original_img,heat_img,depth_img,label_img=rotate_img(original_img,heat_img,depth_img,label_img,angle)
        # cv2.imshow('a',original_img)
        # cv2.waitKey(0)
        # break
        # cv2.imwrite(os.path.join(original_img_path,img.replace('original',f'r{i}_original')).replace('\n',''),original_img)
        # cv2.imwrite(os.path.join(heat_img_path,img).replace('\n','').replace('original',f'r{i}_heat'),heat_img)
        # cv2.imwrite(os.path.join(depth_img_path,img).replace('\n','').replace('original',f'r{i}_depth'),depth_img)
        # cv2.imwrite(os.path.join(label_img_path,img).replace('\n','').replace('jpg','png').replace('original',f'r{i}_original'),label_img)

        cv2.imwrite(os.path.join(original_img_path,f'original_{img_number}.jpg'),original_img)
        cv2.imwrite(os.path.join(heat_img_path,f'heat_{img_number}.jpg'),heat_img)
        cv2.imwrite(os.path.join(depth_img_path,f'depth_{img_number}.jpg'),depth_img)
        cv2.imwrite(os.path.join(label_img_path,f'original_{img_number}.png'),label_img)

        # f.write(f'{img}'.replace('original',f'r{i}_original'))
        img_number+=1

print('generate complete (depth part)')

# generate high iou part from heat images
# for angle in [angle_1,angle_2,angle_3,angle_4,angle_5]:
#     for img in heat_iou_list:
#         if 'c_'in img:
#             continue
#         elif 'cc_' in img:
#             continue
#         elif 'v_' in img:
#             continue
#         elif 'h_' in img:
#             continue
#         elif 'vh_' in img:
#             continue

#         original_img=cv2.imread(os.path.join(original_img_path,img).replace('\n',''))
#         heat_img=cv2.imread(os.path.join(heat_img_path,img).replace('\n','').replace('original','heat'))
#         depth_img=cv2.imread(os.path.join(depth_img_path,img).replace('\n','').replace('original','depth'))
#         label_img=cv2.imread(os.path.join(label_img_path,img).replace('\n','').replace('jpg','png'))

#         original_img,heat_img,depth_img,label_img=rotate_img(original_img,heat_img,depth_img,label_img,angle)
#         # cv2.imshow('a',original_img)
#         # cv2.waitKey(0)
#         # break
#         # cv2.imwrite(os.path.join(original_img_path,img.replace('original',f'r{i}_original')).replace('\n',''),original_img)
#         # cv2.imwrite(os.path.join(heat_img_path,img).replace('\n','').replace('original',f'r{i}_heat'),heat_img)
#         # cv2.imwrite(os.path.join(depth_img_path,img).replace('\n','').replace('original',f'r{i}_depth'),depth_img)
#         # cv2.imwrite(os.path.join(label_img_path,img).replace('\n','').replace('jpg','png').replace('original',f'r{i}_original'),label_img)

#         cv2.imwrite(os.path.join(original_img_path,f'original_{img_number}.jpg'),original_img)
#         cv2.imwrite(os.path.join(heat_img_path,f'heat_{img_number}.jpg'),heat_img)
#         cv2.imwrite(os.path.join(depth_img_path,f'depth_{img_number}.jpg'),depth_img)
#         cv2.imwrite(os.path.join(label_img_path,f'original_{img_number}.png'),label_img)

#         # f.write(f'{img}'.replace('original',f'r{i}_original'))
#         img_number+=1

# print('generate complete (heat part)')

# dark_img_rotate_list=open('depth_dataset/raw_images/dark_images(rotate).txt','r')
# dark_img_rotate_list=dark_img_rotate_list.readlines()

# final_list=dark_img_list+dark_img_rotate_list


# write to the .txt file
# with open('depth_dataset/raw_images/dark_final.txt','w') as f:
#     for img in final_list:
#         f.write(img)
