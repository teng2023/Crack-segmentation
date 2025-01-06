import cv2
import numpy as np
import os

original_img_path='depth_dataset/crop_images/original_images'
depth_img_path='depth_dataset/crop_images/depth_images'
heat_img_path='depth_dataset/crop_images/heat_images'
label_img_path='depth_dataset/json_to_image_result/SegmentationClassPNG'

train_img_txt=open('depth_dataset/json_to_image_result/data_train1.txt')
train_img_txt=train_img_txt.readlines()

original_img_list=os.listdir(original_img_path)
heat_img_list=os.listdir(heat_img_path)
depth_img_list=os.listdir(depth_img_path)
label_img_list=os.listdir(label_img_path)

task_dict={1:'5 flips',2:'combine',3:'rotate'}
task=task_dict[2]

############################################################################################################
# do flip augmentations (5 methods)

if task=='5 flips':
    # img_type_list=[(original_img_path,original_img_list),(heat_img_path,heat_img_list),(depth_img_path,depth_img_list),(label_img_path,label_img_list)]

    # for img_path,img_type in img_type_list:
    #     for img in img_type:
    #         image=cv2.imread(f'{img_path}/{img}')

    #         # 5 types of augmentation
    #         vertical_flip_img=cv2.flip(image,1)
    #         horizontal_flip_img=cv2.flip(image,0)
    #         vh_flip_img=cv2.flip(image,-1)
    #         # clockwise_rotate_img=cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    #         # counter_rotate_img=cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    #         #save the iamges
    #         cv2.imwrite(f'{img_path}/v_{img}',vertical_flip_img)
    #         cv2.imwrite(f'{img_path}/h_{img}',horizontal_flip_img)
    #         cv2.imwrite(f'{img_path}/vh_{img}',vh_flip_img)
    #         # cv2.imwrite(f'{img_path}/c_{img}',clockwise_rotate_img)
    #         # cv2.imwrite(f'{img_path}/cc_{img}',counter_rotate_img)
        
    #     print(f'finish generating {img_path}')

        for img in train_img_txt:
            original_img=cv2.imread(os.path.join(original_img_path,img).replace('\n',''))
            depth_img=cv2.imread(os.path.join(depth_img_path,img).replace('\n','').replace('original','depth'))
            heat_img=cv2.imread(os.path.join(heat_img_path,img).replace('\n','').replace('original','heat'))
            label_img=cv2.imread(os.path.join(label_img_path,img).replace('\n','').replace('jpg','png'))

            # 5 types of augmentation
            vertical_flip_img=cv2.flip(original_img,1)
            horizontal_flip_img=cv2.flip(original_img,0)
            vh_flip_img=cv2.flip(original_img,-1)
            clockwise_rotate_img=cv2.rotate(original_img, cv2.ROTATE_90_CLOCKWISE)
            counter_rotate_img=cv2.rotate(original_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

            cv2.imwrite(f'{original_img_path}/v_{img}'.replace('\n',''),vertical_flip_img)
            cv2.imwrite(f'{original_img_path}/h_{img}'.replace('\n',''),horizontal_flip_img)
            cv2.imwrite(f'{original_img_path}/vh_{img}'.replace('\n',''),vh_flip_img)
            cv2.imwrite(f'{original_img_path}/c_{img}'.replace('\n',''),clockwise_rotate_img)
            cv2.imwrite(f'{original_img_path}/cc_{img}'.replace('\n',''),counter_rotate_img)

            vertical_flip_img=cv2.flip(depth_img,1)
            horizontal_flip_img=cv2.flip(depth_img,0)
            vh_flip_img=cv2.flip(depth_img,-1)
            clockwise_rotate_img=cv2.rotate(depth_img, cv2.ROTATE_90_CLOCKWISE)
            counter_rotate_img=cv2.rotate(depth_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

            cv2.imwrite(f'{depth_img_path}/v_{img}'.replace('\n','').replace('original','depth'),vertical_flip_img)
            cv2.imwrite(f'{depth_img_path}/h_{img}'.replace('\n','').replace('original','depth'),horizontal_flip_img)
            cv2.imwrite(f'{depth_img_path}/vh_{img}'.replace('\n','').replace('original','depth'),vh_flip_img)
            cv2.imwrite(f'{depth_img_path}/c_{img}'.replace('\n','').replace('original','depth'),clockwise_rotate_img)
            cv2.imwrite(f'{depth_img_path}/cc_{img}'.replace('\n','').replace('original','depth'),counter_rotate_img)

            vertical_flip_img=cv2.flip(heat_img,1)
            horizontal_flip_img=cv2.flip(heat_img,0)
            vh_flip_img=cv2.flip(heat_img,-1)
            clockwise_rotate_img=cv2.rotate(heat_img, cv2.ROTATE_90_CLOCKWISE)
            counter_rotate_img=cv2.rotate(heat_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

            cv2.imwrite(f'{heat_img_path}/v_{img}'.replace('\n','').replace('original','heat'),vertical_flip_img)
            cv2.imwrite(f'{heat_img_path}/h_{img}'.replace('\n','').replace('original','heat'),horizontal_flip_img)
            cv2.imwrite(f'{heat_img_path}/vh_{img}'.replace('\n','').replace('original','heat'),vh_flip_img)
            cv2.imwrite(f'{heat_img_path}/c_{img}'.replace('\n','').replace('original','heat'),clockwise_rotate_img)
            cv2.imwrite(f'{heat_img_path}/cc_{img}'.replace('\n','').replace('original','heat'),counter_rotate_img)

            vertical_flip_img=cv2.flip(label_img,1)
            horizontal_flip_img=cv2.flip(label_img,0)
            vh_flip_img=cv2.flip(label_img,-1)
            clockwise_rotate_img=cv2.rotate(label_img, cv2.ROTATE_90_CLOCKWISE)
            counter_rotate_img=cv2.rotate(label_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

            cv2.imwrite(f'{label_img_path}/v_{img}'.replace('\n','').replace('jpg','png'),vertical_flip_img)
            cv2.imwrite(f'{label_img_path}/h_{img}'.replace('\n','').replace('jpg','png'),horizontal_flip_img)
            cv2.imwrite(f'{label_img_path}/vh_{img}'.replace('\n','').replace('jpg','png'),vh_flip_img)
            cv2.imwrite(f'{label_img_path}/c_{img}'.replace('\n','').replace('jpg','png'),clockwise_rotate_img)
            cv2.imwrite(f'{label_img_path}/cc_{img}'.replace('\n','').replace('jpg','png'),counter_rotate_img)

        print('complete generate')

        cc=[]
        c=[]
        vh=[]
        v=[]
        h=[]
        o=[]
        
        for img_name in train_img_txt:
            o.append(img_name.replace('\n',''))
            c.append(img_name.replace('original','c_original').replace('\n',''))
            cc.append(img_name.replace('original','cc_original').replace('\n',''))
            v.append(img_name.replace('original','v_original').replace('\n',''))
            h.append(img_name.replace('original','h_original').replace('\n',''))
            vh.append(img_name.replace('original','vh_original').replace('\n',''))

        o.sort(key=lambda x:int(x[9:-4]))
        cc.sort(key=lambda x:int(x[12:-4]))
        c.sort(key=lambda x:int(x[11:-4]))
        vh.sort(key=lambda x:int(x[12:-4]))
        v.sort(key=lambda x:int(x[11:-4]))
        h.sort(key=lambda x:int(x[11:-4]))
        train_img_txt=o+c+cc+v+h+vh

        with open('depth_dataset/json_to_image_result/data_train1.txt','w') as f:
            for img in train_img_txt:
                f.write(f'{img}\n')
        
        print('comlpete append new images to txt file')


############################################################################################################
# combine two images together
elif task=='combine':
    original_img_number_list=[]
    for img in train_img_txt:
        img.replace('\n','')
        original_img_number_list.append(int(img[9:-5]))

    img_start_number=1

    for img_number in original_img_number_list:

        # remove the nubmer from random number list
        picked_number=[]
        picked_number.append(img_number)
        random_number_list=set(original_img_number_list)-set(picked_number)

        # assign left images
        left_img_ori=cv2.imread(os.path.join(original_img_path,'original_'+f'{img_number}'+'.jpg'))
        left_img_depth=cv2.imread(os.path.join(depth_img_path,'depth_'+f'{img_number}'+'.jpg'))
        left_img_heat=cv2.imread(os.path.join(heat_img_path,'heat_'+f'{img_number}'+'.jpg'))
        left_img_label=cv2.imread(os.path.join(label_img_path,'original_'+f'{img_number}'+'.png'))

        left_img_ori=left_img_ori[:,:200]
        left_img_depth=left_img_depth[:,:200]
        left_img_heat=left_img_heat[:,:200]
        left_img_label=left_img_label[:,:200]

        # random choosing the image numbers to combine
        right_number_list=np.random.choice(list(random_number_list),replace=False,size=2)

        for number in right_number_list:

            #assign right images
            right_img_ori=cv2.imread(os.path.join(original_img_path,'original_'+f'{number}'+'.jpg'))
            right_img_depth=cv2.imread(os.path.join(depth_img_path,'depth_'+f'{number}'+'.jpg'))
            right_img_heat=cv2.imread(os.path.join(heat_img_path,'heat_'+f'{number}'+'.jpg'))
            right_img_label=cv2.imread(os.path.join(label_img_path,'original_'+f'{number}'+'.png'))

            right_img_ori=right_img_ori[:,200:400]
            right_img_depth=right_img_depth[:,200:400]
            right_img_heat=right_img_heat[:,200:400]
            right_img_label=right_img_label[:,200:400]

            combine_img_ori=np.hstack((left_img_ori,right_img_ori))
            combine_img_depth=np.hstack((left_img_depth,right_img_depth))
            combine_img_heat=np.hstack((left_img_heat,right_img_heat))
            combine_img_label=np.hstack((left_img_label,right_img_label))

            cv2.imwrite(os.path.join(original_img_path,f'combine_original_{img_start_number}.jpg'),combine_img_ori)
            cv2.imwrite(os.path.join(depth_img_path,f'combine_depth_{img_start_number}.jpg'),combine_img_depth)
            cv2.imwrite(os.path.join(heat_img_path,f'combine_heat_{img_start_number}.jpg'),combine_img_heat)
            cv2.imwrite(os.path.join(label_img_path,f'combine_original_{img_start_number}.png'),combine_img_label)

            img_start_number+=1
            # cv2.imshow('a',combine_img_ori)
            # cv2.waitKey(0)
        #     break
        # break

    print('complete first part')

    # switch the images and do it again
    for img_number in original_img_number_list:

        # remove the nubmer from random number list
        picked_number=[]
        picked_number.append(img_number)
        random_number_list=set(original_img_number_list)-set(picked_number)

        # assign left images
        left_img_ori=cv2.imread(os.path.join(original_img_path,'original_'+f'{img_number}'+'.jpg'))
        left_img_depth=cv2.imread(os.path.join(depth_img_path,'depth_'+f'{img_number}'+'.jpg'))
        left_img_heat=cv2.imread(os.path.join(heat_img_path,'heat_'+f'{img_number}'+'.jpg'))
        left_img_label=cv2.imread(os.path.join(label_img_path,'original_'+f'{img_number}'+'.png'))

        left_img_ori=left_img_ori[:,200:400]
        left_img_depth=left_img_depth[:,200:400]
        left_img_heat=left_img_heat[:,200:400]
        left_img_label=left_img_label[:,200:400]

        # random choosing the image numbers to combine
        right_number_list=np.random.choice(list(random_number_list),replace=False,size=2)

        for number in right_number_list:

            #assign right images
            right_img_ori=cv2.imread(os.path.join(original_img_path,'original_'+f'{number}'+'.jpg'))
            right_img_depth=cv2.imread(os.path.join(depth_img_path,'depth_'+f'{number}'+'.jpg'))
            right_img_heat=cv2.imread(os.path.join(heat_img_path,'heat_'+f'{number}'+'.jpg'))
            right_img_label=cv2.imread(os.path.join(label_img_path,'original_'+f'{number}'+'.png'))

            right_img_ori=right_img_ori[:,:200]
            right_img_depth=right_img_depth[:,:200]
            right_img_heat=right_img_heat[:,:200]
            right_img_label=right_img_label[:,:200]

            combine_img_ori=np.hstack((left_img_ori,right_img_ori))
            combine_img_depth=np.hstack((left_img_depth,right_img_depth))
            combine_img_heat=np.hstack((left_img_heat,right_img_heat))
            combine_img_label=np.hstack((left_img_label,right_img_label))

            cv2.imwrite(os.path.join(original_img_path,f'combine_original_{img_start_number}.jpg'),combine_img_ori)
            cv2.imwrite(os.path.join(depth_img_path,f'combine_depth_{img_start_number}.jpg'),combine_img_depth)
            cv2.imwrite(os.path.join(heat_img_path,f'combine_heat_{img_start_number}.jpg'),combine_img_heat)
            cv2.imwrite(os.path.join(label_img_path,f'combine_original_{img_start_number}.png'),combine_img_label)

            img_start_number+=1

            # cv2.imshow('a',combine_img_ori)
            # cv2.waitKey(0)
        #     break
        # break
    print('finish generate') 

    with open('depth_dataset/json_to_image_result/data_train1.txt','a') as f:
        for i in range(1,len(train_img_txt)*2*2+1):
            f.write(f'combine_original_{i}.jpg\n')

elif task=='rotate':

    def rotate_img(img1,img2,img3,img4,angle):
        (h,w,d)=img1.shape
        center=(w//2,h//2)

        M=cv2.getRotationMatrix2D(center,angle,1.5)
        img1=cv2.warpAffine(img1,M,(w,h))
        img2=cv2.warpAffine(img2,M,(w,h))
        img3=cv2.warpAffine(img3,M,(w,h))
        img4=cv2.warpAffine(img4,M,(w,h))

        return img1,img2,img3,img4
    
    angle_1=97.5
    angle_2=162.5
    angle_3=227.5
    angle_4=292.5
    # angle_5=325
    # angle_6=300
    # angle_7=292.5
    # angle_8=337.5

    angle_number=1

    for angle in [angle_1,angle_2,angle_3,angle_4]:
        
        for img in train_img_txt:
            img=img.replace('\n','')

            original_img=cv2.imread(os.path.join(original_img_path,img))
            depth_img=cv2.imread(os.path.join(depth_img_path,img).replace('original','depth'))
            heat_img=cv2.imread(os.path.join(heat_img_path,img).replace('original','heat'))
            label_img=cv2.imread(os.path.join(label_img_path,img).replace('jpg','png'))

            # cv2.imshow('a',original_img)
            # cv2.waitKey(0)

            original_img,heat_img,depth_img,label_img=rotate_img(original_img,depth_img,heat_img,label_img,angle)

            # cv2.imshow('a',original_img)
            # cv2.waitKey(0)

            cv2.imwrite(os.path.join(original_img_path,f'r{angle_number}_{img}'),original_img)
            cv2.imwrite(os.path.join(depth_img_path,f'r{angle_number}_{img}'.replace('original','depth')),depth_img)
            cv2.imwrite(os.path.join(heat_img_path,f'r{angle_number}_{img}'.replace('original','heat')),heat_img)
            cv2.imwrite(os.path.join(label_img_path,f'r{angle_number}_{img}'.replace('jpg','png')),label_img)

        angle_number+=1

    print('complete generate')

    r1=[]
    r2=[]
    r3=[]
    r4=[]
    # r5=[]
    # r6=[]
    # r7=[]
    # r8=[]
    o=[]
    
    for img_name in train_img_txt:
        o.append(img_name.replace('\n',''))
        r1.append(img_name.replace('original','r1_original').replace('\n',''))
        r2.append(img_name.replace('original','r2_original').replace('\n',''))
        r3.append(img_name.replace('original','r3_original').replace('\n',''))
        r4.append(img_name.replace('original','r4_original').replace('\n',''))
        # r5.append(img_name.replace('original','r5_original').replace('\n',''))
        # r6.append(img_name.replace('original','r4_original').replace('\n',''))
        # r7.append(img_name.replace('original','r4_original').replace('\n',''))
        # r8.append(img_name.replace('original','r4_original').replace('\n',''))

    train_img_txt=o+r1+r2+r3+r4

    with open('depth_dataset/json_to_image_result/data_train1.txt','w') as f:
        for img in train_img_txt:
            f.write(f'{img}\n')