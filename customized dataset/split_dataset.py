import os
import random

original_img_path='depth_dataset/crop_images/original_images'
img_name='original'

split_ratio=0.7

# create bright_images.txt
bright_img_list=os.listdir('depth_dataset/crop_images/bright_images')
# cc=[]
# c=[]
# vh=[]
# v=[]
# h=[]
# for img_name in bright_img_list:
    # c.append(img_name.replace('original','c_original'))
    # cc.append(img_name.replace('original','cc_original'))
    # v.append(img_name.replace('original','v_original'))
    # h.append(img_name.replace('original','h_original'))
    # vh.append(img_name.replace('original','vh_original'))

bright_img_list.sort(key=lambda x:int(x[9:-4]))
# cc.sort(key=lambda x:int(x[12:-4]))
# c.sort(key=lambda x:int(x[11:-4]))
# vh.sort(key=lambda x:int(x[12:-4]))
# v.sort(key=lambda x:int(x[11:-4]))
# h.sort(key=lambda x:int(x[11:-4]))
# bright_img_list=bright_img_list+v+h+vh

with open('depth_dataset/crop_images/bright_images.txt','w') as f:
    for i in range(len(bright_img_list)):
        f.write(f'{bright_img_list[i]}\n')

# create dark images.txt
dark_img_list=os.listdir('depth_dataset/crop_images/dark_images')
# cc=[]
# c=[]
# vh=[]
# v=[]
# h=[]
# for img_name in dark_img_list:
    # c.append(img_name.replace('original','c_original'))
    # cc.append(img_name.replace('original','cc_original'))
    # v.append(img_name.replace('original','v_original'))
    # h.append(img_name.replace('original','h_original'))
    # vh.append(img_name.replace('original','vh_original'))

dark_img_list.sort(key=lambda x:int(x[9:-4]))
# cc.sort(key=lambda x:int(x[12:-4]))
# c.sort(key=lambda x:int(x[11:-4]))
# vh.sort(key=lambda x:int(x[12:-4]))
# v.sort(key=lambda x:int(x[11:-4]))
# h.sort(key=lambda x:int(x[11:-4]))
# dark_img_list=dark_img_list+v+h+vh

with open('depth_dataset/crop_images/dark_images.txt','w') as f:
    for i in range(len(dark_img_list)):
        f.write(f'{dark_img_list[i]}\n')
# dark_img_list=open('depth_dataset/crop_images/dark_final.txt','r')
# dark_img_list=dark_img_list.readlines()
# for i in range(len(dark_img_list)):
#     dark_img_list[i]=dark_img_list[i].replace('\n','')

# counting data number
original_img_list=os.listdir(original_img_path)
original_img_number=len(original_img_list)

# create data_all.txt to store all the file name of the data
with open('depth_dataset/json_to_image_result/data_all.txt','w') as f:
    for i in range(original_img_number):
        f.write(f'{original_img_list[i]}\n')
#######################################################################################
# randomly split dataset
dark_train_number=round(len(dark_img_list)*split_ratio)

bright_train_number=round(len(bright_img_list)*split_ratio)

dark_train_list=random.sample(dark_img_list,k=dark_train_number)
dark_test_list=list(set(dark_img_list)-set(dark_train_list))

bright_train_list=random.sample(bright_img_list,k=bright_train_number)
bright_test_list=list(set(bright_img_list)-set(bright_train_list))

train_set_list=dark_train_list+bright_train_list
test_set_list=dark_test_list+bright_test_list

# adjust sorting method numerically (test_dark and test_bright)
# cc=[]
# c=[]
# vh=[]
# v=[]
# h=[]
# o=[]
# noise=[]
# for i in range(len(dark_test_list)):

    # if 'r1' in dark_test_list[i] or 'r2' in dark_test_list[i] or 'r3' in dark_test_list[i] or 'r4' in dark_test_list[i]:
    #     noise.append(dark_test_list[i])
    # elif dark_test_list[i][0:2]=='cc':
    #     cc.append(dark_test_list[i])
    # elif dark_test_list[i][0]=='c':
    #     c.append(dark_test_list[i])
    # if dark_test_list[i][0:2]=='vh':
    #     vh.append(dark_test_list[i])
    # elif dark_test_list[i][0]=='v':
    #     v.append(dark_test_list[i])
    # elif dark_test_list[i][0]=='h':
    #     h.append(dark_test_list[i])
    # elif dark_test_list[i][0]=='o':
    #     o.append(dark_test_list[i])

# cc.sort(key=lambda x:int(x[12:-4]))
# c.sort(key=lambda x:int(x[11:-4]))
# vh.sort(key=lambda x:int(x[12:-4]))
# v.sort(key=lambda x:int(x[11:-4]))
# h.sort(key=lambda x:int(x[11:-4]))
# o.sort(key=lambda x:int(x[9:-4]))

# dark_test_list=o+v+h+vh

# cc=[]
# c=[]
# vh=[]
# v=[]
# h=[]
# o=[]
# for i in range(len(bright_test_list)):

    # if bright_test_list[i][0:2]=='cc':
    #     cc.append(bright_test_list[i])
    # elif bright_test_list[i][0]=='c':
    #     c.append(bright_test_list[i])
    # if bright_test_list[i][0:2]=='vh':
    #     vh.append(bright_test_list[i])
    # elif bright_test_list[i][0]=='v':
    #     v.append(bright_test_list[i])
    # elif bright_test_list[i][0]=='h':
    #     h.append(bright_test_list[i])
    # elif bright_test_list[i][0]=='o':
    #     o.append(bright_test_list[i])

# cc.sort(key=lambda x:int(x[12:-4]))
# c.sort(key=lambda x:int(x[11:-4]))
# vh.sort(key=lambda x:int(x[12:-4]))
# v.sort(key=lambda x:int(x[11:-4]))
# h.sort(key=lambda x:int(x[11:-4]))
# o.sort(key=lambda x:int(x[9:-4]))

# bright_test_list=o+v+h+vh

# adjust sorting method numerically (data_train and data_test)
# cc=[]
# c=[]
# vh=[]
# v=[]
# h=[]
# o=[]
# noise=[]
# for i in range(len(test_set_list)):

    # if 'r1' in test_set_list[i] or 'r2' in test_set_list[i] or 'r3' in test_set_list[i] or 'r4' in test_set_list[i]:
    #     noise.append(test_set_list[i])
    # elif test_set_list[i][0:2]=='cc':
    #     cc.append(test_set_list[i])
    # elif test_set_list[i][0]=='c':
    #     c.append(test_set_list[i])
    # if test_set_list[i][0:2]=='vh':
    #     vh.append(test_set_list[i])
    # elif test_set_list[i][0]=='v':
    #     v.append(test_set_list[i])
    # elif test_set_list[i][0]=='h':
    #     h.append(test_set_list[i])
    # else:
    #     o.append(test_set_list[i])

# cc.sort(key=lambda x:int(x[12:-4]))
# c.sort(key=lambda x:int(x[11:-4]))
# vh.sort(key=lambda x:int(x[12:-4]))
# v.sort(key=lambda x:int(x[11:-4]))
# h.sort(key=lambda x:int(x[11:-4]))
# o.sort(key=lambda x:int(x[9:-4]))

# test_set_list=o+v+h+vh

# cc=[]
# c=[]
# vh=[]
# v=[]
# h=[]
# o=[]
# noise=[]
# for i in range(len(train_set_list)):

    # if 'r1' in train_set_list[i] or 'r2' in train_set_list[i] or 'r3' in train_set_list[i] or 'r4' in train_set_list[i]:
    #     noise.append(train_set_list[i])
    # elif train_set_list[i][0:2]=='cc':
    #     cc.append(train_set_list[i])
    # elif train_set_list[i][0]=='c':
    #     c.append(train_set_list[i])
    # if train_set_list[i][0:2]=='vh':
    #     vh.append(train_set_list[i])
    # elif train_set_list[i][0]=='v':
    #     v.append(train_set_list[i])
    # elif train_set_list[i][0]=='h':
    #     h.append(train_set_list[i])
    # else:
    #     o.append(train_set_list[i])
    # else:
    #     noise.append(train_set_list[i])

# cc.sort(key=lambda x:int(x[12:-4]))
# c.sort(key=lambda x:int(x[11:-4]))
# vh.sort(key=lambda x:int(x[12:-4]))
# v.sort(key=lambda x:int(x[11:-4]))
# h.sort(key=lambda x:int(x[11:-4]))
# o.sort(key=lambda x:int(x[9:-4]))

# train_set_list=o+v+h+vh

# create data_train.txt
n=1
while True:
    if not os.path.exists(f'depth_dataset/json_to_image_result/data_train{n}.txt'):
        break
    n+=1
    
with open(f'depth_dataset/json_to_image_result/data_train{n}.txt','w') as f:
    for i in range(len(train_set_list)):
        f.write(f'{train_set_list[i]}\n')

# create data_test.txt
n=1
while True:
    if not os.path.exists(f'depth_dataset/json_to_image_result/data_test{n}.txt'):
        break
    n+=1
    
with open(f'depth_dataset/json_to_image_result/data_test{n}.txt','w') as f:
    for i in range(len(test_set_list)):
        f.write(f'{test_set_list[i]}\n')

# create test_dark.txt
n=1
while True:
    if not os.path.exists(f'depth_dataset/json_to_image_result/test_dark{n}.txt'):
        break
    n+=1
    
with open(f'depth_dataset/json_to_image_result/test_dark{n}.txt','w') as f:
    for i in range(len(dark_test_list)):
        f.write(f'{dark_test_list[i]}\n')

# create test_bright.txt
n=1
while True:
    if not os.path.exists(f'depth_dataset/json_to_image_result/test_bright{n}.txt'):
        break
    n+=1
    
with open(f'depth_dataset/json_to_image_result/test_bright{n}.txt','w') as f:
    for i in range(len(bright_test_list)):
        f.write(f'{bright_test_list[i]}\n')