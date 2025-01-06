from torch.utils.data import Dataset
import cv2
import os
import numpy as np
import torch
import imageio

original_img_dir='depth_dataset/crop_images/original_images'
depth_img_dir='depth_dataset/crop_images/depth_images'
label_img_dir='depth_dataset/json_to_image_result/SegmentationClassPNG'
heat_img_dir='depth_dataset/crop_images/heat_images'

new_heat_img_dir='depth_dataset/crop_images/new_heat_images'
noise_heat_img_path='depth_dataset/crop_images/noise_heat_images'
noise_heat_label_img_path='depth_dataset/crop_images/noise_heat_images_label'

run_time=1
while True:
    if (not os.path.exists(f'depth_dataset/json_to_image_result/data_test{run_time}.txt')) and (not os.path.exists(f'depth_dataset/json_to_image_result/data_train{run_time}.txt')):
        if run_time>1:
            run_time-=1
        break
    run_time+=1

train_data_txt=f'depth_dataset/json_to_image_result/data_train{run_time}.txt'
test_all_data_txt=f'depth_dataset/json_to_image_result/data_test{run_time}.txt'

test_bright_data_txt=f'depth_dataset/json_to_image_result/test_bright{run_time}.txt'
test_dark_data_txt=f'depth_dataset/json_to_image_result/test_dark{run_time}.txt'

new_train_list='depth_dataset/json_to_image_result/data_train_noise_heat1.txt'

class DepthDataset(Dataset):
    def __init__(self,type,mode,n_class):
        self.type=type
        if type=='original':
            self.img_dir=original_img_dir
        elif type=='depth':
            self.img_dir=depth_img_dir
        elif type=='heat' or 'noise_heat':
            self.img_dir=heat_img_dir
        elif type=='new_heat':
            self.img_dir=new_heat_img_dir

        if type=='noise_heat' and mode=='train':
            file=open(new_train_list,'r')
        elif mode=='train':
            file=open(train_data_txt,'r')
        elif mode=='test_all':
            file=open(test_all_data_txt,'r')
        elif mode=='test_bright':
            file=open(test_bright_data_txt,'r')
        elif mode=='test_dark':
            file=open(test_dark_data_txt,'r')
        self.data=file.readlines()  

        self.n_class=n_class

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        # read oringinal/depth image and label image
        if self.type=='original':
            img=cv2.imread(os.path.join(self.img_dir,self.data[idx]).replace('\\','/').replace('\n',''))                                                    #BGR and numpy array
        elif self.type=='depth' or self.type=='heat':
            img=cv2.imread(os.path.join(self.img_dir,self.data[idx]).replace('\\','/').replace('\n','').replace('original',f'{self.type}'),cv2.IMREAD_GRAYSCALE)   #change to gray scale image
        elif self.type=='new_heat':
            img=cv2.imread(os.path.join(self.img_dir,self.data[idx]).replace('\\','/').replace('\n','').replace('original','heat'))
        elif self.type=='noise_heat':
            img=cv2.imread(os.path.join(self.img_dir,self.data[idx]).replace('\\','/').replace('\n','').replace('original','heat'),cv2.IMREAD_GRAYSCALE)
        label_img_raw=cv2.imread(os.path.join(label_img_dir,self.data[idx]).replace('\\','/').replace('.jpg\n','.png'))                                     #BGR and numpy array

        # create a 1 channel label image
        label_img_raw=label_img_raw.transpose((2,0,1))      #shape=(3,400,400)
        label_img=np.zeros((400,400))
        label_img=(label_img_raw[2]>label_img).astype(int)    #shape=(400,400)

        # pre-process of original image
        if self.type=='original':
            img=np.transpose(img,(2,0,1))/255.
            img[0]=(img[0]-0.485)/0.229
            img[1]=(img[1]-0.456)/0.224
            img[2]=(img[2]-0.406)/0.225
        elif self.type=='depth' or self.type=='heat' or self.type=='noise_heat':
            img=np.expand_dims(img,0)
        elif self.type=='new_heat':
            img=np.transpose(img,(2,0,1))

        # convert to tensor
        img=torch.from_numpy(img.copy()).float()
        label_img=torch.from_numpy(label_img.copy()).int()

        # create target image
        (h,w)=label_img.shape
        target_img=torch.zeros(self.n_class,h,w)
        for n in range(self.n_class):
            target_img[n][label_img==n]=1

        self.data[idx]=self.data[idx].replace('noise1_','').replace('noise2_','').replace('noise3_','')

        # depth images with gray scale
        gray_depth_img=cv2.imread(os.path.join(depth_img_dir,self.data[idx]).replace('\\','/').replace('\n','').replace('original','depth'),cv2.IMREAD_GRAYSCALE)
        gray_depth_img=np.expand_dims(gray_depth_img,0)
        gray_depth_img=torch.from_numpy(gray_depth_img.copy()).float()

        #heat images with gray scale
        gray_heat_img=cv2.imread(os.path.join(heat_img_dir,self.data[idx]).replace('\\','/').replace('\n','').replace('original','heat'),cv2.IMREAD_GRAYSCALE)
        gray_heat_img=np.expand_dims(gray_heat_img,0)
        gray_heat_img=torch.from_numpy(gray_heat_img.copy()).float()

        new_heat_img=cv2.imread(os.path.join(new_heat_img_dir,self.data[idx]).replace('\\','/').replace('\n','').replace('original','heat'))
        new_heat_img=np.expand_dims(new_heat_img,0)
        new_heat_img=torch.from_numpy(new_heat_img.copy()).float()

        # wrap original image, target image, label image and gray image, gray image is for model_voting_performance.py
        img_batch={'original':img,'target':target_img,'label':label_img,'gray_depth':gray_depth_img,'gray_heat':gray_heat_img,'new_heat_img':new_heat_img}

        return img_batch







# label_img_o1=cv2.imread('depth_dataset/json_to_image_result/SegmentationClassPNG/original_107.png')
# cv2.imshow('a',label_img_o1)
# cv2.waitKey(0)

# label_img_o1=label_img_o1.transpose((2,0,1))      #shape=(3,400,400)
# label_img=np.zeros((400,400))
# label_img=(label_img_o1[2]>label_img).astype(int)    #shape=(400,400)
# print(sum(label_img))

# target_img=torch.zeros(2,400,400)
# for n in range(2):
#     target_img[n][label_img==n]=1

# blank=np.zeros((3,400,400))
# blank[2]=target_img[1]
# for i in range(400):
#     for j in range(400):
#         if blank[2][i][j]>0:
#             blank[2][i][j]=255
# blank=blank.transpose((1,2,0))
# cv2.imshow('a',blank)
# cv2.waitKey(0)
