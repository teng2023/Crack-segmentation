import torch
import cv2
import numpy as np
from model import Res18SqueezeSESCUNet,Res18SqueezeSESCUNet_depth
from model_voting import VotingModel_3_3_3

####################################################################################################
# adjustable parameter

image_type_dict={1:'single',2:'voting333'}
image_type=image_type_dict[1]

image_number=80

# original
path1='crack_model_test/check points(custom)/resnet18+squeeze_se_sc+unet(original)4/Epoch99_of_500.pth'
path2='crack_model_test/check points(custom)/resnet18+squeeze_se_sc+unet(original)2/Epoch166_of_500.pth'
path3='crack_model_test/check points(custom)/resnet18+squeeze_se_sc+unet(original)3/Epoch61_of_500.pth'
# depth
path4='crack_model_test/check points(custom)/resnet18+squeeze_se_sc+unet(depth)2/Epoch145_of_500.pth'
path5='crack_model_test/check points(custom)/resnet18+squeeze_se_sc+unet(depth)3/Epoch69_of_500.pth'
path6='crack_model_test/check points(custom)/resnet18+squeeze_se_sc+unet(depth)4/Epoch176_of_500.pth'
#heat (based on crack IoU)
path7='crack_model_test/check points(custom)/resnet18+squeeze_se_sc+unet(heat)4/Epoch27_of_500.pth'
path8='crack_model_test/check points(custom)/resnet18+squeeze_se_sc+unet(heat)3/Epoch22_of_500.pth'
path9='crack_model_test/check points(custom)/resnet18+squeeze_se_sc+unet(heat)2/Epoch16_of_500.pth'

####################################################################################################
# test_txt=open('crack_model_test/custom_dataset_img_select.txt','r')
# test_txt=test_txt.readlines()

# with open('crack_model_test/custom_dataset_iou.txt','w') as f:
#     for img in test_txt:
#         img=img.replace('\n','')
#         image_number=int(img[9:-4])
# original image
original_img_path=f'depth_dataset/crop_images/original_images/original_{image_number}.jpg'
original_img=cv2.imread(original_img_path)

# depth image, shape=(h,w,c) with BGR
image_path=f'depth_dataset/crop_images/depth_images/depth_{image_number}.jpg'
depth_img=cv2.imread(image_path)                                            # 3 channel image
depth_img_gray=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)                  # 1 channel iamge (grayscale)
depth_img_gray_3d=cv2.cvtColor(depth_img_gray,cv2.COLOR_GRAY2BGR)


# heat image
heat_img_path=f'depth_dataset/crop_images/heat_images/heat_{image_number}.jpg'
heat_img=cv2.imread(heat_img_path)                                            # 3 channel image
heat_img_gray=cv2.imread(heat_img_path,cv2.IMREAD_GRAYSCALE)                  # 1 channel iamge (grayscale)
heat_img_gray_3d=cv2.cvtColor(heat_img_gray,cv2.COLOR_GRAY2BGR)

# pre-processing input images
depth_img_gray_input=np.expand_dims(depth_img_gray,0)
depth_img_gray_input=torch.from_numpy(depth_img_gray_input.copy()).float()    # convert from numpy to torch
depth_img_gray_input=torch.unsqueeze(depth_img_gray_input,0)                  # add channel 'batch'

heat_img_gray_input=np.expand_dims(heat_img_gray,0)
heat_img_gray_input=torch.from_numpy(heat_img_gray_input.copy()).float()    # convert from numpy to torch
heat_img_gray_input=torch.unsqueeze(heat_img_gray_input,0)                  # add channel 'batch'

# original_img_input=np.transpose(original_img,(2,0,1))

original_img_input=np.transpose(original_img,(2,0,1))/255.
original_img_input[0]=(original_img_input[0]-0.485)/0.229
original_img_input[1]=(original_img_input[1]-0.456)/0.224
original_img_input[2]=(original_img_input[2]-0.406)/0.225

original_img_input=torch.from_numpy(original_img_input.copy()).float()      # convert from numpy to torch
original_img_input=torch.unsqueeze(original_img_input,0)                    # add channel 'batch'


# create a empty array to fill with the output if exsits
seg_img_o1=np.zeros((3,400,400)).astype(dtype='uint8')
seg_img_d1=np.zeros((3,400,400)).astype(dtype='uint8')
seg_img_h1=np.zeros((3,400,400)).astype(dtype='uint8')
seg_img_vote=np.zeros((3,400,400)).astype(dtype='uint8')

# select model -> send image to model -> post-processing output

if image_type=='single':
    # single original
    model_o1=Res18SqueezeSESCUNet(num_class=2,init_weight_type='xavier')
    weight_path_o1=path1
    weight_o1=torch.load(weight_path_o1)
    model_o1.load_state_dict(weight_o1['model'],strict=False)
    model_o1.eval()
    output_o1=model_o1(original_img_input)

    pred_o1=output_o1.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
    pred_o1=pred_o1.numpy()

    seg_img_o1[2]=pred_o1
    for i in range(400):
        for j in range(400):
            if seg_img_o1[2][i][j]!=0:
                seg_img_o1[2][i][j]=255

    seg_img_o1=seg_img_o1.transpose((1,2,0))

    # single depth
    model_d1=Res18SqueezeSESCUNet_depth(num_class=2,init_weight_type='xavier')
    weight_path_d1=path4
    weight_d1=torch.load(weight_path_d1)
    model_d1.load_state_dict(weight_d1['model'],strict=False)

    output_d1=model_d1(depth_img_gray_input)


    pred_d1=output_d1.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
    pred_d1=pred_d1.numpy()

    seg_img_d1[2]=pred_d1
    for i in range(400):
        for j in range(400):
            if seg_img_d1[2][i][j]!=0:
                seg_img_d1[2][i][j]=255

    seg_img_d1=seg_img_d1.transpose((1,2,0))

    # single heat
    model_h1=Res18SqueezeSESCUNet_depth(num_class=2,init_weight_type='xavier')
    weight_path_h1=path7
    weight_h1=torch.load(weight_path_h1)
    model_h1.load_state_dict(weight_h1['model'],strict=False)

    output_h1=model_h1(heat_img_gray_input)

    pred_h1=output_h1.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
    pred_h1=pred_h1.numpy()

    seg_img_h1[2]=pred_h1
    for i in range(400):
        for j in range(400):
            if seg_img_h1[2][i][j]!=0:
                seg_img_h1[2][i][j]=255

    seg_img_h1=seg_img_h1.transpose((1,2,0))

elif image_type=='voting333':
    model=VotingModel_3_3_3('original','original','original','depth','depth','depth','heat','heat','heat',path1,path2,path3,path4,path5,path6,path7,path8,path9)
    pred_v1=model(original_img_input,depth_img_gray_input,heat_img_gray_input)

    seg_img_vote[2]=pred_v1
    for i in range(400):
        for j in range(400):
            if seg_img_vote[2][i][j]!=0:
                seg_img_vote[2][i][j]=255

    seg_img_vote=seg_img_vote.transpose((1,2,0))

#if output doesn't fill in the empty array, the shape should be change into (h,w,c) because of opencv
if seg_img_o1.shape==(3,400,400):
    seg_img_o1=seg_img_o1.transpose((1,2,0))
if seg_img_d1.shape==(3,400,400):
    seg_img_d1=seg_img_d1.transpose((1,2,0))
if seg_img_h1.shape==(3,400,400):
    seg_img_h1=seg_img_h1.transpose((1,2,0))
if seg_img_vote.shape==(3,400,400):
    seg_img_vote=seg_img_vote.transpose((1,2,0))

# print iou########################################################################################################

# label images
label_img_o1_1=cv2.imread(f'depth_dataset/json_to_image_result/SegmentationClassPNG/original_{image_number}.png')



label_img_o1=label_img_o1_1.transpose((2,0,1))      #shape=(3,400,400)
label_img=np.zeros((1,400,400))
label_img[0]=(label_img_o1[2]>0).astype(int)    #shape=(400,400)


# def iou(pred,target,len_batch):
#     ious=[]
#     background_iou=[]
#     crack_iou=[]
#     for i in range(len_batch):
#         for cls in range(2):
#             pred_index=pred==cls
#             target_index=target==cls
#             intersection=pred_index[target_index].sum()
#             union=pred_index.sum()+target_index.sum()-intersection
#             if union==0:
#                 if (cls+1)%2 != 0:
#                     background_iou.append(1)
#                 else:
#                     crack_iou.append(1)
#             else:
#                 if (cls+1)%2 != 0:
#                     background_iou.append(float(intersection)/union)
#                 else:
#                     crack_iou.append(float(intersection)/union)

#     return ious
def iou(pred,target,len_batch):
    ious=[]
    background_iou=[]
    crack_iou=[]
    for i in range(len_batch):
        for cls in range(2):
            pred_index=pred[i]==cls
            target_index=target[i]==cls
            intersection=pred_index[target_index].sum()
            union=pred_index.sum()+target_index.sum()-intersection
            if union==0:
                if (cls+1)%2 != 0:
                    background_iou.append(1)
                else:
                    crack_iou.append(1)
            else:
                if (cls+1)%2 != 0:
                    background_iou.append(float(intersection)/union)
                else:
                    crack_iou.append(float(intersection)/union)

    ious.append(sum(background_iou)/len(background_iou))
    ious.append(sum(crack_iou)/len(crack_iou))

    return ious
        
        


# a=np.array(([[[1,0,0],[0,1,0],[0,0,1]],[[1,0,0],[0,1,0],[0,0,1]],[[1,0,0],[0,1,0],[0,0,1]]])).astype('bool')
# b=np.ones((3,3,3)).astype('bool')

# print(a[b])
# f.write(f'{img}\t{iou(pred_v1,label_img,1)}\n')
if image_type=='single':
    print(iou(pred_o1,label_img,1))
    print(iou(pred_d1,label_img,1))
    print(iou(pred_h1,label_img,1))
else:
    print(iou(pred_v1,label_img,1))

########################################################################################################

# show the result
# if 'depth'==image_type:
#     result_stack=np.hstack((original_img,depth_img,seg_img_d1))
#     cv2.imshow(f'original image / depth image / {image_type} result',result_stack)
# elif 'original'==image_type:
#     result_stack=np.hstack((original_img,depth_img,label_img_o1_1,seg_img_o1))
#     cv2.imshow(f'original image / depth image / {image_type} result',result_stack)
# elif 'voting_o2_d1'==image_type:
#     result_stack=np.hstack((original_img,depth_img,seg_img_vote))
#     cv2.imshow(f'original image / depth image / {image_type} result',result_stack)
# else:
# if image_type=='single':
#     result1=np.hstack((original_img,depth_img,heat_img))
#     result2=np.hstack((seg_img_o1,seg_img_d1,seg_img_h1))
#     result=np.vstack((result1,result2))
#     cv2.imshow('original image / depth image / voting result',result)
#     cv2.waitKey(0)
# elif image_type=='voting333':

    # result=np.hstack((original_img,label_img_o1_1,seg_img_vote))
    # result=np.hstack((original_img,seg_img_vote))
    # cv2.imshow(f'{image_number}',result)
    # cv2.waitKey(0)
        
        # key=cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # if key==ord("s"):
        #     with open('crack_model_test/custom_dataset_img_select.txt','a') as ff:
        #         ff.write(f'{img}\n')
        # if key==ord("g"):
        #     continue
        # if key==ord("q"):
        #     break

#########################################################################################################
#########################################################################################################
#########################################################################################################
# check the images one by one
# low_iou_txt=open('depth_dataset/json_to_image_result/low_iou_images.txt','r')
# low_iou_txt=low_iou_txt.readlines()

# with open('depth_dataset/json_to_image_result/test_img_iou_2.txt','w') as f:
#     for img in low_iou_txt:
#         img=img.split('\t')

#         ori_image=cv2.imread(f'depth_dataset/crop_images/original_images/{img[0]}'.replace('\n',''))
#         depth_img=cv2.imread(f'depth_dataset/crop_images/depth_images/{img[0]}'.replace('\n','').replace('original','depth'))
#         label_img=cv2.imread(f'depth_dataset/json_to_image_result/SegmentationClassPNG/{img[0]}'.replace('\n','').replace('jpg','png'))

#         original_img_input=np.transpose(ori_image,(2,0,1))/255.
#         original_img_input[0]=(original_img_input[0]-0.485)/0.229
#         original_img_input[1]=(original_img_input[1]-0.456)/0.224
#         original_img_input[2]=(original_img_input[2]-0.406)/0.225

#         original_img_input=torch.from_numpy(original_img_input.copy()).float()      # convert from numpy to torch
#         original_img_input=torch.unsqueeze(original_img_input,0)                    # add channel 'batch'


#         output_o1=model_o1(original_img_input)

#         pred_o1=output_o1.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(400,400)
#         pred_o1=pred_o1.numpy()

#         seg_img_o1=np.zeros((3,400,400)).astype(dtype='uint8')
#         seg_img_o1[2]=pred_o1
#         for i in range(400):
#             for j in range(400):
#                 if seg_img_o1[2][i][j]!=0:
#                     seg_img_o1[2][i][j]=255

#         seg_img_o1=seg_img_o1.transpose((1,2,0))

#         result=np.hstack((ori_image,depth_img,label_img,seg_img_o1))
#         cv2.imshow(f'{img[0]}',result)
#         key=cv2.waitKey(0)
#         cv2.destroyAllWindows()

#         if key==ord("s"):
            
#                 f.write(f'{img[0]}\n')
#         if key==ord("g"):
#             continue
#         if key==ord("q"):
#             break