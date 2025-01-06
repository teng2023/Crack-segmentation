import torch
import cv2
import numpy as np
from model import Res18SqueezeSESCUNet
import os

def iou(pred,target,len_batch):
    ious=[]
    background_iou=[]
    crack_iou=[]
    for i in range(len_batch):
        for cls in range(2):
            pred_index=pred==cls
            target_index=target==cls
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

test_txt=open('boat\database/v_crack/8751crack_400400_50/test1.txt','r')
test_txt=test_txt.readlines()
high_iou_txt=open('crack_model_test/high_iou_crackseg9k.txt','r')
high_iou_txt=high_iou_txt.readlines()
selected_img_txt=open('crack_model_test/selected_img.txt','r')
selected_img_txt=selected_img_txt.readlines()

img_path='boat/database/v_crack/8751crack_400400_50/images/'
label_path='boat/database/v_crack/8751crack_400400_50/colorlabels/'

model=Res18SqueezeSESCUNet(num_class=2,init_weight_type='xavier')
weight_path='crack_model_test/check points/resnet18+squeeze_se_sc+unet15/Epoch193_of_250.pth'
weight=torch.load(weight_path)
model.load_state_dict(weight['model'],strict=False)
model.eval()

# for img in test_txt:
# for img in high_iou_txt:
for img in selected_img_txt:
    img_name=img.replace('\n','')

    original_img=cv2.imread(os.path.join(img_path,img_name+'.jpg'))
    label_img=cv2.imread(os.path.join(label_path,img_name+'.png'))

    # pre-processing original image
    original_img_input=np.transpose(original_img,(2,0,1))/255.
    original_img_input[0]=(original_img_input[0]-0.485)/0.229
    original_img_input[1]=(original_img_input[1]-0.456)/0.224
    original_img_input[2]=(original_img_input[2]-0.406)/0.225

    original_img_input=torch.from_numpy(original_img_input.copy()).float()      # convert from numpy to torch
    original_img_input=torch.unsqueeze(original_img_input,0)                    # add channel 'batch'

    label_img_input=label_img.transpose((2,0,1))      #shape=(3,400,400)
    label_img_empty=np.zeros((400,400))
    label_img_input=(label_img_input[0]>label_img_empty).astype(int)    #shape=(400,400)


    output=model(original_img_input)

    seg_img=np.zeros((3,400,400)).astype(dtype='uint8')
    pred=output.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(400,400)
    pred=pred.numpy()

    seg_img[2]=pred
    for i in range(400):
        for j in range(400):
            if seg_img[2][i][j]!=0:
                seg_img[2][i][j]=255

    seg_img=seg_img.transpose((1,2,0))

    iou_number=iou(pred,label_img_input,1)
    print(iou(pred,label_img_input,1))
    # if iou_number[1]>0.8:
    #     with open('crack_model_test/high_iou_crackseg9k.txt','a') as f:
    #         f.write(f'{img_name}\n')
    # if iou_number[1]<0.2:
    #     with open('crack_model_test/low_iou_crackseg9k.txt','a') as f:
    #         f.write(f'{img_name}\n')

    result=np.hstack((label_img,original_img,seg_img))

    cv2.imshow('a',result)
    key=cv2.waitKey(0)
    cv2.destroyAllWindows()

    if key==ord("s"):
        with open('crack_model_test/selected_img.txt','a') as f:
            f.write(f'{img}')
    if key==ord("g"):
        continue
    if key==ord("q"):
        break