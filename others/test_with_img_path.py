import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms

import cv2
import copy
import numpy as np

# from Ucrack_model import ucrack
# from utils.metrics import cal_iou, cal_pixel_accuracy, cal_num_correct
from model import Res18ShuffleUNet,Res18SqueezeUNet,Res18SqueezeSESCUNet,Res18MobileUNet,Res18SqueezeNextUNet,Res18SqueezeSESCUNet_depth
from quantized_model import Res18SqueezeUNet_Quant

shape_catmap = ['N/A', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
num_classes = 2
shapeclass_num = 27
colorlabel = None

bright = 1
lr = 1 #1or-1(-1表翻轉)
ud = 1 #1or-1(-1表顛倒)

# img = cv2.imread(f'database/v_crack/from_network/b.jpg')#網路抓得
imagename = 'a_40_2'
img = cv2.imread(f'database/v_crack/8751crack_400400_50/images/{imagename}.jpg')#公開資料集
colorlabel = cv2.imread(f'database/v_crack/8751crack_400400_50/colorlabels/{imagename}.png').astype('int32')#公開資料集

# imagename = 'CFD_098'
# img=cv2.imread(f'boat/database/v_crack/8751crack_400400_50/images/{imagename}.jpg')
# # img = cv2.imread(f'database/v_crack/self_catch2/images/{imagename}.jpg')#自建資料集
# colorlabel = cv2.imread(f'boat/database/v_crack/8751crack_400400_50/colorlabels/{imagename}.png').astype('int32')#自建資料集


#翻轉圖片
img[::1, ::1] = img[::ud, ::lr]
#裁切圖片
img = cv2.resize(img,(400,400))
# img = img[:400, :400]

o = copy.deepcopy(img)/255*bright
img[:, :, ::1]=img[:, :, ::-1] #bgr->rgb

toten = torchvision.transforms.ToTensor()
nor = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)

img = toten(img)*bright
# img = nor(img)
img = torch.unsqueeze(img, dim=0)

cal_iou_ = False
if colorlabel is not None:
    cal_iou_ = True
cal_iou_=False
#-----------------------a模型是要放目前覺得最好的權重--------------------------
outputs1 = None
outputs2 = None
outputs3 = None

Ucrack_visible_light_segmentation_net_part11 = 'hrsegnetb48'
Ucrack_visible_light_segmentation_net_part21 = 'Ucrack_data/Ucrack_visible_light_segmentation_net_part2/resnet18/local/1th_train/1_39_9881_9868.pth'
Ucrack_infrared_ray_segmentation_net1 = 'Ucrack_data/Ucrack_infrared_ray_segmentation_net/srdefnet18/local/1th_train/1_43_9860_9852.pth'
Ucrack_shape_judgment_net1 = 'Ucrack_data/Ucrack_shape_judgment_net/1th_train/1_4_9489_9242.pth'
path1 = [Ucrack_visible_light_segmentation_net_part11, Ucrack_visible_light_segmentation_net_part21, Ucrack_infrared_ray_segmentation_net1, Ucrack_shape_judgment_net1]
visible_seg_rate1 = [0.2, 0.8]#[part1, part2]
crack_seg_rate1 = [0.9, 0.1]#[visible, infrared]


model_path='crack_model_test/check points/resnet18+squeeze_se_sc+unet3/Epoch196_of_250.pth'
model1 = Res18SqueezeSESCUNet(num_class=2,init_weight_type='xavier')

# model_path='crack_model_test/check points/resnet18+squeeze+unet+quant7/Epoch142_of_250.pth'
# model1 = Res18SqueezeUNet_Quant(num_class=2,init_weight_type='xavier')

# model1=torch.load('resnet_with_squeezenet_and_unet.pth')

# model_path='crack_model_test/check points/resnet18+mobile+unet3/Epoch68_of_200.pth'
# model1 = Res18MobileUNet(num_class=2,init_weight=True)

weight=torch.load(model_path)
model1.load_state_dict(weight['model'],strict=False)

model1.eval()
outputs1=model1(img)

#-----------------------[0]->可見光1,[1]->可見光2,[2]->紅外線,[3]->可見光,[4]->可見光+紅外線,[5]->shape--------------------------
if outputs1 is not None:
    print(o.shape)
    feature_outputs1 = np.zeros( (400, 400, 3))

    _, o_index = torch.max(outputs1,dim=1)
    seg = torch.unsqueeze(o_index, dim=1)
    print(seg.shape)
    seg = seg[0,0].to(dtype=torch.float32)
    seg = np.array(seg)
    feature_outputs1[:,:,2] = seg
    print(seg.shape)
    print(seg)
# if cal_iou_:
#     colorlabel = colorlabel[:400, :400]
#     colorlabel[:, :, ::1]=colorlabel[:, :, ::-1] #bgr->rgb

#     colormap = [[0, 0, 0], [0, 0, 255]]#rgb
#     cattable = torch.zeros(256**3)
#     for i, rgb in enumerate(colormap):
#         cattable[(rgb[0] * 256 + rgb[1]) * 256 + rgb[2]] = i

#     idx = (colorlabel[:, :, 0] * 256 + colorlabel[:, :, 1]) * 256 + colorlabel[:, :, 2]
#     catlabel = cattable[idx]
#     catlabel = catlabel.unsqueeze(0)
#     catnum = (cattable>0).sum()+1

#     if ud==-1:
#         catlabel = torch.flip(catlabel, dims=[1])
#     if lr==-1:
#         catlabel = torch.flip(catlabel, dims=[2])

#     mask = torch.zeros([1, catnum, catlabel.shape[1], catlabel.shape[2]])
#     for i in range(catnum):
#         mask[0, i] = catlabel==i

    # if outputs1:
    #     for i in range(len(feature_outputs1)-1):
    #         if outputs1[i] is not None:
    #             print(f'{figure_name1[i]:60}, pixel_accuracy={cal_pixel_accuracy(outputs1[i], mask)*100:7.5f}%, iou={cal_iou(outputs1[i], mask)}')
    # if outputs2:
    #     for i in range(len(feature_outputs2)-1):
    #         if outputs2[i] is not None:
    #             print(f'{figure_name2[i]:60}, pixel_accuracy={cal_pixel_accuracy(outputs2[i], mask)*100:7.5f}%, iou={cal_iou(outputs2[i], mask)}')
    # if outputs3:
    #     for i in range(len(feature_outputs3)-1):
    #         if outputs3[i] is not None:
    #             print(f'{figure_name3[i]:60}, pixel_accuracy={cal_pixel_accuracy(outputs3[i], mask)*100:7.5f}%, iou={cal_iou(outputs3[i], mask)}')

cv2.imshow('origin',o)
cv2.imshow('l',feature_outputs1)
cv2.waitKey(0)