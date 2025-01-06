if '__file__' in globals(): 
    import sys,os
    sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

import torch
import torch.nn as nn
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np

from model_rebuild import unt_srdefnet18

from thop import profile

#import downsampling model
from resnet_18 import ResNet18
from resnet18_shuffle import ShuffleResNet18
from resnet18_mobilenet import MobileResNet18
from resnet18_squeeze import SqueezeResNet18,SqueezeSCResNet18,SqueezeSEResNet18,SqueezeSESCResNet18
from resnet18_ghost import GhostResNet18
from resnet18_squeeze_next import SqueezeNextResNet18

from unet_model import UNet

#import upsampling model
from unet_right_half import UNet_Right_Half,FCNs_concat,FCNs_add

#import crack information
from boat.utils.dataset import crack_semantic
import boat.utils.augment as augment

from training import iou,pixel_accuracy

num_class=2
init_weight=True
init_weight_type='xavier'
be_backbone=True

downsample_model={'resnet18':ResNet18(be_backbone=be_backbone,pretrain=True,init_weight_type=init_weight_type),
                  'shuffle_resnet18':ShuffleResNet18(be_backbone=True,init_weight_type=init_weight_type),
                  'mobile_resnet18':MobileResNet18(be_backbone=True,init_weight_type=init_weight_type),
                  'squeeze_resnet18':SqueezeResNet18(be_backbone=True,init_weight_type=init_weight_type),
                  'ghost_resnet18':GhostResNet18(be_backbone=True,init_weight_type=init_weight_type),
                  'squeezenext_resnet18':SqueezeNextResNet18(be_backbone=True,init_weight_type=init_weight_type),
                  'squeeze_resnet18_SC':SqueezeSCResNet18(be_backbone=True,init_weight_type=init_weight_type),
                  'squeeze_resnet18_SE':SqueezeSEResNet18(be_backbone=True,init_weight_type=init_weight_type),
                  'squeeze_resnet18_SC_SE':SqueezeSESCResNet18(be_backbone=True,init_weight_type=init_weight_type)
                  }
upsample_model={'unet':UNet_Right_Half(num_class=num_class,init_weight=init_weight),
                'fcn_concat':FCNs_concat(num_class=num_class,init_weight=init_weight),
                'fcn_add':FCNs_add(num_class=num_class,init_weight=init_weight)
                }

class CrackModel(nn.Module):
    def __init__(self,downsample,upsample):
        super().__init__()
        self.downsampling=downsample_model[downsample]
        self.upsampling=upsample_model[upsample]

    def forward(self,x):
        x=self.downsampling(x)
        x=self.upsampling(x)

        return x

    def count_parameter(self,input_shape=(1,3,400,400)):
        input_data=torch.randn(input_shape)
        
        return profile(model=self,inputs=(input_data,))
    
def frame_per_sec(model):
    #GPU problem
    use_gpu=torch.cuda.is_available()
    device=torch.device("cuda:0" if use_gpu else "cpu")

    ###################################################################################################################
    #crack conditions
    colormap=[[0, 0, 0], [0, 0, 255]]#rgb
    catmap=['background','crack']
    num_class=len(catmap)

    test_transforms=[augment.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225),dataset_cat='semantic')]

    cattable = torch.zeros(256**3)
    for i, rgb in enumerate(colormap):
        cattable[(rgb[0] * 256 + rgb[1]) * 256 + rgb[2]] = i

    ###################################################################################################################
    #loading one image
    # img_file='boat/database/v_crack/8751crack_400400_50/aaa.txt'
    # test_data=crack_semantic(img_file,cattable,transform=augment.Compose(test_transforms,dataset_cat ='semantic'))
    # img_loader=DataLoader(test_data,batch_size=1,shuffle=False,num_workers=2)
    # load_img=tqdm(tqdm(img_loader,total=len(img_loader),leave=True,desc='Test[1/1]'))

    #loading the test set
    test_file='boat/database/v_crack/8751crack_400400_50/test1.txt'
    test_data=crack_semantic(test_file,cattable,transform=augment.Compose(test_transforms,dataset_cat='semantic'))
    test_loader=DataLoader(test_data,batch_size=1,shuffle=False,num_workers=2)
    load_img=tqdm(test_loader,total=len(test_loader),leave=True,desc='Test[1/1]')

    # if use_gpu:
    #     model.to(device)
    
    total_ious=[]
    pixel_accs=[]
    total_time_list=[]

    with torch.no_grad():
        model.eval()
        for img,mask in load_img:
            # img=img.to(device)
            # mask=mask.to(device)

            start_time=time.time()
            output=model(img) 
            end_time=time.time()

            N,_,h,w=output.shape
            pred=output.permute(0,2,3,1).reshape(-1,num_class).argmax(dim=1).reshape(N,h,w)

            target=mask.argmax(dim=1).reshape(N,h,w) #.to(device)

            total_ious.append(iou(pred,target,N))
            pixel_accs.extend(pixel_accuracy(pred,target,N))

            ious=torch.tensor(total_ious,device=device).transpose(0,1).mean(dim=1)
            pixel_accu=torch.tensor(pixel_accs).mean()
            meaniou=torch.nanmean(ious)
            
            total_time_list.append(end_time-start_time)
        
        total_time=sum(total_time_list)
        single_time=total_time/len(test_loader)
        fps=1/single_time
    
    return meaniou,pixel_accu,ious[0],ious[1],total_time,fps

def load_model(model=None,model_path=None):
    weight=torch.load(model_path)
    model.load_state_dict(weight['model'],strict=True)

if __name__ == '__main__':
        
    model_list=['shuffle_resnet18','mobile_resnet18','ghost_resnet18','squeezenext_resnet18']

    # for model_name in model_list:

    old_weight='crack_model_test/13_100_9921_9921.pth'
    model=unt_srdefnet18(num_classes=num_class,pretrained_own=old_weight)
    # model_name='resnet18'
    # model=CrackModel(downsample=model_name,upsample='unet')
    # model=UNet(n_channels=3,n_classes=2,bilinear=True)
    # model_path='crack_model_test/check points/resnet18+squeeze_se_sc+unet3/Epoch196_of_250.pth'
    # load_model(model,model_path)

    # flops,params=model.count_parameter()
    # print(f'FLOPs = {flops/1000**3}G')
    # print(f'parameters = {params/1000**2}M')

    fps_list=[]
    for i in range(10):
        meaniou,pixel_accu,background_iou,crack_iou,total_time,fps=frame_per_sec(model)
        fps_list.append(fps)

    # print(model_name)
    print(f'average of FPS:{np.mean(fps_list)}')
    print(f'standarad deviation of FPS:{np.std(fps_list)}')

    # print(f'mean iou = {meaniou}')
    # print(f'pixel accuracy = {pixel_accu}')
    # print(f'background iou = {background_iou}')
    # print(f'crack iou = {crack_iou}')
    # print(f'total time = {total_time}')
    # print(f'fps = {fps}')
