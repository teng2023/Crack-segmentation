import torch
import torch.nn as nn

from unet_right_half import UNet_Right_Half,FCNs_concat,FCNs_add
from resnet_18 import ResNet18
from resnet18_shuffle import ShuffleResNet18
from resnet18_mobilenet import MobileResNet18
from resnet18_squeeze import SqueezeResNet18,SqueezeSCResNet18,SqueezeSEResNet18,SqueezeSESCResNet18,SqueezeSESCResNet18_depth
from resnet18_ghost import GhostResNet18
from resnet18_squeeze_next import SqueezeNextResNet18

from thop import profile
from util import get_param

init_weight=True

class Res18UNet(nn.Module):
    def __init__(self,num_class,init_weight_type):
        super().__init__()
        self.resnet=ResNet18(be_backbone=True,pretrain=True,init_weight_type=init_weight_type)
        self.unet=UNet_Right_Half(num_class=num_class,init_weight=init_weight)

    def forward(self,x):
        x=self.resnet(x)
        x=self.unet(x)

        return x

class Res18ShuffleUNet(nn.Module):
    def __init__(self,num_class,init_weight_type):
        super().__init__()
        self.resshuffle=ShuffleResNet18(be_backbone=True,init_weight_type=init_weight_type)
        self.unet=UNet_Right_Half(num_class=num_class,init_weight=init_weight)
    
    def forward(self,x):
        x=self.resshuffle(x)
        x=self.unet(x)

        return x
    
class Res18MobileUNet(nn.Module):
    def __init__(self,num_class,init_weight_type):
        super().__init__()
        self.resmobile=MobileResNet18(be_backbone=True,init_weight_type=init_weight_type)
        self.unet=UNet_Right_Half(num_class=num_class,init_weight=init_weight)
    
    def forward(self,x):
        x=self.resmobile(x)
        x=self.unet(x)

        return x
    
class Res18SqueezeUNet(nn.Module):
    def __init__(self,num_class,init_weight_type):
        super().__init__()
        self.ressqueeze=SqueezeResNet18(be_backbone=True,init_weight_type=init_weight_type)
        self.unet=UNet_Right_Half(num_class=num_class,init_weight=init_weight)
    
    def forward(self,x):
        x=self.ressqueeze(x)
        x=self.unet(x)

        return x
    
class Res18GhostUNet(nn.Module):
    def __init__(self,num_class,init_weight_type):
        super().__init__()
        self.res_ghost=GhostResNet18(be_backbone=True,init_weight_type=False)        # init_weight = False
        self.unet=UNet_Right_Half(num_class=num_class,init_weight=init_weight)

    def forward(self,x):
        x=self.res_ghost(x)
        x=self.unet(x)

        return x

class Res18MobileFCNs_concat(nn.Module):
    def __init__(self,num_class,init_weight_type):
        super().__init__()
        self.resmobile=MobileResNet18(be_backbone=True,init_weight_type=init_weight_type)
        self.fcns=FCNs_concat(num_class=num_class,init_weight=init_weight)

    def forward(self,x):
        x=self.resmobile(x)
        x=self.fcns(x)

        return x

class Res18MobileFCNs_add(nn.Module):
    def __init__(self,num_class,init_weight_type):
        super().__init__()
        self.resmobile=MobileResNet18(be_backbone=True,init_weight_type=init_weight_type)
        self.fcns=FCNs_add(num_class=num_class,init_weight=init_weight)

    def forward(self,x):
        x=self.resmobile(x)
        x=self.fcns(x)

        return x
    
class Res18ShuffleFCNs_concat(nn.Module):
    def __init__(self,num_class,init_weight_type):
        super().__init__()
        self.resshuffle=ShuffleResNet18(be_backbone=True,init_weight_type=init_weight_type)
        self.fcns=FCNs_concat(num_class=num_class,init_weight=init_weight)
    
    def forward(self,x):
        x=self.resshuffle(x)
        x=self.fcns(x)

        return x


class Res18ShuffleFCNs_add(nn.Module):
    def __init__(self,num_class,init_weight_type):
        super().__init__()
        self.resshuffle=ShuffleResNet18(be_backbone=True,init_weight_type=init_weight_type)
        self.fcns=FCNs_add(num_class=num_class,init_weight=init_weight)
    
    def forward(self,x):
        x=self.resshuffle(x)
        x=self.fcns(x)

        return x
    
class Res18SqueezeFCNs_add(nn.Module):
    def __init__(self,num_class,init_weight_type):
        super().__init__()
        self.ressqueeze=SqueezeResNet18(be_backbone=True,init_weight_type=init_weight_type)
        self.fcns=FCNs_add(num_class=num_class,init_weight=init_weight)
    
    def forward(self,x):
        x=self.ressqueeze(x)
        x=self.fcns(x)

        return x

class Res18SqueezeSCUNet(nn.Module):
    def __init__(self,num_class,init_weight_type):
        super().__init__()
        self.ressqueezesc=SqueezeSCResNet18(be_backbone=True,init_weight_type=init_weight_type)
        self.unet=UNet_Right_Half(num_class=num_class,init_weight=init_weight)
    
    def forward(self,x):
        x=self.ressqueezesc(x)
        x=self.unet(x)

        return x
    
class Res18SqueezeSEUNet(nn.Module):
    def __init__(self,num_class,init_weight_type):
        super().__init__()
        self.ressqueezese=SqueezeSEResNet18(be_backbone=True,init_weight_type=init_weight_type)
        self.unet=UNet_Right_Half(num_class=num_class,init_weight=init_weight)
    
    def forward(self,x):
        x=self.ressqueezese(x)
        x=self.unet(x)

        return x
    
class Res18SqueezeSESCUNet(nn.Module):  # 3 channel image
    def __init__(self,num_class,init_weight_type):
        super().__init__()
        self.ressqueezesesc=SqueezeSESCResNet18(be_backbone=True,init_weight_type=init_weight_type)
        self.unet=UNet_Right_Half(num_class=num_class,init_weight=init_weight)
    
    def forward(self,x):
        x=self.ressqueezesesc(x)
        x=self.unet(x)

        return x
    
class Res18SqueezeSESCUNet_depth(nn.Module):  # 1 channel image
    def __init__(self,num_class,init_weight_type):
        super().__init__()
        self.ressqueezesesc=SqueezeSESCResNet18_depth(be_backbone=True,init_weight_type=init_weight_type)
        self.unet=UNet_Right_Half(num_class=num_class,init_weight=init_weight)
    
    def forward(self,x):
        x=self.ressqueezesesc(x)
        x=self.unet(x)

        return x
    
class Res18SqueezeNextUNet(nn.Module):
    def __init__(self,num_class,init_weight_type):
        super().__init__()
        self.ressqueezenext=SqueezeNextResNet18(be_backbone=True,init_weight_type=init_weight_type)
        self.unet=UNet_Right_Half(num_class=num_class,init_weight=init_weight)

    def forward(self,x):
        x=self.ressqueezenext(x)
        x=self.unet(x)

        return x

if __name__=='__main__':    
    # a=ResNet18(be_backbone=True,pretrain=True)
    # print(a.named_children)

    # import torchvision
    # print(torchvision.models.resnet18())
    # b=torchvision.models.resnet18()

    data=torch.randn(1,3,400,400)
    init_weight_type='xavier'

####################################################################################
# UNet for upsampling
    print('############### UNet for upsampling ###############\n')

    model1=Res18UNet(num_class=2,init_weight_type=init_weight_type)
    output=model1(data)
    print(output.shape)
    print('ResNet + UNet:')
    print(get_param(model1))
    print()
    flops,params=profile(model1,inputs=(data,))
    print(f'FLOPs = {flops/1000**3}G')
    print(f'Parameters = {params}')

    model2=Res18ShuffleUNet(num_class=2,init_weight_type=init_weight_type)
    output=model2(data)
    print(output.shape)
    print('ResNet(Shuffle) + UNet:')
    print(get_param(model2))
    print()

    model3=Res18MobileUNet(num_class=2,init_weight_type=init_weight_type)
    output=model3(data)
    print(output.shape)
    print('ResNet(Mobile) + UNet:')
    print(get_param(model3))
    print()

    model4=Res18SqueezeUNet(num_class=2,init_weight_type=init_weight_type)
    output=model4(data)
    print(output.shape)
    print('ResNet(Squeeze) + UNet:')
    print(get_param(model4))
    print()
    flops,params=profile(model4,inputs=(data,))
    print(f'FLOPs = {flops/1000**3}G')
    print(f'Parameters = {params}')

    model5=Res18GhostUNet(num_class=2,init_weight_type=init_weight_type)
    output=model5(data)
    print(output.shape)
    print('ResNet(Ghost) + UNet:')
    print(get_param(model5))
    print()

    model11=Res18SqueezeSCUNet(num_class=2,init_weight_type=init_weight_type)
    output=model11(data)
    print(output.shape)
    print('ResNet(Squeeze with skip connection) + UNet:')
    print(get_param(model11))
    print()
    flops,params=profile(model11,inputs=(data,))
    print(f'FLOPs = {flops/1000**3}G')
    print(f'Parameters = {params}')

    model12=Res18SqueezeSEUNet(num_class=2,init_weight_type=init_weight_type)
    output=model12(data)
    print(output.shape)
    print('ResNet(Squeeze with squeeze and excitation) + UNet:')
    print(get_param(model12))
    print()
    flops,params=profile(model12,inputs=(data,))
    print(f'FLOPs = {flops/1000**3}G')
    print(f'Parameters = {params}')

    model13=Res18SqueezeSESCUNet(num_class=2,init_weight_type=init_weight_type)
    output=model13(data)
    print(output.shape)
    print('ResNet(Squeeze with squeeze and excitation and skip connection) + UNet:')
    print(get_param(model13))
    print()
    flops,params=profile(model13,inputs=(data,))
    print(f'FLOPs = {flops/1000**3}G')
    print(f'Parameters = {params}')

    model14=Res18SqueezeNextUNet(num_class=2,init_weight_type=init_weight_type)
    output=model14(data)
    print(output.shape)
    print('ResNet(SqueezeNext with squeeze and excitation) + UNet:')
    print(get_param(model14))
    print()
    flops,params=profile(model14,inputs=(data,))
    print(f'FLOPs = {flops/1000**3}G')
    print(f'Parameters = {params}')

####################################################################################
# FCNs with concat for upsampling
    print('############### FCNs with concat for upsampling ###############\n')

    model6=Res18MobileFCNs_concat(num_class=2,init_weight_type=init_weight_type)
    output=model6(data)
    print(output.shape)
    print('ResNet(Mobile) + FCNs with concat:')
    print(get_param(model6))
    print()

    model8=Res18ShuffleFCNs_concat(num_class=2,init_weight_type=init_weight_type)
    output=model8(data)
    print(output.shape)
    print('ResNet(Shuffle) + FCNs with concat:')
    print(get_param(model8))
    print()

####################################################################################
# FCNs with add for upsampling
    print('############### FCNs with add for upsampling ###############\n')
 
    model7=Res18MobileFCNs_add(num_class=2,init_weight_type=init_weight_type)
    output=model7(data)
    print(output.shape)
    print('ResNet(Mobile) + FCNs with add:')
    print(get_param(model7))
    print()

    model9=Res18ShuffleFCNs_add(num_class=2,init_weight_type=init_weight_type)
    output=model9(data)
    print(output.shape)
    print('ResNet(Shuffle) + FCNs with add:')
    print(get_param(model9))
    print()

    model10=Res18SqueezeFCNs_add(num_class=2,init_weight_type=init_weight_type)
    output=model10(data)
    print(output.shape)
    print('ResNet(Squeeze) + FCNs with add:')
    print(get_param(model10))
    print()