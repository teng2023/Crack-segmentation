import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import resnet18
import math

class BasicBlock(nn.Module):   #two convolutions
    def __init__(self,in_channel,first_block,downsampling,init_weight_type):
        super().__init__()
        self.in_channel=in_channel
        self.downsampling=downsampling
        self.init_weight_type=init_weight_type
        
        if first_block:             #every first block in the layer needs to double the channels
            self.out_channel=in_channel*2
        else:
            self.out_channel=in_channel
        
        if downsampling:            #every layers' first block, the stride of the convlution is 2, except for the first layer 
            self.stride=2
        else:
            self.stride=1
        
        self.conv1=nn.Conv2d(self.in_channel,self.out_channel,kernel_size=3,stride=self.stride,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(self.out_channel)
        self.relu=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(self.out_channel,self.out_channel,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(self.out_channel)
        if self.downsampling:
            self.downsample=nn.Sequential(nn.Conv2d(self.in_channel,self.out_channel,kernel_size=1,stride=2,bias=False),nn.BatchNorm2d(self.out_channel))

        if init_weight_type:
            self.initialize_weight()

    def forward(self,x):
        skip_connection=x
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.conv2(x)
        self.bn2(x)
        if self.downsampling:
            skip_connection=self.downsample(skip_connection)
            x=x+skip_connection
        
        return x
    
    def initialize_weight(self):
        if self.init_weight_type=='xavier':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        elif self.init_weight_type=='kaiming':
            for layer in self.modules():
                for module in layer:
                    if isinstance(module,nn.BatchNorm2d):
                        nn.init.constant_(module.weight,1)
                        nn.init.constant_(module.bias,0)
                    elif isinstance(module,nn.ConvTranspose2d):
                        nn.init.kaiming_normal_(module.weight,mode='fan_out',nonlinearity='relu')
                        if module.bias is not None:
                            nn.init.constant_(module.bias,0)
class ResNet18(nn.Module):
    def __init__(self,be_backbone,pretrain,init_weight_type):
        super().__init__()
        self.be_backbone=be_backbone
        self.init_weight_type=init_weight_type

        self.conv1=nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU(inplace=True)
        self.maxpooling=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        
        self.layer1=nn.Sequential(BasicBlock(in_channel=64,first_block=False,downsampling=False,init_weight_type=init_weight_type),BasicBlock(in_channel=64,first_block=False,downsampling=False,init_weight_type=init_weight_type))
        self.layer2=nn.Sequential(BasicBlock(in_channel=64,first_block=True,downsampling=True,init_weight_type=init_weight_type),BasicBlock(in_channel=128,first_block=False,downsampling=False,init_weight_type=init_weight_type))
        self.layer3=nn.Sequential(BasicBlock(in_channel=128,first_block=True,downsampling=True,init_weight_type=init_weight_type),BasicBlock(in_channel=256,first_block=False,downsampling=False,init_weight_type=init_weight_type))
        self.layer4=nn.Sequential(BasicBlock(in_channel=256,first_block=True,downsampling=True,init_weight_type=init_weight_type),BasicBlock(in_channel=512,first_block=False,downsampling=False,init_weight_type=init_weight_type))

        self.avgpooling=nn.AdaptiveAvgPool2d(output_size=1)
        self.fc=nn.Linear(512,1000)

        if pretrain:
            self.load_state_dict(resnet18(weights='DEFAULT').state_dict())

        if init_weight_type:
            self.initialize_weight()
    
    def forward(self,x):
        resnet_output=[]
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        if self.be_backbone:
            resnet_output.append(x)
        
        x=self.maxpooling(x)

        x=self.layer1(x)
        if self.be_backbone:
            resnet_output.append(x)

        x=self.layer2(x)
        if self.be_backbone:
            resnet_output.append(x)

        x=self.layer3(x)
        if self.be_backbone:
            resnet_output.append(x)

        x=self.layer4(x)
        resnet_output.append(x)

        if not self.be_backbone:
            x=self.avgpooling(x)
            x=x.view(-1,512)
            x=self.fc(x)

        return resnet_output
    
    def initialize_weight(self):
        if self.init_weight_type=='xavier':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        elif self.init_weight_type=='kaiming':
            for layer in self.modules():
                for module in layer:
                    if isinstance(module,nn.BatchNorm2d):
                        nn.init.constant_(module.weight,1)
                        nn.init.constant_(module.bias,0)
                    elif isinstance(module,nn.ConvTranspose2d):
                        nn.init.kaiming_normal_(module.weight,mode='fan_out',nonlinearity='relu')
                        if module.bias is not None:
                            nn.init.constant_(module.bias,0)