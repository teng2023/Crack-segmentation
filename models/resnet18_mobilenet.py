import torch
from torch import nn, Tensor
import math
from torchvision.models.mobilenet import mobilenet_v2
from typing import Any, Callable, List, Optional
from torchvision.ops.misc import Conv2dNormActivation

class InvertedResBlock(nn.Module):
    def __init__(
        self, inp: int, oup: int, stride: int, expand_ratio: int, norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(
                Conv2dNormActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6)
            )
        layers.extend(
            [
                # dw
                Conv2dNormActivation(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    norm_layer=norm_layer,
                    activation_layer=nn.ReLU6,
                ),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileResNet18(nn.Module):
    def __init__(self,be_backbone,init_weight_type):
        super().__init__()
        self.be_backbone=be_backbone
        self.init_weight_type=init_weight_type

        self.conv1=nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU(inplace=True)
        self.maxpooling=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        # self.layer1=nn.Sequential(InvertedResBlock(downsample=False,in_channel=64,init_weight=init_weight),InvertedResBlock(downsample=False,in_channel=64,init_weight=init_weight))
        # self.layer2=nn.Sequential(InvertedResBlock(downsample=True,in_channel=64,init_weight=init_weight),InvertedResBlock(downsample=False,in_channel=128,init_weight=init_weight))
        # self.layer3=nn.Sequential(InvertedResBlock(downsample=True,in_channel=128,init_weight=init_weight),InvertedResBlock(downsample=False,in_channel=256,init_weight=init_weight))
        # self.layer4=nn.Sequential(InvertedResBlock(downsample=True,in_channel=256,init_weight=init_weight),InvertedResBlock(downsample=False,in_channel=512,init_weight=init_weight))

        # self.layer1=nn.Sequential(InvertedResBlock(in_ch=64,out_ch=64,stride=1,expand_ratio=6),InvertedResBlock(in_ch=64,out_ch=64,stride=1,expand_ratio=6))
        # self.layer2=nn.Sequential(InvertedResBlock(in_ch=64,out_ch=128,stride=2,expand_ratio=6),InvertedResBlock(in_ch=128,out_ch=128,stride=1,expand_ratio=6))
        # self.layer3=nn.Sequential(InvertedResBlock(in_ch=128,out_ch=256,stride=2,expand_ratio=6),InvertedResBlock(in_ch=256,out_ch=256,stride=1,expand_ratio=6))
        # self.layer4=nn.Sequential(InvertedResBlock(in_ch=256,out_ch=512,stride=2,expand_ratio=6),InvertedResBlock(in_ch=512,out_ch=512,stride=1,expand_ratio=6))

        self.layer1=nn.Sequential(InvertedResBlock(inp=64,oup=64,stride=1,expand_ratio=6),InvertedResBlock(inp=64,oup=64,stride=1,expand_ratio=6))
        self.layer2=nn.Sequential(InvertedResBlock(inp=64,oup=128,stride=2,expand_ratio=6),InvertedResBlock(inp=128,oup=128,stride=1,expand_ratio=6))
        self.layer3=nn.Sequential(InvertedResBlock(inp=128,oup=256,stride=2,expand_ratio=6),InvertedResBlock(inp=256,oup=256,stride=1,expand_ratio=6))
        self.layer4=nn.Sequential(InvertedResBlock(inp=256,oup=512,stride=2,expand_ratio=6),InvertedResBlock(inp=512,oup=512,stride=1,expand_ratio=6))

        # self.avgpooling=nn.AdaptiveAvgPool2d(output_size=1)
        # self.fc=nn.Linear(512,1000)

        if self.init_weight_type:
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
                            
# class InvertedResBlock(nn.Module):
#     def __init__(self,in_ch,out_ch,stride,expand_ratio):
#         super().__init__()
#         self.stride=stride
#         hidden_dim=int(round(in_ch*expand_ratio))
#         self.use_res_connect=self.stride==1 and in_ch==out_ch
#         layers=[]

#         if expand_ratio!=1:
#             layers.extend([nn.Conv2d(in_ch,hidden_dim,kernel_size=1,stride=1,padding=0), #pointwise convolution
#                           nn.BatchNorm2d(hidden_dim),
#                           nn.ReLU6(inplace=True)
#                           ])
#         layers.extend([nn.Conv2d(hidden_dim,hidden_dim,kernel_size=3,stride=1,padding=1,groups=hidden_dim), #depthwise convolution
#                       nn.BatchNorm2d(hidden_dim),
#                       nn.ReLU6(inplace=True),
#                       nn.Conv2d(hidden_dim,out_ch,kernel_size=1,stride=1,padding=0,bias=False), #pointwise linear
#                       nn.BatchNorm2d(out_ch),
#                       ])
        
#         self.conv=nn.Sequential(*layers)
#         self.out_channels=out_ch
#         self._is_cn=stride>1

#     def forward(self,x):
#         if self.use_res_connect:
#             return x+self.conv(x)
#         else:
#             self.conv(x)


# class InvertedResBlock(nn.Module):
#     def __init__(self,downsample,in_channel,init_weight):
#         super().__init__()

#         self.downsample=downsample
#         if self.downsample:
#             self.in_channel=in_channel
#             self.out_channel=in_channel*2
#             stride=2
#         else:
#             self.in_channel=in_channel
#             self.out_channel=self.in_channel
#             stride=1

#         self.conv1=nn.Conv2d(self.in_channel,self.out_channel,kernel_size=1,stride=1,padding=0)
#         self.conv2=nn.Conv2d(self.out_channel,self.out_channel,kernel_size=3,stride=stride,padding=1,groups=self.out_channel)
#         self.conv3=nn.Conv2d(self.out_channel,self.out_channel,kernel_size=1,stride=1,padding=0)
        
#         self.bn=nn.BatchNorm2d(self.out_channel)
#         self.relu6=nn.ReLU6(inplace=True)
#         self.linear=nn.Conv2d(self.out_channel,self.out_channel,kernel_size=1,stride=1,padding=0)

#         if init_weight:
#             self.initialize_weight()

#     def forward(self,x):
#         if not self.downsample:
#             y=x
        
#         x=self.relu6(self.bn(self.conv1(x)))
#         x=self.relu6(self.bn(self.conv2(x)))
#         x=self.linear(self.bn(self.conv3(x)))
        
#         if not self.downsample:
#             x=x+y

#         return x
    
#     def initialize_weight(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
if __name__=='__main__':

    a=torch.randn(1,3,400,400)
    model=MobileResNet18(be_backbone=True,init_weight=True)
    output=model(a)
    print(output[0].shape)
    print(output[1].shape)
    print(output[2].shape)
    print(output[3].shape)
    print(output[4].shape)
    # print(model.named_modules)
    print(model.modules)
