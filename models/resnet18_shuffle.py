import torch
import torch.nn as nn
import math
from torchvision.models.shufflenetv2 import ShuffleNetV2

def channel_shuffle(x,groups):
    n,c,h,w=x.size()
    channel_per_group=c//groups
    x=x.view(n,groups,channel_per_group,h,w)
    x=torch.transpose(x,1,2).contiguous()
    x=x.view(n,-1,h,w)

    return x

class ShuffleBlock(nn.Module):  #InvertedResidual in orchvision.models.shufflenetv2
    def __init__(self,in_ch,out_ch,stride):
        super().__init__()
        self.stride=stride
        branch_features=out_ch//2

        if self.stride>1:
            self.branch1=nn.Sequential(
                nn.Conv2d(in_ch,in_ch,kernel_size=3,stride=self.stride,padding=1,groups=in_ch),
                nn.BatchNorm2d(in_ch),
                nn.Conv2d(in_ch,branch_features,kernel_size=1,stride=1,padding=0,bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch1=nn.Sequential()

        self.branch2=nn.Sequential(
            nn.Conv2d(in_ch if(self.stride>1) else branch_features,branch_features,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_features,branch_features,kernel_size=3,stride=self.stride,padding=1,groups=branch_features),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features,branch_features,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out

class ShuffleResNet18(nn.Module):   #substitute basic residual blocks to shuffle blocks
    def __init__(self,be_backbone,init_weight_type):
        super().__init__()
        self.be_backbone=be_backbone
        self.init_weight_type=init_weight_type

        self.conv1=nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU(inplace=True)
        self.maxpooling=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        
        # self.layer1=nn.Sequential(ShuffleBlock(downsample=False,in_channel=64,init_weight=init_weight),ShuffleBlock(downsample=False,in_channel=64,init_weight=init_weight))
        # self.layer2=nn.Sequential(ShuffleBlock(downsample=True,in_channel=64,init_weight=init_weight),ShuffleBlock(downsample=False,in_channel=128,init_weight=init_weight))
        # self.layer3=nn.Sequential(ShuffleBlock(downsample=True,in_channel=128,init_weight=init_weight),ShuffleBlock(downsample=False,in_channel=256,init_weight=init_weight))
        # self.layer4=nn.Sequential(ShuffleBlock(downsample=True,in_channel=256,init_weight=init_weight),ShuffleBlock(downsample=False,in_channel=512,init_weight=init_weight))

        self.layer1=nn.Sequential(ShuffleBlock(in_ch=64,out_ch=64,stride=1),ShuffleBlock(in_ch=64,out_ch=64,stride=1))
        self.layer2=nn.Sequential(ShuffleBlock(in_ch=64,out_ch=128,stride=2),ShuffleBlock(in_ch=128,out_ch=128,stride=1))
        self.layer3=nn.Sequential(ShuffleBlock(in_ch=128,out_ch=256,stride=2),ShuffleBlock(in_ch=256,out_ch=256,stride=1))
        self.layer4=nn.Sequential(ShuffleBlock(in_ch=256,out_ch=512,stride=2),ShuffleBlock(in_ch=512,out_ch=512,stride=1))



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

# class Channel_Shuffle(nn.Module):
#     def __init__(self,num_groups):
#         super().__init__()
#         self.num_groups=num_groups
#     def forward(self,x):
#         b,c,h,w=x.shape
#         c_per_group=c//self.num_groups
#         x=torch.reshape(x,(b,self.num_groups,c_per_group,h,w))
#         x=x.transpose(1,2)
#         out=torch.reshape(x,(b,-1,h,w))
#         return out

# class ShuffleBlock(nn.Module):
#     def __init__(self,downsample,in_channel,init_weight):
#         super().__init__()
#         self.downsample=downsample

#         if self.downsample:
#             self.in_channel=in_channel
#             self.out_channel=in_channel
#             stride=2
#         else:
#             self.in_channel=int(in_channel/2)
#             self.out_channel=self.in_channel
#             stride=1

#         self.conv1=nn.Conv2d(self.in_channel,self.out_channel,kernel_size=1,stride=1,padding=0)
#         self.conv2=nn.Conv2d(self.out_channel,self.out_channel,kernel_size=3,stride=stride,padding=1,groups=self.out_channel)
#         self.conv3=nn.Conv2d(self.out_channel,self.out_channel,kernel_size=1,stride=1,padding=0)
#         self.bn=nn.BatchNorm2d(self.out_channel)
#         self.relu=nn.ReLU(inplace=True)
#         # self.shuffle=nn.ChannelShuffle(4)
#         self.shuffle=Channel_Shuffle(4)

#         if init_weight:
#             self.initialize_weight()

#     def forward(self,x):
#         if not self.downsample:
#             x1,x2=x.chunk(2,dim=1)
#         else:
#             x1=x
#             x2=x
#             x2=self.bn(self.conv2(x2))
#             x2=self.relu(self.bn(self.conv3(x2)))

#         x1=self.relu(self.bn(self.conv1(x1)))
#         x1=self.bn(self.conv2(x1))
#         x1=self.relu(self.bn(self.conv3(x1)))

#         x1=torch.cat((x1,x2),dim=1)
#         out=self.shuffle(x1)

#         return out

#     def initialize_weight(self):
#          for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()

if __name__=='__main__':
    a=torch.randn(1,3,400,400)
    # model=ShuffleBlock(downsample=False,in_channel=128)
    model=ShuffleResNet18(be_backbone=True,init_weight=True)

    print(model.modules)
    output=model(a)
    print(output[0].shape)
    print(output[1].shape)
    print(output[2].shape)
    print(output[3].shape)
    print(output[4].shape)

    # tensor=torch.tensor([[[[0,0,0,0],[1,1,1,1],[2,2,2,2],[3,3,3,3]],[[4,4,4,4],[5,5,5,5],[6,6,6,6],[7,7,7,7]],[[8,8,8,8],[9,9,9,9],[10,10,10,10],[11,11,11,11]],[[11,11,11,11],[12,12,12,12],[13,13,13,13],[14,14,14,14]]]])
    # print(tensor.shape)
    # print(tensor)
    # cs=Channel_Shuffle(2)
    # print(cs(tensor))

    # data=torch.ones(1,12,1,1)
    # for i in range(12):
    #     data[0][i]=i
    # print(data)
    # cs1=Channel_Shuffle(2)
    # print(cs1(data))
    # cs2=Channel_Shuffle(3)
    # print(cs2(data))
