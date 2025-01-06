import torch
import torch.nn as nn
import math

class GhostModule(nn.Conv2d):   # ghost convolution (official version)
    def __init__(self,in_channels,out_channels,kernel_size,dw_size=3,ratio=2,stride=1,padding=0,dilation=1,groups=1,bias=True):
        super().__init__(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias)
        self.weight=None
        self.ratio=ratio
        self.dw_size=dw_size
        self.dilation=(dw_size-1)//2
        self.init_channels=math.ceil(out_channels/ratio)
        self.new_channels=self.init_channels*(ratio-1)

        self.conv1=nn.Conv2d(self.in_channels,self.init_channels,kernel_size,self.stride,padding=self.padding)
        self.conv2=nn.Conv2d(self.init_channels,self.new_channels,self.dw_size,1,padding=int(self.dw_size/2),groups=self.init_channels)

        self.weight1=nn.Parameter(torch.Tensor(self.init_channels,self.in_channels,kernel_size,kernel_size))
        self.bn1=nn.BatchNorm2d(self.init_channels)

        if self.new_channels>0:
            self.weight2=nn.Parameter(torch.Tensor(self.new_channels,1,self.dw_size,self.dw_size))
            self.bn2=nn.BatchNorm2d(self.out_channels-self.init_channels)

        if bias:
            self.bias=nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias',None)
        
        self.reset_custome_parameters()

    def reset_custome_parameters(self):
        nn.init.kaiming_uniform_(self.weight1,a=math.sqrt(5))

        if self.new_channels>0:
            nn.init.kaiming_uniform_(self.weight2,a=math.sqrt(5))
        if self.bias is not None:
            nn.init.constant_(self.bias,0)

    def forward(self,input):
        x1=self.conv1(input)

        if self.new_channels==0:
            return x1
        
        x2=self.conv2(x1)
        x2=x2[: self.out_channels-self.init_channels,:,:]
        x=torch.cat([x1,x2],dim=1)
        
        return x

class GhostBottle(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        if in_ch!=out_ch:
            self.downsample=True
            self.dwconv=nn.Conv2d(in_ch*4,in_ch*4,kernel_size=3,stride=2,padding=1,groups=in_ch*4)
            self.bn2=nn.BatchNorm2d(in_ch*4)
            self.skip_connection=GhostModule(in_ch,out_ch,kernel_size=1,stride=2)
        else:
            self.downsample=False

        self.conv1=GhostModule(in_ch,in_ch*4,kernel_size=1,ratio=4)
        self.conv2=GhostModule(in_ch*4,out_ch,kernel_size=1,ratio=4)

        self.bn1=nn.BatchNorm2d(in_ch*4)
        self.bn3=nn.BatchNorm2d(out_ch)
        self.relu=nn.ReLU(inplace=True)

    def forward(self,x):
        residual=x

        output=self.relu(self.bn1(self.conv1(x)))
        
        if self.downsample:
            output=self.bn2(self.dwconv(output))
            residual=self.skip_connection(x)
        
        output=self.bn3(self.conv2(output))

        output=output+residual

        return output


# class Bottleneck(nn.Module):    # official version
#     expansion=4
#     def __init__(self,in_channel,out_channel,stride=1,downsample=False,s=4,d=3):
#         super().__init__()
#         self.conv1=GhostModule(in_channel,out_channel,kernel_size=1,dw_size=d,ratio=s,bias=None)
#         self.conv2=GhostModule(out_channel,out_channel,kernel_size=3,dw_size=d,ratio=s,stride=stride,padding=1,bias=None)
#         self.conv3=GhostModule(out_channel,out_channel*4,kernel_size=1,dw_size=d,ratio=s,bias=False)

#         # self.bn=nn.BatchNorm2d()
#         self.relu=nn.ReLU(inplace=True)
#         self.downsample=downsample
#         self.stride=stride

#         if stride!=1 or in_channel!=out_channel*self.expansion:
#             downsample=GhostModule(in_channel,out_channel*self.expansion,ratio=s,dw_size=d,kernel_size=1,stride=stride,bias=False)

#     def forward(self,x):
#         residual=x

#         out=self.conv1(x)
#         out=self.relu(out)
#         out=self.conv2(out)
#         out=self.relu(out)
#         out=self.conv3(out)

#         if self.downsample:
#             residual=self.downsample(x)
#         out+=residual
#         out=self.relu(out)

#         return out
        
class GhostResNet18(nn.Module):
    def __init__(self,be_backbone,init_weight_type):
        super().__init__()
        self.be_backbone=be_backbone
        self.init_weight_type=init_weight_type

        self.conv1=nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU(inplace=True)
        self.maxpooling=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        # self.layer1=nn.Sequential(Bottleneck(in_channel=64,out_channel=64),Bottleneck(in_channel=64,out_channel=64))
        # self.layer2=nn.Sequential(Bottleneck(in_channel=64,out_channel=128,stride=2),Bottleneck(in_channel=128,out_channel=128))
        # self.layer3=nn.Sequential(Bottleneck(in_channel=128,out_channel=256,stride=2),Bottleneck(in_channel=256,out_channel=256))
        # self.layer4=nn.Sequential(Bottleneck(in_channel=256,out_channel=512,stride=2),Bottleneck(in_channel=512,out_channel=512))

        self.layer1=nn.Sequential(GhostBottle(in_ch=64,out_ch=64),GhostBottle(in_ch=64,out_ch=64))
        self.layer2=nn.Sequential(GhostBottle(in_ch=64,out_ch=128),GhostBottle(in_ch=128,out_ch=128))
        self.layer3=nn.Sequential(GhostBottle(in_ch=128,out_ch=256),GhostBottle(in_ch=256,out_ch=256))
        self.layer4=nn.Sequential(GhostBottle(in_ch=256,out_ch=512),GhostBottle(in_ch=512,out_ch=512))


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
    
    def initialize_weight(self):    #Xavier
        if self.init_weight_type=='xavier':
            for m in [self.conv1,self.bn1]: #because ghost module already initialized
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        elif self.init_weight_type=='kaiming':
            initial_target=[self.conv1,self.bn1]
            for layer in initial_target:
                for module in layer:
                    if isinstance(module,nn.BatchNorm2d):
                        nn.init.constant_(module.weight,1)
                        nn.init.constant_(module.bias,0)
                    elif isinstance(module,nn.ConvTranspose2d):
                        nn.init.kaiming_normal_(module.weight,mode='fan_out',nonlinearity='relu')
                        if module.bias is not None:
                            nn.init.constant_(module.bias,0)


if __name__=='__main__':
    data=torch.randn(1,3,400,400)

    # model=GhostModule(in_channels=8,out_channels=8,kernel_size=1,stride=2)
    # output=model(data)
    # print(output.shape)


    # model=GhostBottle(in_ch=8,out_ch=8)
    # output=model(data)
    # print(output.shape)


    model=GhostResNet18(be_backbone=True,init_weight_type='xavier')
    output=model(data)
    print(output[0].shape)
    print(output[1].shape)
    print(output[2].shape)
    print(output[3].shape)
    print(output[4].shape)
    # print(model.modules)