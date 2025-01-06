import torch
import torch.nn as nn
from torchvision.models.squeezenet import SqueezeNet
import math

class FireModule(nn.Module):
    def __init__(self,in_channel,out_channel,se=False):
        super().__init__()

        self.squeeze_in_ch=in_channel
        self.squeeze_out_ch=int(out_channel/8)
        self.expand_in_ch=self.squeeze_out_ch
        self.expand_out_ch=int(out_channel/2)

        self.squeeze_conv=nn.Conv2d(self.squeeze_in_ch,self.squeeze_out_ch,kernel_size=1,stride=1,padding=0)
        self.expand_conv1=nn.Conv2d(self.expand_in_ch,self.expand_out_ch,kernel_size=1,stride=1,padding=0)
        self.expand_conv2=nn.Conv2d(self.expand_in_ch,self.expand_out_ch,kernel_size=3,stride=1,padding=1)

        # self.bn=nn.BatchNorm2d()
        self.bn1=nn.BatchNorm2d(self.squeeze_out_ch)
        self.bn2=nn.BatchNorm2d(self.expand_out_ch*2)

        self.relu=nn.ReLU(inplace=True)

        #squeeze and excitation
        self.se=se
        if self.se:
            self.out_ch=out_channel
            r=int(self.out_ch/16)
            self.squeeze_layer=nn.AdaptiveAvgPool2d(1)
            self.excitation_layer=nn.Sequential(nn.Linear(out_channel,r),nn.ReLU(inplace=True),nn.Linear(r,out_channel),nn.Sigmoid())

    def forward(self,x):

        x=self.bn1(self.squeeze_conv(x))
        x=self.bn2(torch.cat((self.expand_conv1(x),self.expand_conv2(x)),dim=1))

        if self.se:
            y=self.squeeze_layer(x).view(-1,self.out_ch)
            y=self.excitation_layer(y).view(-1,self.out_ch,1,1)
            x=x*y

        return x

class SqueezeResNet18(nn.Module):
    def __init__(self,be_backbone,init_weight_type):
        super().__init__()
        self.be_backbone=be_backbone
        self.init_weight_type=init_weight_type
        
        self.conv1=nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU(inplace=True)
        self.maxpooling=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.layer1=nn.Sequential(FireModule(in_channel=64,out_channel=64),FireModule(in_channel=64,out_channel=64))
        self.layer2=nn.Sequential(FireModule(in_channel=64,out_channel=128),FireModule(in_channel=128,out_channel=128))
        self.layer3=nn.Sequential(FireModule(in_channel=128,out_channel=256),FireModule(in_channel=256,out_channel=256))
        self.layer4=nn.Sequential(FireModule(in_channel=256,out_channel=512),FireModule(in_channel=512,out_channel=512))

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

        x=self.maxpooling(x)

        x=self.layer2(x)
        if self.be_backbone:
            resnet_output.append(x)

        x=self.maxpooling(x)

        x=self.layer3(x)
        if self.be_backbone:
            resnet_output.append(x)

        x=self.maxpooling(x)

        x=self.layer4(x)
        resnet_output.append(x)

        x=self.maxpooling(x)

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

class SqueezeSEResNet18(nn.Module):
    def __init__(self,be_backbone,init_weight_type):
        super().__init__()
        self.be_backbone=be_backbone
        self.init_weight_type=init_weight_type

        self.conv1=nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU(inplace=True)
        self.maxpooling=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.layer1=nn.Sequential(FireModule(in_channel=64,out_channel=64,se=True),FireModule(in_channel=64,out_channel=64,se=True))
        self.layer2=nn.Sequential(FireModule(in_channel=64,out_channel=128,se=True),FireModule(in_channel=128,out_channel=128,se=True))
        self.layer3=nn.Sequential(FireModule(in_channel=128,out_channel=256,se=True),FireModule(in_channel=256,out_channel=256,se=True))
        self.layer4=nn.Sequential(FireModule(in_channel=256,out_channel=512,se=True),FireModule(in_channel=512,out_channel=512,se=True))

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

        x=self.maxpooling(x)

        x=self.layer2(x)
        if self.be_backbone:
            resnet_output.append(x)

        x=self.maxpooling(x)

        x=self.layer3(x)
        if self.be_backbone:
            resnet_output.append(x)

        x=self.maxpooling(x)

        x=self.layer4(x)
        resnet_output.append(x)

        x=self.maxpooling(x)

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


# add skip connection 
class FireModule_SC(nn.Module):
    def __init__(self,in_channel,out_channel,downsample,se=False):
        super().__init__()

        self.squeeze_in_ch=in_channel
        self.squeeze_out_ch=int(out_channel/8)
        self.expand_in_ch=self.squeeze_out_ch
        self.expand_out_ch=int(out_channel/2)
        self.downsample=downsample

        self.squeeze_conv=nn.Conv2d(self.squeeze_in_ch,self.squeeze_out_ch,kernel_size=1,stride=1,padding=0)
        self.expand_conv1=nn.Conv2d(self.expand_in_ch,self.expand_out_ch,kernel_size=1,stride=1,padding=0)
        self.expand_conv2=nn.Conv2d(self.expand_in_ch,self.expand_out_ch,kernel_size=3,stride=1,padding=1)

        # self.bn=nn.BatchNorm2d()
        self.bn1=nn.BatchNorm2d(self.squeeze_out_ch)
        self.bn2=nn.BatchNorm2d(self.expand_out_ch*2)

        self.relu=nn.ReLU(inplace=True)

        #add skip connection
        self.downsampling=nn.Sequential(nn.Conv2d(in_channel,out_channel,kernel_size=1,stride=1,bias=False),nn.BatchNorm2d(out_channel))

        #squeeze and excitation
        self.se=se
        if self.se:
            self.out_ch=out_channel
            r=int(self.out_ch/16)
            self.squeeze_layer=nn.AdaptiveAvgPool2d(1)
            self.excitation_layer=nn.Sequential(nn.Linear(out_channel,r),nn.ReLU(inplace=True),nn.Linear(r,out_channel),nn.Sigmoid())

    def forward(self,x):
        skip_connection=x

        x=self.bn1(self.squeeze_conv(x))
        x=self.bn2(torch.cat((self.expand_conv1(x),self.expand_conv2(x)),dim=1))

        if self.downsample:
            skip_connection=self.downsampling(skip_connection)

        if self.se:
            y=self.squeeze_layer(x).view(-1,self.out_ch)
            y=self.excitation_layer(y).view(-1,self.out_ch,1,1)
            x=x*y

        x=x+skip_connection

        return x

class SqueezeSCResNet18(nn.Module):
    def __init__(self,be_backbone,init_weight_type):
        super().__init__()
        self.be_backbone=be_backbone
        self.init_weight_type=init_weight_type

        self.conv1=nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU(inplace=True)
        self.maxpooling=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.layer1=nn.Sequential(FireModule_SC(in_channel=64,out_channel=64,downsample=False),FireModule_SC(in_channel=64,out_channel=64,downsample=False))
        self.layer2=nn.Sequential(FireModule_SC(in_channel=64,out_channel=128,downsample=True),FireModule_SC(in_channel=128,out_channel=128,downsample=False))
        self.layer3=nn.Sequential(FireModule_SC(in_channel=128,out_channel=256,downsample=True),FireModule_SC(in_channel=256,out_channel=256,downsample=False))
        self.layer4=nn.Sequential(FireModule_SC(in_channel=256,out_channel=512,downsample=True),FireModule_SC(in_channel=512,out_channel=512,downsample=False))

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

        x=self.maxpooling(x)

        x=self.layer2(x)
        if self.be_backbone:
            resnet_output.append(x)

        x=self.maxpooling(x)

        x=self.layer3(x)
        if self.be_backbone:
            resnet_output.append(x)

        x=self.maxpooling(x)

        x=self.layer4(x)
        resnet_output.append(x)

        x=self.maxpooling(x)

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

class SqueezeSESCResNet18(nn.Module):
    def __init__(self,be_backbone,init_weight_type):
        super().__init__()
        self.be_backbone=be_backbone
        self.init_weight_type=init_weight_type

        self.conv1=nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU(inplace=True)
        self.maxpooling=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.layer1=nn.Sequential(FireModule_SC(in_channel=64,out_channel=64,downsample=False,se=True),FireModule_SC(in_channel=64,out_channel=64,downsample=False,se=True))
        self.layer2=nn.Sequential(FireModule_SC(in_channel=64,out_channel=128,downsample=True,se=True),FireModule_SC(in_channel=128,out_channel=128,downsample=False,se=True))
        self.layer3=nn.Sequential(FireModule_SC(in_channel=128,out_channel=256,downsample=True,se=True),FireModule_SC(in_channel=256,out_channel=256,downsample=False,se=True))
        self.layer4=nn.Sequential(FireModule_SC(in_channel=256,out_channel=512,downsample=True,se=True),FireModule_SC(in_channel=512,out_channel=512,downsample=False,se=True))

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

        x=self.maxpooling(x)

        x=self.layer2(x)
        if self.be_backbone:
            resnet_output.append(x)

        x=self.maxpooling(x)

        x=self.layer3(x)
        if self.be_backbone:
            resnet_output.append(x)

        x=self.maxpooling(x)

        x=self.layer4(x)
        resnet_output.append(x)

        x=self.maxpooling(x)

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

class SqueezeSESCResNet18_depth(nn.Module):
    def __init__(self,be_backbone,init_weight_type):
        super().__init__()
        self.be_backbone=be_backbone
        self.init_weight_type=init_weight_type

        self.conv1=nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU(inplace=True)
        self.maxpooling=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.layer1=nn.Sequential(FireModule_SC(in_channel=64,out_channel=64,downsample=False,se=True),FireModule_SC(in_channel=64,out_channel=64,downsample=False,se=True))
        self.layer2=nn.Sequential(FireModule_SC(in_channel=64,out_channel=128,downsample=True,se=True),FireModule_SC(in_channel=128,out_channel=128,downsample=False,se=True))
        self.layer3=nn.Sequential(FireModule_SC(in_channel=128,out_channel=256,downsample=True,se=True),FireModule_SC(in_channel=256,out_channel=256,downsample=False,se=True))
        self.layer4=nn.Sequential(FireModule_SC(in_channel=256,out_channel=512,downsample=True,se=True),FireModule_SC(in_channel=512,out_channel=512,downsample=False,se=True))

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

        x=self.maxpooling(x)

        x=self.layer2(x)
        if self.be_backbone:
            resnet_output.append(x)

        x=self.maxpooling(x)

        x=self.layer3(x)
        if self.be_backbone:
            resnet_output.append(x)

        x=self.maxpooling(x)

        x=self.layer4(x)
        resnet_output.append(x)

        x=self.maxpooling(x)

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

# class Fire(nn.Module):  #pytorch version
#     def __init__(self, inplanes: int, squeeze_planes: int, expand1x1_planes: int, expand3x3_planes: int) -> None:
#         super().__init__()
#         self.inplanes = inplanes
#         self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
#         self.squeeze_activation = nn.ReLU(inplace=True)
#         self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
#         self.expand1x1_activation = nn.ReLU(inplace=True)
#         self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
#         self.expand3x3_activation = nn.ReLU(inplace=True)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.squeeze_activation(self.squeeze(x))
#         return torch.cat(
#             [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
#         )

if __name__=='__main__':
    data=torch.randn(1,3,400,400)
    # model=SqueezeResNet18(be_backbone=True,init_weight_type='xavier')
    # output=model(data)
    # print(output[0].shape)
    # print(output[1].shape)
    # print(output[2].shape)
    # print(output[3].shape)
    # print(output[4].shape)

    # model2=SqueezeSCResNet18(be_backbone=True,init_weight=True)
    # output2=model2(data)
    # print(output2[0].shape)
    # print(output2[1].shape)
    # print(output2[2].shape)
    # print(output2[3].shape)
    # print(output2[4].shape)

    # model3=SqueezeSEResNet18(be_backbone=True,init_weight=True)
    # output3=model3(data)
    # print(output3[0].shape)
    # print(output3[1].shape)
    # print(output3[2].shape)
    # print(output3[3].shape)
    # print(output3[4].shape)
    
    # model4=SqueezeSESCResNet18(be_backbone=True,init_weight_type='xavier')
    # output4=model4(data)
    # print(output4[0].shape)
    # print(output4[1].shape)
    # print(output4[2].shape)
    # print(output4[3].shape)
    # print(output4[4].shape)

    model=SqueezeNet()
    print(model.modules)
