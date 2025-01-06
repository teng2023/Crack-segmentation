import torch
import torch.nn as nn
import math

class SqueezeNextBlock(nn.Module):
    def __init__(self,in_channel,out_channel,downsample,se):
        super().__init__()

        self.downsample=downsample
        if self.downsample:
            stride=2
        else:
            stride=1
        
        self.conv1=nn.Conv2d(in_channel,out_channel//2,kernel_size=1,stride=stride,padding=0)
        self.bn1=nn.BatchNorm2d(out_channel//2)
        self.conv2=nn.Conv2d(out_channel//2,out_channel//4,kernel_size=1,stride=1,padding=0)
        self.bn2=nn.BatchNorm2d(out_channel//4)
        self.conv3=nn.Conv2d(out_channel//4,out_channel//2,kernel_size=(1,3),stride=1,padding=(0,1))
        self.bn3=nn.BatchNorm2d(out_channel//2)
        self.conv4=nn.Conv2d(out_channel//2,out_channel//2,kernel_size=(3,1),stride=1,padding=(1,0))
        self.bn4=nn.BatchNorm2d(out_channel//2)
        self.conv5=nn.Conv2d(out_channel//2,out_channel,kernel_size=1,stride=1,padding=0)
        self.bn5=nn.BatchNorm2d(out_channel)
        self.relu=nn.ReLU(inplace=True)


        #add skip connection
        self.downsampling=nn.Sequential(nn.Conv2d(in_channel,out_channel,kernel_size=1,stride=2,bias=False),nn.BatchNorm2d(out_channel))
        
        #squeeze and excitation
        self.se=se
        if se:
            self.out_ch=out_channel
            r=int(self.out_ch/16)
            self.squeeze_layer=nn.AdaptiveAvgPool2d(1)
            self.excitation_layer=nn.Sequential(nn.Linear(out_channel,r),nn.ReLU(inplace=True),nn.Linear(r,out_channel),nn.Sigmoid())

    def forward(self,x):
        skip_connection=x

        x=self.relu(self.bn1(self.conv1(x)))
        x=self.relu(self.bn2(self.conv2(x)))
        x=self.relu(self.bn3(self.conv3(x)))
        x=self.relu(self.bn4(self.conv4(x)))
        x=self.relu(self.bn5(self.conv5(x)))

        if self.downsample:
            skip_connection=self.downsampling(skip_connection)

        if self.se:
            y=self.squeeze_layer(x).view(-1,self.out_ch)
            y=self.excitation_layer(y).view(-1,self.out_ch,1,1)
            x=x*y

        x=x+skip_connection

        return x

class SqueezeNextResNet18(nn.Module):
    def __init__(self,be_backbone,init_weight_type):
        super().__init__()
        self.be_backbone=be_backbone
        self.init_weight_type=init_weight_type

        self.conv1=nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU(inplace=True)
        self.maxpooling=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.layer1=nn.Sequential(SqueezeNextBlock(in_channel=64,out_channel=64,downsample=False,se=True),SqueezeNextBlock(in_channel=64,out_channel=64,downsample=False,se=True))
        self.layer2=nn.Sequential(SqueezeNextBlock(in_channel=64,out_channel=128,downsample=True,se=True),SqueezeNextBlock(in_channel=128,out_channel=128,downsample=False,se=True))
        self.layer3=nn.Sequential(SqueezeNextBlock(in_channel=128,out_channel=256,downsample=True,se=True),SqueezeNextBlock(in_channel=256,out_channel=256,downsample=False,se=True))
        self.layer4=nn.Sequential(SqueezeNextBlock(in_channel=256,out_channel=512,downsample=True,se=True),SqueezeNextBlock(in_channel=512,out_channel=512,downsample=False,se=True))

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
                            
if __name__=='__main__':
    data=torch.randn(1,3,400,400)

    model4=SqueezeNextResNet18(be_backbone=True,init_weight=True)
    output4=model4(data)
    print(output4[0].shape)
    print(output4[1].shape)
    print(output4[2].shape)
    print(output4[3].shape)
    print(output4[4].shape)
