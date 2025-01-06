import torch
import torch.nn as nn
import math

class Double_Conv(nn.Module):
    def __init__(self,first,in_ch,sample,position=None):
        super().__init__()
        if sample=='down':
            if first:
                self.double_conv=nn.Sequential(nn.Conv2d(1,in_ch,kernel_size=3,stride=1,padding=0),nn.BatchNorm2d(in_ch),nn.ReLU(inplace=True),
                                            nn.Conv2d(in_ch,in_ch,kernel_size=3,stride=1,padding=0),nn.BatchNorm2d(in_ch),nn.ReLU(inplace=True))
            else:
                self.double_conv=nn.Sequential(nn.Conv2d(in_ch,in_ch*2,kernel_size=3,stride=1,padding=0),nn.BatchNorm2d(in_ch*2),nn.ReLU(inplace=True),
                                            nn.Conv2d(in_ch*2,in_ch*2,kernel_size=3,stride=1,padding=0),nn.BatchNorm2d(in_ch*2),nn.ReLU(inplace=True))
        elif sample=='up':
            if position=='first':
                self.c=int(in_ch/2)
                self.double_conv=nn.Sequential(nn.Conv2d(in_ch+self.c,in_ch,kernel_size=3,stride=1,padding=1),nn.BatchNorm2d(in_ch),nn.ReLU(inplace=True),
                                                nn.Conv2d(in_ch,in_ch,kernel_size=3,stride=1,padding=1),nn.BatchNorm2d(in_ch),nn.ReLU(inplace=True))
            elif position=='middle':
                self.c1=int(in_ch*2)
                self.c2=int(in_ch/2)
                self.double_conv=nn.Sequential(nn.Conv2d(self.c1+self.c2,in_ch,kernel_size=3,stride=1,padding=1),nn.BatchNorm2d(in_ch),nn.ReLU(inplace=True),
                                                nn.Conv2d(in_ch,in_ch,kernel_size=3,stride=1,padding=1),nn.BatchNorm2d(in_ch),nn.ReLU(inplace=True))
            elif position=='last':
                self.c=int(in_ch*2)
                self.double_conv=nn.Sequential(nn.Conv2d(in_ch+self.c,in_ch,kernel_size=3,stride=1,padding=1),nn.BatchNorm2d(in_ch),nn.ReLU(inplace=True),
                                                nn.Conv2d(in_ch,in_ch,kernel_size=3,stride=1,padding=1),nn.BatchNorm2d(in_ch),nn.ReLU(inplace=True))
    
    def forward(self,x):
        x=self.double_conv(x)

        return x

class UNet_Right_Half(nn.Module):   #modified by UNet
    def __init__(self,num_class,init_weight):
        super().__init__()
        self.up_conv4=Double_Conv(first=True,in_ch=512,sample='up',position='first')
        self.up_conv3=Double_Conv(first=False,in_ch=256,sample='up',position='middle')
        self.up_conv2=Double_Conv(first=False,in_ch=128,sample='up',position='middle')
        self.up_conv1=Double_Conv(first=False,in_ch=64,sample='up',position='last')
        self.upsample=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_fixsize=nn.Upsample(size=25,mode='bilinear', align_corners=True)
        self.final_conv=nn.Conv2d(64,num_class,kernel_size=3,stride=1,padding=1)

        if init_weight:
            self.initialize_weight()

    
    def forward(self,cat_feat):
        x=cat_feat[4]

        x=self.upsample_fixsize(x)
        x=torch.cat((x,cat_feat[3]),dim=1)
        x=self.up_conv4(x)

        x=self.upsample(x)
        x=torch.cat((x,cat_feat[2]),dim=1)
        x=self.up_conv3(x)

        x=self.upsample(x)
        x=torch.cat((x,cat_feat[1]),dim=1)
        x=self.up_conv2(x)

        x=self.upsample(x)
        x=torch.cat((x,cat_feat[0]),dim=1)
        x=self.up_conv1(x)

        x=self.upsample(x)

        x=self.final_conv(x)

        return x

    def initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m,nn.Sequential):
                for mm in m:
                    if isinstance(m, nn.Conv2d):
                        n = mm.kernel_size[0] * mm.kernel_size[1] * mm.out_channels
                        mm.weight.data.normal_(0, math.sqrt(2. / n))
                    elif isinstance(m, nn.BatchNorm2d):
                        mm.weight.data.fill_(1)
                        mm.bias.data.zero_()

class FCNs_concat(nn.Module):
    def __init__(self,num_class,init_weight):
        super().__init__()
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=0)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(768, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(512)
        self.deconv3 = nn.ConvTranspose2d(640, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(256)
        self.deconv4 = nn.ConvTranspose2d(320, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(128)
        self.deconv5 = nn.ConvTranspose2d(192, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(64)
        self.classifier = nn.Conv2d(64, num_class, kernel_size=1)

        if init_weight:
            self.initialize_weight()

    def forward(self,output):
        score = output[4]  
        x4 = output[3]  
        x3 = output[2]  
        x2 = output[1]  
        x1 = output[0] 

        score = self.bn1(self.relu(self.deconv1(score)))     
        score = torch.cat((score,x4),dim=1)                                
        score = self.bn2(self.relu(self.deconv2(score)))  
        score = torch.cat((score,x3),dim=1)                                
        score = self.bn3(self.relu(self.deconv3(score)))  
        score = torch.cat((score,x2),dim=1)                                
        score = self.bn4(self.relu(self.deconv4(score)))  
        score = torch.cat((score,x1),dim=1)                                
        score = self.bn5(self.relu(self.deconv5(score)))  
        score = self.classifier(score)                    

        return score  
    
    def initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class FCNs_add(nn.Module):
    def __init__(self,num_class,init_weight):
        super().__init__()
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=0)   #13 to 25
        self.bn1     = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, num_class, kernel_size=1)

        if init_weight:
            self.initialize_weight()

    def forward(self,output):
        x5 = output[4]  
        x4 = output[3]  
        x3 = output[2]  
        x2 = output[1]  
        x1 = output[0]  

        score = self.bn1(self.relu(self.deconv1(x5)))     
        score = score+x4                                
        score = self.bn2(self.relu(self.deconv2(score)))  
        score = score+x3                                
        score = self.bn3(self.relu(self.deconv3(score)))  
        score = score+x2                                 
        score = self.bn4(self.relu(self.deconv4(score)))  
        score = score+x1                                
        score = self.bn5(self.relu(self.deconv5(score)))  
        score = self.classifier(score)                    

        return score  
    
    def initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

if __name__=='__main__':
    model=UNet_Right_Half(num_class=2,init_weight=True)
    for m in model.parameters():
        print(m)
