import torch
import torch.nn as nn
import torch.nn.functional as F
import math

'''
Quantized
'''
def quantize_tensor(tensor, num_bits=8):
  max = tensor.max()
  min = tensor.min()
  scale = (max - min) / (2**num_bits - 1)
  quantized_tensor = torch.round(tensor/scale)
  return quantized_tensor, scale

def dequantize_tensor(quantized_tensor, scale):
  return quantized_tensor*scale

class FakeQuantOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, num_bits=8):
        x, scale = quantize_tensor(x, num_bits=num_bits)
        x = dequantize_tensor(x, scale)
        return x
    @staticmethod
    def backward(ctx, grad_output):
        # straight through estimator
        return grad_output, None, None, None

class Conv_Quant(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, num_bits=8):
    super().__init__()
    self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    self.stride = stride
    self.padding = padding
    self.num_bits = num_bits

  def forward(self, x):
    # Convolution layer.
    w = self.conv.weight
    b = self.conv.bias
    # Quantized & dequantized.
    w = FakeQuantOp.apply(w, self.num_bits)
    b = FakeQuantOp.apply(b, self.num_bits)
    # Apply convolution computation.
    x = F.conv2d(input=x, weight=w, bias=b, stride=self.stride, padding=self.padding)
    return x
  
class ConvBN_Quant(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,bias=True, num_bits=8):
    super().__init__()
    self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    self.bn = nn.BatchNorm2d(num_features=out_channels)
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.num_bits = num_bits
    self.bias = bias

  def forward(self, x):
    # Convolution layer.
    w = self.conv.weight
    b = self.conv.bias
    # BatchNorm layer.
    beta = self.bn.weight
    gamma = self.bn.bias
    # Mean & Variance.
    mean = self.bn.running_mean
    var = self.bn.running_var
    var_sqrt = torch.sqrt(var + self.bn.eps)
    # Fused weight and bias.
    w = w / (var_sqrt.unsqueeze(1).unsqueeze(1).unsqueeze(1)) * (beta.unsqueeze(1).unsqueeze(1).unsqueeze(1))
    if self.bias: 
        b = (b - mean) / var_sqrt * beta + gamma
        b = FakeQuantOp.apply(b, self.num_bits)

    # Quantized & dequantized.
    w = FakeQuantOp.apply(w, self.num_bits)
    # Apply convolution computation.
    x = F.conv2d(input=x, weight=w, bias=b, stride=self.stride, padding=self.padding)
    return x
  
class Linear_Quant(nn.Module):
  def __init__(self, in_features, out_features, num_bits=8):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.num_bits = num_bits
    self.fc = nn.Linear(in_features=in_features, out_features=out_features)

  def forward(self,x):
    w = self.fc.weight
    b = self.fc.bias
    w = FakeQuantOp.apply(w, self.num_bits)
    b = FakeQuantOp.apply(b, self.num_bits)
    x = F.linear(input=x, weight=w, bias=b)
    return x

'''
Original model with quantized class
'''
class FireModule(nn.Module):
    def __init__(self,in_channel,out_channel,se=False):
        super().__init__()

        self.squeeze_in_ch=in_channel
        self.squeeze_out_ch=int(out_channel/8)
        self.expand_in_ch=self.squeeze_out_ch
        self.expand_out_ch=int(out_channel/2)

        '''Quantized layers'''
        self.conv_bn1 = ConvBN_Quant(self.squeeze_in_ch,self.squeeze_out_ch,kernel_size=1,stride=1,padding=0)
        self.expand_conv_bn1 = ConvBN_Quant(self.expand_in_ch,self.expand_out_ch,kernel_size=1,stride=1,padding=0)
        self.expand_conv_bn2 = ConvBN_Quant(self.expand_in_ch,self.expand_out_ch,kernel_size=3,stride=1,padding=1)
        self.se=se
        if self.se:
            self.out_ch=out_channel
            r=int(self.out_ch/16)
            self.squeeze_layer=nn.AdaptiveAvgPool2d(1)
            self.excitation_layer=nn.Sequential(Linear_Quant(out_channel,r),nn.ReLU(inplace=True),Linear_Quant(r,out_channel),nn.Sigmoid())
    
    def forward(self,x):

        x=self.conv_bn1(x)
        x=torch.cat((self.expand_conv_bn1(x),self.expand_conv_bn2(x)),dim=1)

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
        

        self.relu=nn.ReLU(inplace=True)
        self.maxpooling=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.layer1=nn.Sequential(FireModule(in_channel=64,out_channel=64),FireModule(in_channel=64,out_channel=64))
        self.layer2=nn.Sequential(FireModule(in_channel=64,out_channel=128),FireModule(in_channel=128,out_channel=128))
        self.layer3=nn.Sequential(FireModule(in_channel=128,out_channel=256),FireModule(in_channel=256,out_channel=256))
        self.layer4=nn.Sequential(FireModule(in_channel=256,out_channel=512),FireModule(in_channel=512,out_channel=512))
        
        '''Quantized layers'''
        self.conv_bn1 = ConvBN_Quant(3,64,kernel_size=7,stride=2,padding=3,bias=False)


        if self.init_weight_type:
            self.initialize_weight()

    def forward(self,x):
        resnet_output=[]
        x=self.conv_bn1(x)
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

        return resnet_output
    
    def initialize_weight(self):
        for m in self.modules():
            if isinstance(m, ConvBN_Quant):
                n = m.conv.kernel_size[0] * m.conv.kernel_size[1] * m.conv.out_channels
                m.conv.weight.data.normal_(0, math.sqrt(2. / n))
                m.bn.weight.data.fill_(1)
                m.bn.bias.data.zero_()
            elif isinstance(m, Conv_Quant):
                n = m.conv.kernel_size[0] * m.conv.kernel_size[1] * m.conv.out_channels
                m.conv.weight.data.normal_(0, math.sqrt(2. / n))

class Double_Conv(nn.Module):
    def __init__(self,first,in_ch,sample,position=None):
        super().__init__()
        '''Quantized layers'''
        if sample=='down':
            if first:
                self.double_conv=nn.Sequential(ConvBN_Quant(1,in_ch,kernel_size=3,stride=1,padding=0),nn.ReLU(inplace=True),
                                            ConvBN_Quant(in_ch,in_ch,kernel_size=3,stride=1,padding=0),nn.ReLU(inplace=True))
            else:
                self.double_conv=nn.Sequential(ConvBN_Quant(in_ch,in_ch*2,kernel_size=3,stride=1,padding=0),nn.ReLU(inplace=True),
                                            ConvBN_Quant(in_ch*2,in_ch*2,kernel_size=3,stride=1,padding=0),nn.ReLU(inplace=True))
        elif sample=='up':
            if position=='first':
                self.c=int(in_ch/2)
                self.double_conv=nn.Sequential(ConvBN_Quant(in_ch+self.c,in_ch,kernel_size=3,stride=1,padding=1),nn.ReLU(inplace=True),
                                                ConvBN_Quant(in_ch,in_ch,kernel_size=3,stride=1,padding=1),nn.ReLU(inplace=True))
            elif position=='middle':
                self.c1=int(in_ch*2)
                self.c2=int(in_ch/2)
                self.double_conv=nn.Sequential(ConvBN_Quant(self.c1+self.c2,in_ch,kernel_size=3,stride=1,padding=1),nn.ReLU(inplace=True),
                                                ConvBN_Quant(in_ch,in_ch,kernel_size=3,stride=1,padding=1),nn.ReLU(inplace=True))
            elif position=='last':
                self.c=int(in_ch*2)
                self.double_conv=nn.Sequential(ConvBN_Quant(in_ch+self.c,in_ch,kernel_size=3,stride=1,padding=1),nn.ReLU(inplace=True),
                                                ConvBN_Quant(in_ch,in_ch,kernel_size=3,stride=1,padding=1),nn.ReLU(inplace=True))
    
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
        self.final_conv_quant = Conv_Quant(64,num_class,kernel_size=3,stride=1,padding=1)

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

        x=self.final_conv_quant(x)

        return x

    def initialize_weight(self):
        for m in self.modules():
            if isinstance(m, ConvBN_Quant):
                n = m.conv.kernel_size[0] * m.conv.kernel_size[1] * m.conv.out_channels
                m.conv.weight.data.normal_(0, math.sqrt(2. / n))
                m.bn.weight.data.fill_(1)
                m.bn.bias.data.zero_()
            elif isinstance(m, Conv_Quant):
                n = m.conv.kernel_size[0] * m.conv.kernel_size[1] * m.conv.out_channels
                m.conv.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m,nn.Sequential):
                for mm in m:
                    if isinstance(mm, ConvBN_Quant):
                        n = mm.conv.kernel_size[0] * mm.conv.kernel_size[1] * mm.conv.out_channels
                        mm.conv.weight.data.normal_(0, math.sqrt(2. / n))
                        mm.bn.weight.data.fill_(1)
                        mm.bn.bias.data.zero_()
                    elif isinstance(mm, Conv_Quant):
                        n = mm.conv.kernel_size[0] * mm.conv.kernel_size[1] * mm.conv.out_channels
                        mm.conv.weight.data.normal_(0, math.sqrt(2. / n))

class Res18SqueezeUNet_Quant(nn.Module):
    def __init__(self,num_class,init_weight_type):
        super().__init__()
        self.ressqueeze=SqueezeResNet18(be_backbone=True,init_weight_type=False)
        self.unet=UNet_Right_Half(num_class=num_class,init_weight=False)
    
    def forward(self,x):
        x=self.ressqueeze(x)
        x=self.unet(x)

        return x

if  __name__=='__main__':
    model_load=Res18SqueezeUNet_Quant(num_class=2,init_weight_type='xavier')
    weight_load=torch.load('crack_model_test/check points/resnet18+squeeze+unet+quant7/Epoch142_of_250.pth',map_location=torch.device('cpu'))
    model_load.load_state_dict(weight_load['model'],strict=False)
    save_mdoel=torch.save(model_load,'crack_model_test/resnet_with_squeezenet_and_unet.pth', _use_new_zipfile_serialization=False)