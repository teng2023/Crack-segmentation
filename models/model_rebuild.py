import torch.nn as nn
import torch
import math
import torchvision

def unt_srdefnet18(num_classes=2, pretrained_own=None):
    '''
    num_classes : colorlabels的種類個數

    pretrained_own : 非None表要載入自己的權重(權重檔的相對位置)
    '''
    in_channels_list = [192, 320, 640, 768]
    
    if pretrained_own:
        model = Unet_model(srdefnet18(), in_channels_list, num_classes=num_classes)
        model.load_state_dict(pretrained_own)
    else:
        model = Unet_model(srdefnet18(), in_channels_list, num_classes=num_classes)
    
    return model

def srdefnet18(pretrained_own=False):
    '''
    pretrained_own : 非None表要載入自己的權重(權重檔的相對位置)
    '''
    model = DefNet(Def_Block_SR, [2, 2, 2, 2])
    if pretrained_own:
        model.load_state_dict(pretrained_own, strict=False)

    return model

class Unet_model(nn.Module):
    def __init__(self, backbone, in_channels_list, num_classes=2, out_channels_list=[64, 128, 256, 512], input_size=400):
        super().__init__()
        self.final = nn.Conv2d(64, num_classes, 1)
        self.up_conv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up_concat1 = Unet_Up(in_channels_list[0], out_channels_list[0], int(input_size / 2 ** 1))
        self.up_concat2 = Unet_Up(in_channels_list[1], out_channels_list[1], int(input_size / 2 ** 2))
        self.up_concat3 = Unet_Up(in_channels_list[2], out_channels_list[2], int(input_size / 2 ** 3))
        self.up_concat4 = Unet_Up(in_channels_list[3], out_channels_list[3], int(input_size / 2 ** 4))
        self.backbone = backbone

        #設定conv和batchnorm的初始權重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):
        [feat1, feat2, feat3, feat4, feat5] = self.backbone(inputs)
        outputs = self.up_concat4(feat4, feat5)
        outputs = self.up_concat3(feat3, outputs)
        outputs = self.up_concat2(feat2, outputs)
        outputs = self.up_concat1(feat1, outputs)
        
        outputs = self.up_conv(outputs)
        outputs = self.final(outputs)
        return outputs
    
    def load_state_dict(self, path):
        w = torch.load(path)
        super().load_state_dict(w)

class Unet_Up(nn.Module):
    def __init__(self, in_channels, out_channels, inputs1_size):
        super(Unet_Up, self).__init__()
        self.up     = nn.UpsamplingBilinear2d(size=inputs1_size)
        self.conv1  = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2  = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) 
        self.relu   = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs
    
class DefNet(nn.Module):
    '''
    block : BasicBlock(resnet18和resnet34) or Bottleneck(resnet50,resnet101和resnet152)

    layers : 此layer要重複幾次block
    '''
    def __init__(self, block, layers):
        super().__init__()

        self.basicplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.layer1 = self._make_layer(block, 1, layers[0])
        self.layer2 = self._make_layer(block, 2, layers[1])
        self.layer3 = self._make_layer(block, 3, layers[2])
        self.layer4 = self._make_layer(block, 4, layers[3])

        #設定conv和batchnorm的初始權重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, layer_th, blocks):
        if layer_th - 1 == 0:
            stride = 1
            down_rate = 1
        else:
            stride = 2
            down_rate = block.expansion / 2
        channels = self.basicplanes * 2 ** (layer_th - 1)
        downsample = nn.Sequential(
            nn.Conv2d(int(channels * down_rate), channels * block.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(channels * block.expansion),
        )

        layers = []
        layers.append(block(self.basicplanes, channels, stride, downsample))
        for i in range(1, blocks):
            layers.append(block(self.basicplanes, channels))

        return nn.Sequential(*layers)
    
    def forward(self, inputs):
        inputs = self.conv1(inputs)
        inputs = self.bn1(inputs)
        feat1  = self.relu(inputs)

        inputs = self.maxpool(feat1)
        feat2  = self.layer1(inputs)
        feat3  = self.layer2(feat2)
        feat4  = self.layer3(feat3)
        feat5  = self.layer4(feat4)

        return [feat1, feat2, feat3, feat4, feat5]
    
class Def_Block_SR(nn.Module):
    '''srdefnet18和srdefnet34的基本block

    . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

    1.若downsample不是none,表殘差分支需進行下採樣,而非identity
    '''
    expansion = 1

    def __init__(self, basicplanes, channels, stride=1, downsample=None):
        super().__init__()
        if downsample:
            if channels == basicplanes:
                in_channels_rate = 1
            else:
                in_channels_rate = 0.5
        else:
            in_channels_rate = self.expansion

        self.conv1 = nn.Conv2d(int(channels * in_channels_rate), channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.offset = nn.Conv2d(channels, 2*9, 3, 1, 1)
        self.dconv = torchvision.ops.DeformConv2d(channels, channels * self.expansion, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        
    def forward(self, inputs):
        identity = inputs

        outputs = self.conv1(inputs)
        outputs = self.bn1(outputs)
        outputs = self.relu(outputs)

        offset = self.offset(outputs)
        outputs = self.dconv(outputs, offset)
        outputs = self.bn2(outputs)

        if self.downsample is not None:
            identity = self.downsample(inputs)

        outputs = outputs + identity
        outputs = self.relu(outputs)

        return outputs