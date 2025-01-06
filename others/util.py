import torch.nn as nn
from torchvision.ops.misc import Conv2dNormActivation
# import numpy as np
# import pydensecrf.densecrf as dcrf
# import pydensecrf.utils as utils

class DepthwiseSeparableConv(nn.Module):
    def __init__(self,in_channel,out_channel,stride,kernel,reverse):
        super().__init__()
        self.reverse=reverse    #if reverse, pointwise then depthwise

        if self.reverse:
            channel=out_channel
        else:
            channel=in_channel

        # Depthwise Convolution
        self.depthconv=Conv2dNormActivation(
                in_channels=channel,out_channels=channel,kernel_size=kernel,stride=stride,padding=1,
                groups=channel,  # the key of depthwise convolution
                norm_layer=nn.BatchNorm2d,activation_layer=nn.ReLU,inplace=True,bias=False)
        
        # Pointwise Convolution
        self.pointconv=Conv2dNormActivation(
                in_channels=in_channel,out_channels=out_channel,kernel_size=1,stride=1,padding=0,groups=1,
                norm_layer=nn.BatchNorm2d,activation_layer=nn.ReLU,inplace=True,bias=False)
    
    def forward(self,x):
        if self.reverse:
            x=self.pointconv(x)
            x=self.depthconv(x)
        else:
            x=self.depthconv(x)
            x=self.pointconv(x)

        return x

def get_param(model):
    total_sum=sum(p.numel() for p in model.parameters())
    trainable_sum=sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total':total_sum,'trainable':trainable_sum}


# MAX_ITER = 10
# POS_W = 3
# POS_XY_STD = 1
# Bi_W = 4
# Bi_XY_STD = 67
# Bi_RGB_STD = 3

# def dense_crf(img, output_probs):
#     c = output_probs.shape[0]
#     h = output_probs.shape[1]
#     w = output_probs.shape[2]

#     U = utils.unary_from_softmax(output_probs)
#     U = np.ascontiguousarray(U)

#     img = np.ascontiguousarray(img)

#     d = dcrf.DenseCRF2D(w, h, c)
#     d.setUnaryEnergy(U)
#     d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
#     d.addPairwiseBilateral(sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=img, compat=Bi_W)

#     Q = d.inference(MAX_ITER)
#     Q = np.array(Q).reshape((c, h, w))
#     return Q




if __name__=='__main__':

    import torch
    a=nn.ChannelShuffle(1)

    data=torch.tensor([[[[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4],[5,5,5,5],[6,6,6,6],[7,7,7,7],[8,8,8,8]]]])
    print(data.shape)
    print(a(data))