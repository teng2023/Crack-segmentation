import torch
import torch.nn as nn
from model import Res18SqueezeSESCUNet,Res18SqueezeSESCUNet_depth
import numpy as np

num_class=2
init_weight_type='xavier'

#gpu problem
use_gpu=torch.cuda.is_available()
device=torch.device("cuda:0" if use_gpu else "cpu") 

# class VotingModel(nn.Module):
#     def __init__(self,**model_dict):
#         super().__init__()
#         self.model_dict=model_dict
#         self.models=[]
        
#         for i in range(len(self.model_dict['original'])):
#             model=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
#             model.to(device)
#             weight=torch.load(self.model_dict['original'][i])
#             model.load_state_dict(weight['model'],strict=False)
#             self.models.append(model)

#     def forward(self,img1,img2,img3):
#         final_output=np.zeros((400,400))

#         if len(self.model_dict['original']) !=0:
#             for i in range(len(self.model_dict['original'])):
#                 output=self.models[i](img1)
#                 pred=output.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(400,400)
#                 pred=pred.cpu().numpy()
#                 final_output=final_output+pred
        
#         for i in range(400):
#             for j in range(400):
#                 if final_output[i][j]>=2:
#                     final_output[i][j]=1
#                 else:
#                     final_output[i][j]=0

#         return final_output   


class VotingModel_total_3(nn.Module):
    def __init__(self,m1,m2,m3,p1,p2,p3):
        super().__init__()
        if m1=='original':
            self.model1=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif m1=='depth' or m1=='heat':
            self.model1=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if m2=='original':
            self.model2=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif m2=='depth' or m2=='heat':
            self.model2=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if m3=='original':
            self.model3=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif m3=='depth' or m3=='heat':
            self.model3=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if use_gpu:
            self.model1.to(device)
            self.model2.to(device)
            self.model3.to(device)

        self.p1=torch.load(p1)
        self.p2=torch.load(p2)
        self.p3=torch.load(p3)

        self.model1.load_state_dict(self.p1['model'],strict=False)
        self.model2.load_state_dict(self.p2['model'],strict=False)
        self.model3.load_state_dict(self.p3['model'],strict=False)

        self.img_list=[m1,m2,m3]

    def forward(self,img1,img2,img3):

        img=[]

        for m in self.img_list:
            if m=='original':
                img.append(img1)
            elif m=='depth':
                img.append(img2)
            elif m=='heat':
                img.append(img3)


        output1=self.model1(img[0])
        output2=self.model2(img[1])
        output3=self.model3(img[2])

        pred1=output1.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred1=pred1.cpu().numpy()
        pred2=output2.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred2=pred2.cpu().numpy()
        pred3=output3.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred3=pred3.cpu().numpy()

        result=pred1+pred2+pred3
        for i in range(400):
            for j in range(400):
                if result[0][i][j]>=2:
                    result[0][i][j]=1
                else:
                    result[0][i][j]=0
        
        return result

class VotingModel_total_5(nn.Module):
    def __init__(self,m1,m2,m3,m4,m5,p1,p2,p3,p4,p5):
        super().__init__()
        if m1=='original':
            self.model1=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif m1=='depth' or m1=='heat':
            self.model1=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if m2=='original':
            self.model2=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif m2=='depth' or m2=='heat':
            self.model2=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if m3=='original':
            self.model3=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif m3=='depth' or m3=='heat':
            self.model3=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if m4=='original':
            self.model4=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif m4=='depth' or m4=='heat':
            self.model4=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if m5=='original':
            self.model5=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif m5=='depth' or m5=='heat':
            self.model5=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if use_gpu:
            self.model1.to(device)
            self.model2.to(device)
            self.model3.to(device)
            self.model4.to(device)
            self.model5.to(device)

        self.p1=torch.load(p1)
        self.p2=torch.load(p2)
        self.p3=torch.load(p3)
        self.p4=torch.load(p4)
        self.p5=torch.load(p5)

        self.model1.load_state_dict(self.p1['model'],strict=False)
        self.model2.load_state_dict(self.p2['model'],strict=False)
        self.model3.load_state_dict(self.p3['model'],strict=False)
        self.model4.load_state_dict(self.p4['model'],strict=False)
        self.model5.load_state_dict(self.p5['model'],strict=False)

        self.m1=m1
        self.m2=m2
        self.m3=m3
        self.m4=m4
        self.m5=m5

        self.img_list=[m1,m2,m3,m4,m5]

    def forward(self,img1,img2,img3):

        img=[]

        for m in self.img_list:
            if m=='original':
                img.append(img1)
            elif m=='depth':
                img.append(img2)
            elif m=='heat':
                img.append(img3)

        output1=self.model1(img[0])
        output2=self.model2(img[1])
        output3=self.model3(img[2])
        output4=self.model4(img[3])
        output5=self.model5(img[4])

        pred1=output1.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred1=pred1.cpu().numpy()
        pred2=output2.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred2=pred2.cpu().numpy()
        pred3=output3.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred3=pred3.cpu().numpy()
        pred4=output4.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred4=pred4.cpu().numpy()
        pred5=output5.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred5=pred5.cpu().numpy()

        result=pred1+pred2+pred3+pred4+pred5
        for i in range(400):
            for j in range(400):
                if result[0][i][j]>=3:
                    result[0][i][j]=1
                else:
                    result[0][i][j]=0
        
        return result


class VotingModel_total_7(nn.Module):
    def __init__(self,m1,m2,m3,m4,m5,m6,m7,p1,p2,p3,p4,p5,p6,p7):
        super().__init__()
        if m1=='original':
            self.model1=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif m1=='depth' or m1=='heat':
            self.model1=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if m2=='original':
            self.model2=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif m2=='depth' or m2=='heat':
            self.model2=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if m3=='original':
            self.model3=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif m3=='depth' or m3=='heat':
            self.model3=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if m4=='original':
            self.model4=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif m4=='depth' or m4=='heat':
            self.model4=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if m5=='original':
            self.model5=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif m5=='depth' or m5=='heat':
            self.model5=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if m6=='original':
            self.model6=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif m6=='depth' or m6=='heat':
            self.model6=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if m7=='original':
            self.model7=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif m7=='depth' or m7=='heat':
            self.model7=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if use_gpu:
            self.model1.to(device)
            self.model2.to(device)
            self.model3.to(device)
            self.model4.to(device)
            self.model5.to(device)
            self.model6.to(device)
            self.model7.to(device)

        self.p1=torch.load(p1)
        self.p2=torch.load(p2)
        self.p3=torch.load(p3)
        self.p4=torch.load(p4)
        self.p5=torch.load(p5)
        self.p6=torch.load(p6)
        self.p7=torch.load(p7)

        self.model1.load_state_dict(self.p1['model'],strict=False)
        self.model2.load_state_dict(self.p2['model'],strict=False)
        self.model3.load_state_dict(self.p3['model'],strict=False)
        self.model4.load_state_dict(self.p4['model'],strict=False)
        self.model5.load_state_dict(self.p5['model'],strict=False)
        self.model6.load_state_dict(self.p6['model'],strict=False)
        self.model7.load_state_dict(self.p7['model'],strict=False)

        self.m1=m1
        self.m2=m2
        self.m3=m3
        self.m4=m4
        self.m5=m5
        self.m6=m6
        self.m7=m7
        self.img_list=[m1,m2,m3,m4,m5,m6,m7]

    def forward(self,img1,img2,img3):

        img=[]

        for m in self.img_list:
            if m=='original':
                img.append(img1)
            elif m=='depth':
                img.append(img2)
            elif m=='heat':
                img.append(img3)


        output1=self.model1(img[0])
        output2=self.model2(img[1])
        output3=self.model3(img[2])
        output4=self.model4(img[3])
        output5=self.model5(img[4])
        output6=self.model6(img[5])
        output7=self.model7(img[6])

        pred1=output1.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred1=pred1.cpu().numpy()
        pred2=output2.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred2=pred2.cpu().numpy()
        pred3=output3.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred3=pred3.cpu().numpy()
        pred4=output4.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred4=pred4.cpu().numpy()
        pred5=output5.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred5=pred5.cpu().numpy()
        pred6=output6.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred6=pred6.cpu().numpy()
        pred7=output7.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred7=pred7.cpu().numpy()

        result=pred1+pred2+pred3+pred4+pred5+pred6+pred7
        for i in range(400):
            for j in range(400):
                if result[0][i][j]>=4:
                    result[0][i][j]=1
                else:
                    result[0][i][j]=0
        
        return result


class VotingModel_1_1_3(nn.Module):
    def __init__(self,m1,m2,m3,m4,m5,p1,p2,p3,p4,p5):
        super().__init__()
        if m1=='original':
            self.model1=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif m1=='depth' or m1=='heat':
            self.model1=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if m2=='original':
            self.model2=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif m2=='depth' or m2=='heat':
            self.model2=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if m3=='original':
            self.model3=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif m3=='depth' or m3=='heat':
            self.model3=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if m4=='original':
            self.model4=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif m4=='depth' or m4=='heat':
            self.model4=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if m5=='original':
            self.model5=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif m5=='depth' or m5=='heat':
            self.model5=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if use_gpu:
            self.model1.to(device)
            self.model2.to(device)
            self.model3.to(device)
            self.model4.to(device)
            self.model5.to(device)

        self.p1=torch.load(p1)
        self.p2=torch.load(p2)
        self.p3=torch.load(p3)
        self.p4=torch.load(p4)
        self.p5=torch.load(p5)

        self.model1.load_state_dict(self.p1['model'],strict=False)
        self.model2.load_state_dict(self.p2['model'],strict=False)
        self.model3.load_state_dict(self.p3['model'],strict=False)
        self.model4.load_state_dict(self.p4['model'],strict=False)
        self.model5.load_state_dict(self.p5['model'],strict=False)

    def forward(self,img1,img2,img3):
        output1=self.model1(img1)
        output2=self.model2(img2)
        output3=self.model3(img3)
        output4=self.model4(img3)
        output5=self.model5(img3)

        pred1=output1.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred1=pred1.cpu().numpy()
        pred2=output2.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred2=pred2.cpu().numpy()
        pred3=output3.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred3=pred3.cpu().numpy()
        pred4=output4.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred4=pred4.cpu().numpy()
        pred5=output5.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred5=pred5.cpu().numpy()

        result=pred1+pred2+pred3+pred4+pred5
        for i in range(400):
            for j in range(400):
                if result[0][i][j]>=3:
                    result[0][i][j]=1
                else:
                    result[0][i][j]=0
        
        return result

class VotingModel_1_3_1(nn.Module):
    def __init__(self,m1,m2,m3,m4,m5,p1,p2,p3,p4,p5):
        super().__init__()
        if m1=='original':
            self.model1=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif m1=='depth' or m1=='heat':
            self.model1=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if m2=='original':
            self.model2=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif m2=='depth' or m2=='heat':
            self.model2=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if m3=='original':
            self.model3=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif m3=='depth' or m3=='heat':
            self.model3=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if m4=='original':
            self.model4=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif m4=='depth' or m4=='heat':
            self.model4=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if m5=='original':
            self.model5=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif m5=='depth' or m5=='heat':
            self.model5=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if use_gpu:
            self.model1.to(device)
            self.model2.to(device)
            self.model3.to(device)
            self.model4.to(device)
            self.model5.to(device)

        self.p1=torch.load(p1)
        self.p2=torch.load(p2)
        self.p3=torch.load(p3)
        self.p4=torch.load(p4)
        self.p5=torch.load(p5)

        self.model1.load_state_dict(self.p1['model'],strict=False)
        self.model2.load_state_dict(self.p2['model'],strict=False)
        self.model3.load_state_dict(self.p3['model'],strict=False)
        self.model4.load_state_dict(self.p4['model'],strict=False)
        self.model5.load_state_dict(self.p5['model'],strict=False)

    def forward(self,img1,img2,img3):
        output1=self.model1(img1)
        output2=self.model2(img2)
        output3=self.model3(img2)
        output4=self.model4(img2)
        output5=self.model5(img3)

        pred1=output1.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred1=pred1.cpu().numpy()
        pred2=output2.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred2=pred2.cpu().numpy()
        pred3=output3.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred3=pred3.cpu().numpy()
        pred4=output4.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred4=pred4.cpu().numpy()
        pred5=output5.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred5=pred5.cpu().numpy()

        result=pred1+pred2+pred3+pred4+pred5
        for i in range(400):
            for j in range(400):
                if result[0][i][j]>=3:
                    result[0][i][j]=1
                else:
                    result[0][i][j]=0
        
        return result

class VotingModel_1_2(nn.Module):
    def __init__(self,path_o1,path_o2,path_d1):
        super().__init__()
        # voting models
        self.model_o1=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        self.model_o2=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)
        self.model_d1=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        # load weights
        self.weight_o1=torch.load(path_o1)
        self.weight_o2=torch.load(path_o2)
        self.weight_d1=torch.load(path_d1)

        self.model_o1.load_state_dict(self.weight_o1['model'],strict=False)
        self.model_o2.load_state_dict(self.weight_o2['model'],strict=False)
        self.model_d1.load_state_dict(self.weight_d1['model'],strict=False)

    def forward(self,origin_img,depth_img_gray):
        # send input images to corresponding models
        output_o1=self.model_o1(origin_img)
        output_o2=self.model_o2(depth_img_gray)
        output_d1=self.model_d1(depth_img_gray)

        # post-processing output (1 channel image)
        pred_o1=output_o1.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred_o1=pred_o1.cpu().numpy()
        pred_o2=output_o2.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred_o2=pred_o2.cpu().numpy()
        pred_d1=output_d1.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred_d1=pred_d1.cpu().numpy()

        result=pred_o1+pred_o2+pred_d1
        for i in range(400):
            for j in range(400):
                if result[0][i][j]>=2:
                    result[0][i][j]=1
                else:
                    result[0][i][j]=0
        
        return result


class VotingModel_2_3(nn.Module):
    def __init__(self,m1,m2,m3,m4,m5,p1,p2,p3,p4,p5):
        super().__init__()
        if m1=='original':
            self.model1=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif m1=='depth' or m1=='heat':
            self.model1=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if m2=='original':
            self.model2=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif m2=='depth' or m2=='heat':
            self.model2=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if m3=='original':
            self.model3=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif m3=='depth' or m3=='heat':
            self.model3=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if m4=='original':
            self.model4=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif m4=='depth' or m4=='heat':
            self.model4=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if m5=='original':
            self.model5=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif m5=='depth' or m5=='heat':
            self.model5=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if use_gpu:
            self.model1.to(device)
            self.model2.to(device)
            self.model3.to(device)
            self.model4.to(device)
            self.model5.to(device)

        self.p1=torch.load(p1)
        self.p2=torch.load(p2)
        self.p3=torch.load(p3)
        self.p4=torch.load(p4)
        self.p5=torch.load(p5)

        self.model1.load_state_dict(self.p1['model'],strict=False)
        self.model2.load_state_dict(self.p2['model'],strict=False)
        self.model3.load_state_dict(self.p3['model'],strict=False)
        self.model4.load_state_dict(self.p4['model'],strict=False)
        self.model5.load_state_dict(self.p5['model'],strict=False)

    def forward(self,img1,img2):
        output1=self.model1(img1)
        output2=self.model2(img1)
        output3=self.model3(img2)
        output4=self.model4(img2)
        output5=self.model5(img2)

        pred1=output1.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred1=pred1.cpu().numpy()
        pred2=output2.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred2=pred2.cpu().numpy()
        pred3=output3.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred3=pred3.cpu().numpy()
        pred4=output4.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred4=pred4.cpu().numpy()
        pred5=output5.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred5=pred5.cpu().numpy()

        result=pred1+pred2+pred3+pred4+pred5
        for i in range(400):
            for j in range(400):
                if result[0][i][j]>=3:
                    result[0][i][j]=1
                else:
                    result[0][i][j]=0
        
        return result


class VotingModel_3_3_1(nn.Module):
    def __init__(self,m1,m2,m3,m4,m5,m6,m7,p1,p2,p3,p4,p5,p6,p7):
        super().__init__()
        if m1=='original':
            self.model1=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif m1=='depth' or m1=='heat':
            self.model1=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if m2=='original':
            self.model2=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif m2=='depth' or m2=='heat':
            self.model2=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if m3=='original':
            self.model3=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif m3=='depth' or m3=='heat':
            self.model3=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if m4=='original':
            self.model4=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif m4=='depth' or m4=='heat':
            self.model4=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if m5=='original':
            self.model5=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif m5=='depth' or m5=='heat':
            self.model5=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if m6=='original':
            self.model6=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif m6=='depth' or m6=='heat':
            self.model6=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if m7=='original':
            self.model7=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif m7=='depth' or m7=='heat':
            self.model7=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if use_gpu:
            self.model1.to(device)
            self.model2.to(device)
            self.model3.to(device)
            self.model4.to(device)
            self.model5.to(device)
            self.model6.to(device)
            self.model7.to(device)

        self.p1=torch.load(p1)
        self.p2=torch.load(p2)
        self.p3=torch.load(p3)
        self.p4=torch.load(p4)
        self.p5=torch.load(p5)
        self.p6=torch.load(p6)
        self.p7=torch.load(p7)

        self.model1.load_state_dict(self.p1['model'],strict=False)
        self.model2.load_state_dict(self.p2['model'],strict=False)
        self.model3.load_state_dict(self.p3['model'],strict=False)
        self.model4.load_state_dict(self.p4['model'],strict=False)
        self.model5.load_state_dict(self.p5['model'],strict=False)
        self.model6.load_state_dict(self.p6['model'],strict=False)
        self.model7.load_state_dict(self.p7['model'],strict=False)

    def forward(self,img1,img2,img3):
        output1=self.model1(img1)
        output2=self.model2(img1)
        output3=self.model3(img1)
        output4=self.model4(img2)
        output5=self.model5(img2)
        output6=self.model6(img2)
        output7=self.model7(img3)

        pred1=output1.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred1=pred1.cpu().numpy()
        pred2=output2.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred2=pred2.cpu().numpy()
        pred3=output3.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred3=pred3.cpu().numpy()
        pred4=output4.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred4=pred4.cpu().numpy()
        pred5=output5.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred5=pred5.cpu().numpy()
        pred6=output6.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred6=pred6.cpu().numpy()
        pred7=output7.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred7=pred7.cpu().numpy()

        result=pred1+pred2+pred3+pred4+pred5+pred6+pred7
        for i in range(400):
            for j in range(400):
                if result[0][i][j]>=4:
                    result[0][i][j]=1
                else:
                    result[0][i][j]=0
        
        return result

class VotingModel_3_1_1(nn.Module):
    def __init__(self,m1,m2,m3,m4,m5,p1,p2,p3,p4,p5):
        super().__init__()
        if m1=='original':
            self.model1=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif m1=='depth' or m1=='heat':
            self.model1=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if m2=='original':
            self.model2=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif m2=='depth' or m2=='heat':
            self.model2=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if m3=='original':
            self.model3=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif m3=='depth' or m3=='heat':
            self.model3=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if m4=='original':
            self.model4=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif m4=='depth' or m4=='heat':
            self.model4=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if m5=='original':
            self.model5=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif m5=='depth' or m5=='heat':
            self.model5=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)
        
        if use_gpu:
            self.model1.to(device)
            self.model2.to(device)
            self.model3.to(device)
            self.model4.to(device)
            self.model5.to(device)

        self.p1=torch.load(p1)
        self.p2=torch.load(p2)
        self.p3=torch.load(p3)
        self.p4=torch.load(p4)
        self.p5=torch.load(p5)

        self.model1.load_state_dict(self.p1['model'],strict=False)
        self.model2.load_state_dict(self.p2['model'],strict=False)
        self.model3.load_state_dict(self.p3['model'],strict=False)
        self.model4.load_state_dict(self.p4['model'],strict=False)
        self.model5.load_state_dict(self.p5['model'],strict=False)

    def forward(self,img1,img2,img3):
        output1=self.model1(img1)
        output2=self.model2(img1)
        output3=self.model3(img1)
        output4=self.model4(img2)
        output5=self.model5(img2)


        pred1=output1.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred1=pred1.cpu().numpy()
        pred2=output2.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred2=pred2.cpu().numpy()
        pred3=output3.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred3=pred3.cpu().numpy()
        pred4=output4.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred4=pred4.cpu().numpy()
        pred5=output5.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred5=pred5.cpu().numpy()


        result=pred1+pred2+pred3+pred4+pred5
        for i in range(400):
            for j in range(400):
                if result[0][i][j]>=3:
                    result[0][i][j]=1
                else:
                    result[0][i][j]=0
        
        return result

# new method
# class VotingModel_3_3_3(nn.Module):
#     def __init__(self,m1,m2,m3,m4,m5,m6,m7,m8,m9,p1,p2,p3,p4,p5,p6,p7,p8,p9):
#         super().__init__()
#         if m1=='original':
#             self.model1=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
#         elif m1=='depth' or m1=='heat':
#             self.model1=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

#         if m2=='original':
#             self.model2=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
#         elif m2=='depth' or m2=='heat':
#             self.model2=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

#         if m3=='original':
#             self.model3=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
#         elif m3=='depth' or m3=='heat':
#             self.model3=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

#         if m4=='original':
#             self.model4=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
#         elif m4=='depth' or m4=='heat':
#             self.model4=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

#         if m5=='original':
#             self.model5=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
#         elif m5=='depth' or m5=='heat':
#             self.model5=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

#         if m6=='original':
#             self.model6=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
#         elif m6=='depth' or m6=='heat':
#             self.model6=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

#         if m7=='original':
#             self.model7=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
#         elif m7=='depth' or m7=='heat':
#             self.model7=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

#         if m8=='original':
#             self.model8=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
#         elif m8=='depth' or m8=='heat':
#             self.model8=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

#         if m9=='original':
#             self.model9=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
#         elif m9=='depth' or m9=='heat':
#             self.model9=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

#         # if use_gpu:
#         #     self.model1.to(device)
#         #     self.model2.to(device)
#         #     self.model3.to(device)
#         #     self.model4.to(device)
#         #     self.model5.to(device)
#         #     self.model6.to(device)
#         #     self.model7.to(device)
#         #     self.model8.to(device)
#         #     self.model9.to(device)

#         self.p1=torch.load(p1)
#         self.p2=torch.load(p2)
#         self.p3=torch.load(p3)
#         self.p4=torch.load(p4)
#         self.p5=torch.load(p5)
#         self.p6=torch.load(p6)
#         self.p7=torch.load(p7)
#         self.p8=torch.load(p8)
#         self.p9=torch.load(p9)

#         self.model1.load_state_dict(self.p1['model'],strict=False)
#         self.model2.load_state_dict(self.p2['model'],strict=False)
#         self.model3.load_state_dict(self.p3['model'],strict=False)
#         self.model4.load_state_dict(self.p4['model'],strict=False)
#         self.model5.load_state_dict(self.p5['model'],strict=False)
#         self.model6.load_state_dict(self.p6['model'],strict=False)
#         self.model7.load_state_dict(self.p7['model'],strict=False)
#         self.model8.load_state_dict(self.p8['model'],strict=False)
#         self.model9.load_state_dict(self.p9['model'],strict=False)

#         self.softmax=nn.Softmax()

#     def forward(self,img1,img2,img3):
#         output1=self.model1(img1)
#         output2=self.model2(img1)
#         output3=self.model3(img1)
#         output4=self.model4(img2)
#         output5=self.model5(img2)
#         output6=self.model6(img2)
#         output7=self.model7(img3)
#         output8=self.model8(img3)
#         output9=self.model9(img3)

#         output1=self.softmax(output1)
#         output2=self.softmax(output2)
#         output3=self.softmax(output3)
#         output4=self.softmax(output4)
#         output5=self.softmax(output5)
#         output6=self.softmax(output6)
#         output7=self.softmax(output7)
#         output8=self.softmax(output8)
#         output9=self.softmax(output9)

#         pred1=output1.reshape(2,400,400)
#         pred1=pred1.cpu().numpy()
#         pred2=output2.reshape(2,400,400)
#         pred2=pred2.cpu().numpy()
#         pred3=output3.reshape(2,400,400)
#         pred3=pred3.cpu().numpy()
#         pred4=output4.reshape(2,400,400)
#         pred4=pred4.cpu().numpy()
#         pred5=output5.reshape(2,400,400)
#         pred5=pred5.cpu().numpy()
#         pred6=output6.reshape(2,400,400)
#         pred6=pred6.cpu().numpy()
#         pred7=output7.reshape(2,400,400)
#         pred7=pred7.cpu().numpy()
#         pred8=output8.reshape(2,400,400)
#         pred8=pred8.cpu().numpy()
#         pred9=output9.reshape(2,400,400)
#         pred9=pred9.cpu().numpy()

#         result_sum=pred1+pred2+pred3+pred4+pred5+pred6+pred7+pred8+pred9
#         # first stage
#         crack_result=np.zeros((400,400))
#         for i in range(400):
#             for j in range(400):
#                 if result_sum[1][i][j]>=0.55*9:
#                     crack_result[i][j]=1
#                 else:
#                     crack_result[i][j]=0
        
#         # second stage
#         background_result=np.zeros((400,400))
#         for i in range(400):
#             for j in range(400):
#                 if result_sum[0][i][j]>=0.8*9:
#                     background_result[i][j]=1
#                 else:
#                     background_result[i][j]=0

#         # third stage
#         majority_voting_part=crack_result+background_result

#         pred1=pred1.argmax(axis=0).reshape(1,400,400)
#         pred2=pred2.argmax(axis=0).reshape(1,400,400)
#         pred3=pred3.argmax(axis=0).reshape(1,400,400)
#         pred4=pred4.argmax(axis=0).reshape(1,400,400)
#         pred5=pred5.argmax(axis=0).reshape(1,400,400)
#         pred6=pred6.argmax(axis=0).reshape(1,400,400)
#         pred7=pred7.argmax(axis=0).reshape(1,400,400)
#         pred8=pred8.argmax(axis=0).reshape(1,400,400)
#         pred9=pred9.argmax(axis=0).reshape(1,400,400)

#         result=pred1+pred2+pred3+pred4+pred5+pred6+pred7+pred8+pred9

#         for i in range(400):
#             for j in range(400):
#                 if majority_voting_part[i][j]==0:
#                     if result[0][i][j]>=5:
#                         result[0][i][j]=1
#                     else:
#                         result[0][i][j]=0

#                 else:
#                     if background_result[i][j]==1:
#                         result[0][i][j]=0
#                     elif crack_result[i][j]==1:
#                         result[0][i][j]=1

#         return result   #shape=(1,400,400)

# new mwthods (second version)
class VotingModel_3_3_3(nn.Module):
    def __init__(self,m1,m2,m3,m4,m5,m6,m7,m8,m9,p1,p2,p3,p4,p5,p6,p7,p8,p9):
        super().__init__()
        if m1=='original':
            self.model1=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif m1=='depth' or m1=='heat':
            self.model1=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if m2=='original':
            self.model2=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif m2=='depth' or m2=='heat':
            self.model2=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if m3=='original':
            self.model3=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif m3=='depth' or m3=='heat':
            self.model3=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if m4=='original':
            self.model4=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif m4=='depth' or m4=='heat':
            self.model4=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if m5=='original':
            self.model5=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif m5=='depth' or m5=='heat':
            self.model5=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if m6=='original':
            self.model6=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif m6=='depth' or m6=='heat':
            self.model6=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if m7=='original':
            self.model7=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif m7=='depth' or m7=='heat':
            self.model7=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if m8=='original':
            self.model8=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif m8=='depth' or m8=='heat':
            self.model8=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if m9=='original':
            self.model9=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif m9=='depth' or m9=='heat':
            self.model9=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        # if use_gpu:
        #     self.model1.to(device)
        #     self.model2.to(device)
        #     self.model3.to(device)
        #     self.model4.to(device)
        #     self.model5.to(device)
        #     self.model6.to(device)
        #     self.model7.to(device)
        #     self.model8.to(device)
        #     self.model9.to(device)

        self.p1=torch.load(p1)
        self.p2=torch.load(p2)
        self.p3=torch.load(p3)
        self.p4=torch.load(p4)
        self.p5=torch.load(p5)
        self.p6=torch.load(p6)
        self.p7=torch.load(p7)
        self.p8=torch.load(p8)
        self.p9=torch.load(p9)

        self.model1.load_state_dict(self.p1['model'],strict=False)
        self.model2.load_state_dict(self.p2['model'],strict=False)
        self.model3.load_state_dict(self.p3['model'],strict=False)
        self.model4.load_state_dict(self.p4['model'],strict=False)
        self.model5.load_state_dict(self.p5['model'],strict=False)
        self.model6.load_state_dict(self.p6['model'],strict=False)
        self.model7.load_state_dict(self.p7['model'],strict=False)
        self.model8.load_state_dict(self.p8['model'],strict=False)
        self.model9.load_state_dict(self.p9['model'],strict=False)

        # self.softmax=nn.Softmax()

    def forward(self,img1,img2,img3):
        output1=self.model1(img1)
        output2=self.model2(img1)
        output3=self.model3(img1)
        output4=self.model4(img2)
        output5=self.model5(img2)
        output6=self.model6(img2)
        output7=self.model7(img3)
        output8=self.model8(img3)
        output9=self.model9(img3)

        pred1=output1.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred1=pred1.cpu().numpy()
        pred2=output2.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred2=pred2.cpu().numpy()
        pred3=output3.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred3=pred3.cpu().numpy()
        pred4=output4.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred4=pred4.cpu().numpy()
        pred5=output5.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred5=pred5.cpu().numpy()
        pred6=output6.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred6=pred6.cpu().numpy()
        pred7=output7.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred7=pred7.cpu().numpy()
        pred8=output8.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred8=pred8.cpu().numpy()
        pred9=output9.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred9=pred9.cpu().numpy()

        result_sum1=pred1+pred2+pred3
        # first stage
        result_mask1=np.zeros((1,400,400))
        for i in range(400):
            for j in range(400):
                if result_sum1[0][i][j]==3:
                    result_mask1[0][i][j]=1
                else:
                    result_mask1[0][i][j]=0
        
        result_sum2=pred4+pred5+pred6
        # second stage
        result_mask2=np.zeros((1,400,400))
        for i in range(400):
            for j in range(400):
                if result_mask1[0][i][j]==1:
                    continue
                if result_sum2[0][i][j]==3:
                    result_mask2[0][i][j]=1
                else:
                    result_mask2[0][i][j]=0

        # third stage
        majority_voting_part=result_mask1+result_mask2

        result=pred1+pred2+pred3+pred4+pred5+pred6+pred7+pred8+pred9

        for i in range(400):
            for j in range(400):
                if majority_voting_part[0][i][j]==1:
                    continue
                if result[0][i][j]>4:
                    majority_voting_part[0][i][j]=1
                else:
                    majority_voting_part[0][i][j]=0

        return majority_voting_part   #shape=(1,400,400)

# weight
# class VotingModel_3_3_3(nn.Module):
#     def __init__(self,m1,m2,m3,m4,m5,m6,m7,m8,m9,p1,p2,p3,p4,p5,p6,p7,p8,p9):
#         super().__init__()
#         if m1=='original':
#             self.model1=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
#         elif m1=='depth' or m1=='heat':
#             self.model1=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

#         if m2=='original':
#             self.model2=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
#         elif m2=='depth' or m2=='heat':
#             self.model2=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

#         if m3=='original':
#             self.model3=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
#         elif m3=='depth' or m3=='heat':
#             self.model3=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

#         if m4=='original':
#             self.model4=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
#         elif m4=='depth' or m4=='heat':
#             self.model4=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

#         if m5=='original':
#             self.model5=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
#         elif m5=='depth' or m5=='heat':
#             self.model5=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

#         if m6=='original':
#             self.model6=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
#         elif m6=='depth' or m6=='heat':
#             self.model6=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

#         if m7=='original':
#             self.model7=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
#         elif m7=='depth' or m7=='heat':
#             self.model7=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

#         if m8=='original':
#             self.model8=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
#         elif m8=='depth' or m8=='heat':
#             self.model8=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

#         if m9=='original':
#             self.model9=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
#         elif m9=='depth' or m9=='heat':
#             self.model9=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

#         # if use_gpu:
#         #     self.model1.to(device)
#         #     self.model2.to(device)
#         #     self.model3.to(device)
#         #     self.model4.to(device)
#         #     self.model5.to(device)
#         #     self.model6.to(device)
#         #     self.model7.to(device)
#         #     self.model8.to(device)
#         #     self.model9.to(device)

#         self.p1=torch.load(p1)
#         self.p2=torch.load(p2)
#         self.p3=torch.load(p3)
#         self.p4=torch.load(p4)
#         self.p5=torch.load(p5)
#         self.p6=torch.load(p6)
#         self.p7=torch.load(p7)
#         self.p8=torch.load(p8)
#         self.p9=torch.load(p9)

#         self.model1.load_state_dict(self.p1['model'],strict=False)
#         self.model2.load_state_dict(self.p2['model'],strict=False)
#         self.model3.load_state_dict(self.p3['model'],strict=False)
#         self.model4.load_state_dict(self.p4['model'],strict=False)
#         self.model5.load_state_dict(self.p5['model'],strict=False)
#         self.model6.load_state_dict(self.p6['model'],strict=False)
#         self.model7.load_state_dict(self.p7['model'],strict=False)
#         self.model8.load_state_dict(self.p8['model'],strict=False)
#         self.model9.load_state_dict(self.p9['model'],strict=False)

#     def forward(self,img1,img2,img3):
#         output1=self.model1(img1)
#         output2=self.model2(img1)
#         output3=self.model3(img1)
#         output4=self.model4(img2)
#         output5=self.model5(img2)
#         output6=self.model6(img2)
#         output7=self.model7(img3)
#         output8=self.model8(img3)
#         output9=self.model9(img3)

#         pred1=output1.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
#         pred1=pred1.cpu().numpy()
#         pred2=output2.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
#         pred2=pred2.cpu().numpy()
#         pred3=output3.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
#         pred3=pred3.cpu().numpy()
#         pred4=output4.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
#         pred4=pred4.cpu().numpy()
#         pred5=output5.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
#         pred5=pred5.cpu().numpy()
#         pred6=output6.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
#         pred6=pred6.cpu().numpy()
#         pred7=output7.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
#         pred7=pred7.cpu().numpy()
#         pred8=output8.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
#         pred8=pred8.cpu().numpy()
#         pred9=output9.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
#         pred9=pred9.cpu().numpy()

#         result=(pred1+pred2+pred3+pred4+pred5+pred6)*1.2+(pred7+pred8+pred9)*0.6

#         for i in range(400):
#             for j in range(400):
#                 if result[0][i][j]>4.5:
#                     result[0][i][j]=1
#                 else:
#                     result[0][i][j]=0
        
#         return result
    

class VotingModel_1_1_1(nn.Module):
    def __init__(self,mode1,mode2,mode3,path1,path2,path3):
        super().__init__()
        if mode1=='original':
            self.model1=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif mode1=='heat' or mode1=='depth':
            self.model1=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if mode2=='original':
            self.model2=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif mode2=='heat' or mode2=='depth':
            self.model2=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if mode3=='original':
            self.model3=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        elif mode3=='heat' or mode3=='depth':
            self.model3=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        if use_gpu:
            self.model1.to(device)
            self.model2.to(device)
            self.model3.to(device)
        
        self.weight1=torch.load(path1)
        self.weight2=torch.load(path2)
        self.weight3=torch.load(path3)

        self.model1.load_state_dict(self.weight1['model'],strict=False)
        self.model2.load_state_dict(self.weight2['model'],strict=False)
        self.model3.load_state_dict(self.weight3['model'],strict=False)

    def forward(self,img1,img2,img3):
        output1=self.model1(img1)
        output2=self.model2(img2)
        output3=self.model3(img3)

        pred1=output1.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred1=pred1.cpu().numpy()
        pred2=output2.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred2=pred2.cpu().numpy()
        pred3=output3.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred3=pred3.cpu().numpy()

        result=pred1+pred2+pred3
        for i in range(400):
            for j in range(400):
                if result[0][i][j]>=2:
                    result[0][i][j]=1
                else:
                    result[0][i][j]=0
        
        return result

class VotingModel_2Origin_1Depth(nn.Module):
    def __init__(self,path_o1,path_o2,path_d1):
        super().__init__()
        # voting models
        self.model_o1=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        self.model_o2=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        self.model_d1=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        # load weights
        self.weight_o1=torch.load(path_o1)
        self.weight_o2=torch.load(path_o2)
        self.weight_d1=torch.load(path_d1)

        self.model_o1.load_state_dict(self.weight_o1['model'],strict=False)
        self.model_o2.load_state_dict(self.weight_o2['model'],strict=False)
        self.model_d1.load_state_dict(self.weight_d1['model'],strict=False)

    def forward(self,origin_img,depth_img_gray):
        # send input images to corresponding models
        output_o1=self.model_o1(origin_img)
        output_o2=self.model_o2(origin_img)
        output_d1=self.model_d1(depth_img_gray)

        # post-processing output (1 channel image)
        pred_o1=output_o1.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred_o1=pred_o1.cpu().numpy()
        pred_o2=output_o2.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred_o2=pred_o2.cpu().numpy()
        pred_d1=output_d1.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred_d1=pred_d1.cpu().numpy()

        result=pred_o1+pred_o2+pred_d1
        for i in range(400):
            for j in range(400):
                if result[0][i][j]>=2:
                    result[0][i][j]=1
                else:
                    result[0][i][j]=0
        
        return result
    
class VotingModel_3Origin_2Depth(nn.Module):
    def __init__(self,path_o1,path_o2,path_o3,path_d1,path_d2):
        super().__init__()
        # voting models
        self.model_o1=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        self.model_o2=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        self.model_o3=Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type)
        self.model_d1=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)
        self.model_d2=Res18SqueezeSESCUNet_depth(num_class=num_class,init_weight_type=init_weight_type)

        # load weights
        self.weight_o1=torch.load(path_o1)
        self.weight_o2=torch.load(path_o2)
        self.weight_o3=torch.load(path_o3)
        self.weight_d1=torch.load(path_d1)
        self.weight_d2=torch.load(path_d2)

        self.model_o1.load_state_dict(self.weight_o1['model'],strict=False)
        self.model_o2.load_state_dict(self.weight_o2['model'],strict=False)
        self.model_o3.load_state_dict(self.weight_o3['model'],strict=False)
        self.model_d1.load_state_dict(self.weight_d1['model'],strict=False)
        self.model_d2.load_state_dict(self.weight_d2['model'],strict=False)

    def forward(self,origin_img,depth_img_gray):
        # send input images to corresponding models
        output_o1=self.model_o1(origin_img)
        output_o2=self.model_o2(origin_img)
        output_o3=self.model_o3(origin_img)
        output_d1=self.model_d1(depth_img_gray)
        output_d2=self.model_d2(depth_img_gray)

        # post-processing output (1 channel image)
        pred_o1=output_o1.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred_o1=pred_o1.cpu().numpy()
        pred_o2=output_o2.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred_o2=pred_o2.cpu().numpy()
        pred_o3=output_o3.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred_o3=pred_o3.cpu().numpy()
        pred_d1=output_d1.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred_d1=pred_d1.cpu().numpy()
        pred_d2=output_d2.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
        pred_d2=pred_d2.cpu().numpy()

        result=pred_o1+pred_o2+pred_o3+pred_d1+pred_d2
        for i in range(400):
            for j in range(400):
                if result[0][i][j]>=3:
                    result[0][i][j]=1
                else:
                    result[0][i][j]=0
        
        return result
    