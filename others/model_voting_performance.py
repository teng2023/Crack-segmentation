if '__file__' in globals(): 
    import sys,os
    sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from model import Res18SqueezeSESCUNet,Res18SqueezeSESCUNet_depth
from model_voting import VotingModel_2Origin_1Depth,VotingModel_3Origin_2Depth,VotingModel_1_1_1,VotingModel_3_3_3,VotingModel_3_1_1,VotingModel_3_3_1,\
    VotingModel_2_3,VotingModel_1_2,VotingModel_1_3_1,VotingModel_1_1_3,VotingModel_total_7,VotingModel_total_5,VotingModel_total_3
from depth_dataloader import DepthDataset

score_dir='crack_model_test/score(custom)/model_voting(test_all)'
if not os.path.exists(score_dir):
    os.makedirs(score_dir)
score_dir='crack_model_test/score(custom)/model_voting(test_bright)'
if not os.path.exists(score_dir):
    os.makedirs(score_dir)
score_dir='crack_model_test/score(custom)/model_voting(test_dark)'
if not os.path.exists(score_dir):
    os.makedirs(score_dir)

#################################################################################################################
# original
path1='crack_model_test/check points(custom)/resnet18+squeeze_se_sc+unet(original)4/Epoch99_of_500.pth'
path2='crack_model_test/check points(custom)/resnet18+squeeze_se_sc+unet(original)2/Epoch166_of_500.pth'
path3='crack_model_test/check points(custom)/resnet18+squeeze_se_sc+unet(original)3/Epoch61_of_500.pth'
# depth
path4='crack_model_test/check points(custom)/resnet18+squeeze_se_sc+unet(depth)2/Epoch145_of_500.pth'
path5='crack_model_test/check points(custom)/resnet18+squeeze_se_sc+unet(depth)3/Epoch69_of_500.pth'
path6='crack_model_test/check points(custom)/resnet18+squeeze_se_sc+unet(depth)4/Epoch176_of_500.pth'
#heat (based on crack IoU)
path7='crack_model_test/check points(custom)/resnet18+squeeze_se_sc+unet(heat)4/Epoch27_of_500.pth'
path8='crack_model_test/check points(custom)/resnet18+squeeze_se_sc+unet(heat)11/Epoch4_of_500.pth'
path9='crack_model_test/check points(custom)/resnet18+squeeze_se_sc+unet(heat)9/Epoch43_of_500.pth'
#heat (based on mean IoU)
# path7='crack_model_test/check points(custom)/resnet18+squeeze_se_sc+unet(heat)3/Epoch190_of_500.pth'
# path8='crack_model_test/check points(custom)/resnet18+squeeze_se_sc+unet(heat)4/Epoch215_of_500.pth'
# path9='crack_model_test/check points(custom)/resnet18+squeeze_se_sc+unet(heat)2/Epoch107_of_500.pth'

voting_member_dict={1:'original',2:'depth',3:'heat',
                    4:'2:1',8:'2:1(heat)',6:'1:1:1',16:'1:2',17:'1:2(heat)',22:'2:1(depth and heat)',18:'3(depth)',10:'3(original)',21:'3(heat)',25:'1:2(depth and heat)',
                    5:'3:2',9:'3:2(heat)',14:'2:3',15:'2:3(heat)',11:'3:1:1',19:'1:3:1',20:'1:1:3',26:'3:2(depth and heat)',27:'2:3(depth and heat)',28:'2:2:1',29:'2:1:2',30:'1:2:2',
                    12:'3:3:1',23:'1:3:3',24:'3:1:3',31:'3:2:2',32:'2:2:3',33:'2:3:2',
                    7:'3:3:3',
                    13:'voting',
                    }
test_mode_list=['test_all','test_bright','test_dark']

voting_member=voting_member_dict[12]
test_mode=test_mode_list[1]

#################################################################################################################

if voting_member=='2:1':
    path_dict={'orginal 1':path1,'original 2':path2,'depth 1':path4}
    model=VotingModel_2Origin_1Depth(path1,path2,path4)
elif voting_member=='2:1(depth and heat)':
    path_dict={'depth 1':path4,'depth 2':path5,'depth 3':path6}
    model=VotingModel_1_1_1('depth','depth','heat',path4,path5,path7)
elif voting_member=='1:3:3':
    path_dict={'original 1':path1,'depth 1':path4,'depth 2':path5,'depth 3':path6,'heat 1':path7,'heat 2':path8,'heat 3':path9}
    model=VotingModel_total_7('original','depth','depth','depth','heat','heat','heat',path1,path4,path5,path6,path7,path8,path9)
elif voting_member=='3:1:3':
    path_dict={'original 1':path1,'original 2':path2,'original 3':path3,'depth 1':path4,'heat 1':path7,'heat 2':path8,'heat 3':path9}
    model=VotingModel_total_7('original','original','original','depth','heat','heat','heat',path1,path2,path3,path4,path7,path8,path9)
elif voting_member=='1:2(depth and heat)':
    path_dict={'depth 1':path4,'heat 1':path7,'heat 2':path8}
    model=VotingModel_total_3('depth','heat','heat',path4,path7,path8)
elif voting_member=='3:2(depth and heat)':
    path_dict={'depth 1':path4,'depth 2':path5,'depth 3':path6,'heat 1':path7,'heat 2':path8}
    model=VotingModel_total_5('depth','depth','depth','heat','heat',path4,path5,path6,path7,path8)
elif voting_member=='2:3(depth and heat)':
    path_dict={'depth 1':path4,'depth 2':path5,'heat 1':path7,'heat 2':path8,'heat 3':path9}
    model=VotingModel_total_5('depth','depth','heat','heat','heat',path4,path5,path7,path8,path9)
elif voting_member=='2:2:1':
    path_dict={'original 1':path1,'original 2':path2,'depth 1':path4,'depth 2':path5,'heat 1':path7}
    model=VotingModel_total_5('original','original','depth','depth','heat',path1,path2,path4,path5,path7)
elif voting_member=='2:1:2':
    path_dict={'original 1':path1,'original 2':path2,'depth 1':path4,'heat 1':path7,'heat 2':path8,}
    model=VotingModel_total_5('original','original','depth','heat','heat',
                            path1,path2,path4,path7,path8)
elif voting_member=='1:2:2':
    path_dict={'original 1':path1,'depth 1':path4,'depth 2':path5,'heat 1':path7,'heat 2':path8}
    model=VotingModel_total_5('original','depth','depth','heat','heat',path1,path4,path5,path7,path8)
elif voting_member=='3:2:2':
    path_dict={'original 1':path1,'original 2':path2,'original 3':path3,'depth 1':path4,'depth 2':path5,'heat 1':path7,'heat 2':path8}
    model=VotingModel_total_7('original','original','original','depth','depth','heat','heat',path1,path2,path3,path4,path5,path7,path8)
elif voting_member=='2:2:3':
    path_dict={'original 1':path1,'original 2':path2,'depth 1':path4,'depth 2':path5,'heat 1':path7,'heat 2':path8,'heat 3':path9}
    model=VotingModel_total_7('original','original','depth','depth','heat','heat','heat',path1,path2,path4,path5,path7,path8,path9)
elif voting_member=='2:3:2':
    path_dict={'original 1':path1,'original 2':path2,'depth 1':path4,'depth 2':path5,'depth 3':path6,'heat 1':path7,'heat 2':path8}
    model=VotingModel_total_7('original','original','depth','depth','depth','heat','heat',path1,path2,path4,path5,path6,path7,path8)


elif voting_member=='3:2':
    path_dict={'orginal 1':path1,'original 2':path2,'original 3':path3,'depth 1':path4,'depth 2':path5}
    model=VotingModel_3Origin_2Depth(path1,path2,path3,path4,path5)
elif voting_member=='1:1:1':
    path_dict={'original':path1,'depth':path4,'heat':path7}
    model=VotingModel_1_1_1('original','depth','heat',path1,path4,path7)
elif voting_member=='3:3:3':
    path_dict={'original 1':path1,'original 2':path2,'original 3':path3,'depth 1':path4,'depth 2':path5,'depth 3':path6,'heat 1':path7,'heat 2':path8,'heat 3':path9}
    model=VotingModel_3_3_3('original','original','original','depth','depth','depth','heat','heat','heat',
                            path1,path2,path3,path4,path5,path6,path7,path8,path9)
elif voting_member=='original':
    path_dict={'original':path1}
    model=Res18SqueezeSESCUNet(num_class=2,init_weight_type='xavier')
    weight=torch.load(path1)
    model.load_state_dict(weight['model'],strict=False)
elif voting_member=='depth':
    path_dict={'depth':path4}
    model=Res18SqueezeSESCUNet_depth(num_class=2,init_weight_type='xavier')
    weight=torch.load(path4)
    model.load_state_dict(weight['model'],strict=False)
elif voting_member=='heat':
    path_dict={'heat':path7}
    model=Res18SqueezeSESCUNet_depth(num_class=2,init_weight_type='xavier')
    weight=torch.load(path7)
    model.load_state_dict(weight['model'],strict=False)
elif voting_member=='2:1(heat)':
    path_dict={'orginal 1':path1,'original 2':path2,'heat 1':path7}
    model=VotingModel_2Origin_1Depth(path1,path2,path7)
elif voting_member=='3:2(heat)':
    path_dict={'orginal 1':path1,'original 2':path2,'original 3':path3,'heat 1':path7,'heat 2':path8}
    model=VotingModel_3Origin_2Depth(path1,path2,path3,path7,path8)
elif voting_member=='3(original)':
    path_dict={'orginal 1':path1,'original 2':path2,'original 3':path3}
    model=VotingModel_1_1_1('original','original','original',path1,path2,path3)
elif voting_member=='3:1:1':
    path_dict={'orginal 1':path1,'original 2':path2,'original 3':path3,'depth 1':path4,'heat 1':path7}
    model=VotingModel_3_1_1('original','original','original','depth','heat',path1,path2,path3,path4,path7)
elif voting_member=='3:3:1':
    path_dict={'original 1':path1,'original 2':path2,'original 3':path3,'depth 1':path4,'depth 2':path5,'depth 3':path6,'heat 1':path7}
    model=VotingModel_3_3_1('original','original','original','depth','depth','depth','heat',
                            path1,path2,path3,path4,path5,path6,path7)
# elif voting_member=='voting':
#     model=VotingModel(original=[path1,path2,path3])
#     path_dict={'orginal 1':path1,'original 2':path2,'original 3':path3}
elif voting_member=='2:3':
    path_dict={'orginal 1':path1,'original 2':path2,'depth 1':path4,'depth 2':path5,'depth 3':path6}
    model=VotingModel_2_3('original','original','depth','depth','depth',path1,path2,path4,path5,path6)
elif voting_member=='2:3(heat)':
    path_dict={'orginal 1':path1,'original 2':path2,'heat 1':path7,'heat 2':path8,'heat 3':path9}
    model=VotingModel_2_3('original','original','heat','heat','heat',path1,path2,path7,path8,path9)
elif voting_member=='1:2':
    path_dict={'orginal 1':path1,'depth 1':path4,'depth 2':path5}
    model=VotingModel_1_2(path1,path4,path5)
elif voting_member=='1:2(heat)':
    path_dict={'orginal 1':path1,'heat 1':path7,'heat 2':path8}
    model=VotingModel_1_2(path1,path7,path8)
elif voting_member=='3(depth)':
    path_dict={'depth 1':path4,'depth 2':path5,'depth 3':path6}
    model=VotingModel_1_1_1('depth','depth','depth',path4,path5,path6)
elif voting_member=='1:3:1':
    path_dict={'orginal 1':path1,'depth 1':path4,'depth 2':path5,'depth 3':path6,'heat 3':path9}
    model=VotingModel_1_3_1('original','depth','depth','depth','heat',path1,path4,path5,path6,path7)
elif voting_member=='1:1:3':
    path_dict={'orginal 1':path1,'depth 1':path4,'heat 1':path7,'heat 2':path8,'heat 3':path9}
    model=VotingModel_1_1_3('original','depth','heat','heat','heat',path1,path4,path7,path8,path9)
elif voting_member=='3(heat)':
    path_dict={'heat 1':path7,'heat 2':path8,'heat 3':path9}
    model=VotingModel_1_1_1('heat','heat','heat',path7,path8,path9)

# gpu problem
use_gpu=torch.cuda.is_available()
device=torch.device("cuda:0" if use_gpu else "cpu")
if use_gpu:
    model.to(device)



test_data=DepthDataset(type='original',mode=test_mode,n_class=2)
test_loader=DataLoader(test_data,batch_size=1,shuffle=False,num_workers=2)


test_img=open('depth_dataset/json_to_image_result/data_test1.txt','r')
test_img=test_img.readlines()

def test():

    with torch.no_grad():
        model.eval()
        total_ious=[]
        pixel_accs=[]

        val_tqdm=tqdm(enumerate(test_loader),total=len(test_loader),leave=True,desc='Test[1/1]')

        with open('depth_dataset/json_to_image_result/test_img_iou.txt','w') as f:
            for i,batch in val_tqdm:

                if not use_gpu:
                    original_img=batch['original']
                    gray_depth_img=batch['gray_depth']
                    gray_heat_img=batch['gray_heat']

                else:
                    original_img=batch['original'].to(device)
                    gray_depth_img=batch['gray_depth'].to(device)
                    gray_heat_img=batch['gray_heat'].to(device)

                # if voting_member=='1:1:1' or voting_member=='3:3:3' or voting_member=='3:1:1' or voting_member=='3:3:1' or voting_member=='voting' \
                # or voting_member=='1:3:1' or voting_member=='1:1:3' or voting_member=='1:3:3' or voting_member=='3:1:3' or voting_member=='3:2:2' \
                # or voting_member=='2:3:2' or voting_member=='2:2:3':
                #     output=model(original_img,gray_depth_img,gray_heat_img) 
                if voting_member=='original' :
                    output=model(original_img)
                    output=output.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
                    output=output.cpu().numpy()
                elif voting_member=='depth':
                    output=model(gray_depth_img)
                    output=output.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
                    output=output.cpu().numpy()
                elif voting_member=='heat':
                    output=model(gray_heat_img)
                    output=output.permute(0,2,3,1).reshape(-1,2).argmax(dim=1).reshape(1,400,400)
                    output=output.cpu().numpy()
                elif voting_member=='2:1(heat)' or voting_member=='3:2(heat)' or voting_member=='2:3(heat)' or voting_member=='1:2(heat)':
                    output=model(original_img,gray_heat_img)
                elif voting_member=='3(original)':
                    output=model(original_img,original_img,original_img)
                elif voting_member=='3(depth)':
                    output=model(gray_depth_img,gray_depth_img,gray_depth_img)
                elif voting_member=='3(heat)':
                    output=model(gray_heat_img,gray_heat_img,gray_heat_img)
                elif voting_member=='2:1(depth and heat)':    
                    output=model(gray_depth_img,gray_depth_img,gray_heat_img)
                elif voting_member=='3:2' or voting_member=='2:1' or voting_member=='1:2' or voting_member=='2:3':
                    output=model(original_img,gray_depth_img)
                else:
                    output=model(original_img,gray_depth_img,gray_heat_img) 

                if use_gpu:
                    target=batch['label'].reshape(1,400,400).to(device)
                else:
                    target=batch['label'].reshape(1,400,400)

                target=target.cpu().numpy()

                total_ious.append(iou(output,target,1))

                f.write('{}\t{}\n'.format(test_img[i].replace('\n',''),iou(output,target,1)[1]))

                # with open('depth_dataset/crop_images/generate_img_heat.txt','a') as file:
                # with open('crack_model_test/high_iou_custom_dataset_voting333.txt','a') as file:
                #     if iou(output,target,1)[1]>0.85:
                #         file.write('{}\t{}\n'.format(test_img[i].replace('\n',''),iou(output,target,1)[1]))

                pixel_accs.append(pixel_accuracy(output,target))

        ious=torch.tensor(total_ious).transpose(0,1).mean(dim=1)
        pixel_accu=torch.tensor(pixel_accs).mean()
        meaniou=torch.mean(ious)
        record_data(miou=meaniou,pixel_accuracy=pixel_accu,ciou=ious[1],biou=ious[0])

# def iou(predict,target):
#     ious=[]
#     for i in range(2):
#         pred_index=predict==i
#         target_index=target==i
#         intersection=pred_index[target_index].sum()
#         union=pred_index.sum()+target_index.sum()-intersection
#         if union==0:
#             if (i+1)%2 != 0:
#                 background_iou=1
#             else:
#                 crack_iou=1
#         else:
#             if (i+1)%2 != 0:
#                 background_iou=(float(intersection)/union)
#             else:
#                 crack_iou=(float(intersection)/union)

#     ious.append(background_iou)
#     ious.append(crack_iou)

#     return ious
def iou(pred,target,len_batch):
    ious=[]
    background_iou=[]
    crack_iou=[]
    for i in range(len_batch):
        for cls in range(2):
            pred_index=pred[i]==cls
            target_index=target[i]==cls
            intersection=pred_index[target_index].sum()
            union=pred_index.sum()+target_index.sum()-intersection
            if union==0:
                if (cls+1)%2 != 0:
                    background_iou.append(1)
                else:
                    crack_iou.append(1)
            else:
                if (cls+1)%2 != 0:
                    background_iou.append(float(intersection)/union)
                else:
                    crack_iou.append(float(intersection)/union)

    ious.append(sum(background_iou)/len(background_iou))
    ious.append(sum(crack_iou)/len(crack_iou))

    return ious

def pixel_accuracy(pred,target):
    correct=(pred==target).sum()
    total=(target==target).sum()
    accuracy=(correct/total)
    return accuracy

# run_time=1
# while True:
#     if not os.path.exists(f'crack_model_test/score(custom)/model_voting({test_mode})/{voting_member}_({run_time}).txt'.replace(':','_')):
#         break
#     run_time+=1

def record_data(miou,pixel_accuracy,ciou,biou):
    with open(f'crack_model_test/score(custom)/model_voting({test_mode})/{test_mode}__{voting_member}.txt'.replace(':','_'),'w') as f:
        f.write('{}\n'.format(test_mode))
        count_model=1
        for model,path in path_dict.items():
            f.write('model {}: {}\t{}\n'.format(count_model,model,path[38:]))
            count_model+=1
        f.write('menaIoU : {}\n'.format(miou))
        f.write('pixel accuracy : {}\n'.format(pixel_accuracy))
        f.write('crack IoU : {}\n'.format(ciou))
        f.write('background IoU : {}\n'.format(biou))


if __name__=='__main__':
    test()
