if '__file__' in globals(): 
    import sys,os
    sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

import torch
import torch.nn as nn
from training import iou,pixel_accuracy
from torch.utils.data import DataLoader
from tqdm import tqdm
from boat.utils.dataset import crack_semantic
import boat.utils.augment as augment
import statistics
import numpy as np


from model import Res18SqueezeUNet,Res18SqueezeSCUNet,Res18SqueezeSESCUNet,Res18SqueezeNextUNet,Res18SqueezeSEUNet

batch_size=20
num_workers=2
catmap=['background','crack']
num_class=len(catmap)
init_weight=True
init_weight_type='xavier'

colormap=[[0, 0, 0], [0, 0, 255]]#rgb
cattable = torch.zeros(256**3)
for i, rgb in enumerate(colormap):
    cattable[(rgb[0] * 256 + rgb[1]) * 256 + rgb[2]] = i

test_transforms=[augment.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225),dataset_cat='semantic')]

test1_file='boat/database/v_crack/8751crack_400400_50/test1.txt'
test1_data=crack_semantic(test1_file,cattable,transform=augment.Compose(test_transforms,dataset_cat='semantic'))
test1_loader=DataLoader(test1_data,batch_size=batch_size,shuffle=False,num_workers=num_workers)

test2_file='boat/database/v_crack/8751crack_400400_50/test2.txt'
test2_data=crack_semantic(test2_file,cattable,transform=augment.Compose(test_transforms,dataset_cat='semantic'))
test2_loader=DataLoader(test2_data,batch_size=batch_size,shuffle=False,num_workers=num_workers)

test3_file='boat/database/v_crack/8751crack_400400_50/test3.txt'
test3_data=crack_semantic(test3_file,cattable,transform=augment.Compose(test_transforms,dataset_cat='semantic'))
test3_loader=DataLoader(test3_data,batch_size=batch_size,shuffle=False,num_workers=num_workers)

# test_loader=[test1_loader,test2_loader,test3_loader]

loss_function=nn.BCEWithLogitsLoss()

#gpu problem
use_gpu=torch.cuda.is_available()
device=torch.device("cuda:0" if use_gpu else "cpu")

# model_best_dict={'resnet18+squeeze+unet1':131,'resnet18+squeeze+unet2':143,'resnet18+squeeze+unet3':132,
#                  'resnet18+squeeze_sc+unet1':140,
#                  'resnet18+squeeze+sc_se+unet1':199,'resnet18+squeeze+sc_se+unet2':163,'resnet18+squeeze+sc_se+unet3':196,
#                 }

model_best_dict={'resnet18+squeeze+unet':[(1,131,200),(2,143,200),(3,132,200)], #(order,best epoch,total epoch)
                 'resnet18+squeeze_sc+unet':[(1,140,200)],
                 'resnet18+squeeze_se_sc+unet':[(15,193,250),(16,134,250),(17,223,250)],
                 'resnet18+squeeze_se+unet':[(2,214,250),(3,229,250),(4,207,250)],
                 'resnet18+squeezenext+unet':[(1,102,250)]
                }

model_dict={#'resnet18+shuffle+unet':Res18ShuffleUNet(num_class=num_class,init_weight=init_weight),
            # 'resnet18+mobile+unet':Res18MobileUNet(num_class=num_class,init_weight=init_weight),
            'resnet18+squeeze+unet':Res18SqueezeUNet(num_class=num_class,init_weight_type=init_weight_type),
            # 'resnet18+ghost+unet':Res18GhostUNet(num_class=num_class,init_weight=init_weight),
            'resnet18+squeeze_sc+unet':Res18SqueezeSCUNet(num_class=num_class,init_weight_type=init_weight_type),
            'resnet18+squeeze_se_sc+unet':Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight_type),
            'resnet18+squeeze_se+unet':Res18SqueezeSEUNet(num_class=num_class,init_weight_type=init_weight_type),
            'resnet18+squeezenext+unet':Res18SqueezeNextUNet(num_class=num_class,init_weight_type=init_weight_type)
            }

def main(test_number,test_dataloader,total_number):

    record_data_test(initial=True,test_dataset=test_number)

    n=0

    for model_name in model_dict:
        test_number=len(model_best_dict[model_name])
        pixel_list=[]
        meaniou_list=[]
        crack_iou_list=[]
        bg_iou_list=[]

        for i in range(1,test_number+1):
            n+=1
            record_data_test(model_name=model_name+f'{model_best_dict[model_name][i-1][0]}')
            model_path='crack_model_test/check points/{}/Epoch{}_of_{}.pth'.format(model_name+f'{model_best_dict[model_name][i-1][0]}',model_best_dict[model_name][i-1][1],model_best_dict[model_name][i-1][2])
            p,miou,ci,bgi=test(model=model_dict[model_name],test_loader=test_dataloader,model_path=model_path,dataset_number=n)

            pixel_list.append(p)
            meaniou_list.append(miou)
            crack_iou_list.append(ci)
            bg_iou_list.append(bgi)

        a=np.std(pixel_list)
        b=np.std(meaniou_list)
        c=np.std(crack_iou_list)
        d=np.std(bg_iou_list)

        aa=np.mean(pixel_list)
        bb=np.mean(meaniou_list)
        cc=np.mean(crack_iou_list)
        dd=np.mean(bg_iou_list)

        record_data_test(pixel_av=aa,pixel_std=a,miou_av=bb,miou_std=b,ci_av=cc,ci_std=c,bgi_av=dd,bgi_std=d)
            

def test(model,test_loader,model_path,dataset_number):
    
    if use_gpu:
        model.to(device)
    
    model_loader(model=model,model_path=model_path)

    with torch.no_grad():
        model.eval()
        total_ious=[]
        pixel_accs=[]
        test_loss=0
        test_tqdm=tqdm(test_loader,total=len(test_loader),leave=True,desc='Test[{}/{}]'.format(dataset_number,total_number))
        i=0
        for imgs,masks in test_tqdm:
            i=i+1
            imgs=imgs.to(device)
            masks=masks.to(device)

            outputs=model(imgs) 

            loss=loss_function(outputs,masks)
            test_loss+=loss.item()

            del imgs,loss
            torch.cuda.empty_cache()
            
            N,_,h,w=outputs.shape
            pred=outputs.permute(0,2,3,1).reshape(-1,num_class).argmax(dim=1).reshape(N,h,w)
            
            del outputs
            torch.cuda.empty_cache()

            target=masks.argmax(dim=1).reshape(N,h,w).to(device)

            total_ious.append(iou(pred,target,N))
            pixel_accs.extend(pixel_accuracy(pred,target,N))
            
            del pred,target,masks
            torch.cuda.empty_cache()

            test_tqdm.set_postfix(loss='{:.3f}'.format((test_loss)/(i)))
            
        ious=torch.tensor(total_ious,device=device).transpose(0,1).mean(dim=1)
        pixel_accu=torch.tensor(pixel_accs).mean()
        meaniou=torch.nanmean(ious)
        record_data_test(test_loss=test_loss/len(test_loader),IoU=ious,meaniou=meaniou,pixel_acc=pixel_accu)

        pixel_accu=pixel_accu.cpu().numpy()
        meaniou=meaniou.cpu().numpy()
        ious=ious.cpu().numpy()
    
    return pixel_accu,meaniou,ious[1],ious[0]

def model_loader(model,model_path):
    weight=torch.load(model_path)
    model.load_state_dict(weight['model'],strict=False)


def record_data_test(model_name=None,test_loss=None,meaniou=None,IoU=None,pixel_acc=None,test_dataset=None,initial=None,
                     pixel_av=None,miou_av=None,ci_av=None,bgi_av=None,pixel_std=None,miou_std=None,ci_std=None,bgi_std=None):
    if initial:
        with open('crack_model_test/score/test information/test_information_squeeze_version.txt','w') as f:
            f.write('Total models:{}\n'.format(model_dict.keys()))
    else: 
        with open('crack_model_test/score/test information/test_information_squeeze_version.txt','a') as f:
            if model_name:
                f.write('\n################### Model name: {} ###################\n\n'.format(model_name))
            if test_dataset:
                f.write('########## Test dataset: test {} ##########\n'.format(test_dataset)) 
            if test_loss:
                f.write('Test loss: {}\n'.format(test_loss))
            if IoU is not None:
                f.write('Crack IoU: {}\n'.format(IoU[1]))
                f.write('Background IoU: {}\n'.format(IoU[0]))
            if meaniou:
                f.write('Meaniou: {}\n'.format(meaniou))
            if pixel_acc:
                f.write('Pixel accuracy: {}\n\n'.format(pixel_acc))
            if pixel_av and pixel_std:
                f.write('Pixel accuracy\'s average: {} \t standard deviation: {}\n'.format(pixel_av,pixel_std))
            if miou_av and miou_std:
                f.write('Mean IoU\'s average: {} \t standard deviation: {}\n'.format(miou_av,miou_std))
            if ci_av and ci_std:
                f.write('Crack IoU\'s average: {} \t standard deviation: {}\n'.format(ci_av,ci_std))
            if bgi_av and bgi_std:
                f.write('Background IoU\'s average: {} \t standard deviation: {}\n'.format(bgi_av,bgi_std))

if __name__=='__main__':

    #counting total number to run
    total_number=0
    # for model_name in model_dict:
    #     total_number+=len(model_best_dict[model_name])

    # main(test_number=1,test_dataloader=test1_loader,total_number=total_number)


    # model_load=Res18SqueezeUNet(num_class=num_class,init_weight_type=init_weight_type)
    # weight_load=torch.load('crack_model_test/check points/resnet18+squeeze+unet6/Epoch133_of_200.pth')
    # model_load.load_state_dict(weight_load['model'],strict=False)
    # save_mdoel=torch.save(model_load,'resnet_with_squeezenet_and_unet.pth')
    model=torch.load('resnet_with_squeezenet_and_unet.pth')

