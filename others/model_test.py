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

from model import Res18ShuffleUNet,Res18MobileUNet,Res18SqueezeUNet,Res18GhostUNet,Res18SqueezeSCUNet,Res18SqueezeSESCUNet,Res18SqueezeNextUNet
from model_rebuild import unt_srdefnet18

batch_size=20
num_workers=2
catmap=['background','crack']
num_class=len(catmap)
init_weight='xavier'

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

test_loader=[test1_loader,test2_loader,test3_loader]

loss_function=nn.BCEWithLogitsLoss()


pretrained_own='12_0_9896_9858.pth'


#gpu problem
use_gpu=torch.cuda.is_available()
device=torch.device("cuda:0" if use_gpu else "cpu")

model_best_dict={'resnet18+shuffle+unet1':100,'resnet18+shuffle+unet2':102,'resnet18+shuffle+unet3':58,
                 'resnet18+mobile+unet1':130,'resnet18+mobile+unet2':157,'resnet18+mobile+unet3':68,
                 'resnet18+squeeze+unet1':131,'resnet18+squeeze+unet2':143,'resnet18+squeeze+unet3':132,
                 'resnet18+ghost+unet1':125,'resnet18+ghost+unet2':48,'resnet18+ghost+unet3':107}

model_dict={'resnet18+shuffle+unet':Res18ShuffleUNet(num_class=num_class,init_weight_type=init_weight),
            'resnet18+mobile+unet':Res18MobileUNet(num_class=num_class,init_weight_type=init_weight),
            'resnet18+squeeze+unet':Res18SqueezeUNet(num_class=num_class,init_weight_type=init_weight),
            'resnet18+ghost+unet':Res18GhostUNet(num_class=num_class,init_weight_type=init_weight),
            'resnet18+squeeze_sc+unet':Res18SqueezeSCUNet(num_class=num_class,init_weight_type=init_weight),
            'resnet18+squeeze_se_sc+unet':Res18SqueezeSESCUNet(num_class=num_class,init_weight_type=init_weight),
            'resnet18+squeezenext+unet':Res18SqueezeNextUNet(num_class=num_class,init_weight_type=init_weight),
            'model_rebuild':unt_srdefnet18(num_classes=num_class,pretrained_own=pretrained_own)
            }

def main():

    record_data_test(initial=True)

    for key in model_dict:
        for n in range(1,6):
            record_data_test(model_name=key+f'{n}')
            i=0
            model_path='crack_model_test/check points/{}/Epoch{}_of_200.pth'.format(key+f'{n}',model_best_dict[key+f'{n}'])

            for dataset in test_loader: 

                i+=1
                record_data_test(test_dataset=i)
                test(model=model_dict[key],test_loader=dataset,model_path=model_path,dataset_number=i)

def test(model,test_loader,model_path,dataset_number):
    
    if use_gpu:
        model.to(device)
    
    model_loader(model=model,model_path=model_path)

    with torch.no_grad():
        model.eval()
        total_ious=[]
        pixel_accs=[]
        test_loss=0
        test_tqdm=tqdm(test_loader,total=len(test_loader),leave=True,desc='Test[{}/{}]'.format(dataset_number,3))
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

def model_loader(model,model_path):
    weight=torch.load(model_path)
    model.load_state_dict(weight['model'],strict=False)
    # model.load_state_dict(model_path)


def record_data_test(model_name=None,test_loss=None,meaniou=None,IoU=None,pixel_acc=None,test_dataset=None,initial=None):
    if initial:
        with open('crack_model_test/score/test information/test_information.txt','w') as f:
            f.write('Total models:{}\n'.format(model_dict.keys()))
    else: 
        with open('crack_model_test/score/test information/test_information.txt','a') as f:
            if model_name:
                f.write('\n################### Model name: {} ###################\n\n'.format(model_name))
            if test_dataset:
                f.write('########## Test dataset: test {} ##########\n'.format(test_dataset)) 
            if test_loss:
                f.write('Test loss: {}\n'.format(test_loss))
            if IoU is not None:
                f.write('IoU: {}\n'.format(IoU))
            if meaniou:
                f.write('Meaniou: {}\n'.format(meaniou))
            if pixel_acc:
                f.write('Pixel accuracy: {}\n\n'.format(pixel_acc))

if __name__=='__main__':
    # main()

    #resnet18+squeeze+unet5
    # i=0
    # record_data_test(model_name='resnet18+squeeze+unet5')
    # for dataset in test_loader: 
    #     i+=1
    #     record_data_test(test_dataset=i)
    #     test(model=model_dict['resnet18+squeeze+unet'],test_loader=dataset,model_path=weight,dataset_number=i)

    #resnet18+squeeze+unet6
    # i=0
    # record_data_test(model_name='resnet18+squeeze+unet6')
    # for dataset in test_loader: 
    #     i+=1
    #     record_data_test(test_dataset=i)
    #     test(model=model_dict['resnet18+squeeze+unet'],test_loader=dataset,model_path=weight,dataset_number=i)

    #resnet18+squeeze_sc+unet1
    # weight='crack_model_test/check points/resnet18+squeeze_se_sc+unet1/Epoch199_of_200.pth'
    # i=0
    # record_data_test(model_name='resnet18+squeeze_se_sc+unet1')
    # for dataset in test_loader: 
    #     i+=1
    #     record_data_test(test_dataset=i)
    #     test(model=model_dict['resnet18+squeeze_se_sc+unet'],test_loader=dataset,model_path=weight,dataset_number=i)
    
    #resnet18+squeeze_sc+unet2
    # weight='crack_model_test/check points/resnet18+squeeze_se_sc+unet2/Epoch163_of_300.pth'
    # i=0
    # record_data_test(model_name='resnet18+squeeze_se_sc+unet2')
    # for dataset in test_loader: 
    #     i+=1
    #     record_data_test(test_dataset=i)
    #     test(model=model_dict['resnet18+squeeze_se_sc+unet'],test_loader=dataset,model_path=weight,dataset_number=i)
        
    #resnet18+squeeze_sc+unet3
    # weight='crack_model_test/check points/resnet18+squeeze_se_sc+unet3/Epoch196_of_250.pth'
    # i=0
    # record_data_test(model_name='resnet18+squeeze_se_sc+unet3')
    # for dataset in test_loader: 
    #     i+=1
    #     record_data_test(test_dataset=i)
    #     test(model=model_dict['resnet18+squeeze_se_sc+unet'],test_loader=dataset,model_path=weight,dataset_number=i)

    #resnet18+squeezenext+unet1
    # weight='crack_model_test/check points/resnet18+suqeezenext+unet1/Epoch102_of_250.pth'
    # i=0
    # record_data_test(model_name='resnet18+squeezenext+unet1')
    # for dataset in test_loader: 
    #     i+=1
    #     record_data_test(test_dataset=i)
    #     test(model=model_dict['resnet18+squeezenext+unet'],test_loader=dataset,model_path=weight,dataset_number=i)

    #model rebuild
    weight='20_0_9813_9609.pth'
    record_data_test(model_name='model_rebuild')
    test(model=model_dict['model_rebuild'],test_loader=test1_loader,model_path=weight,dataset_number=1)
