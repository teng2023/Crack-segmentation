if '__file__' in globals(): 
    import sys,os
    sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import torch.optim as optim

from model_rebuild import unt_srdefnet18
from model import Res18SqueezeSESCUNet

from boat.utils.dataset import crack_semantic
import boat.utils.augment as augment

num_class=2
learning_rate=1e-3
model_name='model_rebuild'
train_time=1

model_dir='crack_model_test/check points/{}{}'.format(model_name,train_time)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
score_dir='crack_model_test/score/{}'.format(model_name)
if not os.path.exists(score_dir):
    os.makedirs(score_dir)
###################################################################################################################
#crack conditions
colormap=[[0, 0, 0], [0, 0, 255]]#rgb
catmap=['background','crack']
num_class=len(catmap)

mean=(0.485,0.456,0.406)
std=(0.229,0.224,0.225)

train_transforms=[augment.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225),dataset_cat='semantic')]
validation_transforms=[augment.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225),dataset_cat='semantic')]

cattable = torch.zeros(256**3)
for i, rgb in enumerate(colormap):
    cattable[(rgb[0] * 256 + rgb[1]) * 256 + rgb[2]] = i

###################################################################################################################
val_file='boat/database/v_crack/8751crack_400400_50/val1.txt'
val_data=crack_semantic(val_file,cattable,transform=augment.Compose(validation_transforms,dataset_cat ='semantic'))
val_loader=DataLoader(val_data,batch_size=1,shuffle=False,num_workers=2)

weight='crack_model_test/13_100_9921_9921.pth'
model=unt_srdefnet18(num_classes=num_class,pretrained_own=weight)

#gpu problem
use_gpu=torch.cuda.is_available()
device=torch.device("cuda:0" if use_gpu else "cpu")
if use_gpu:
    model.to(device)

loss_function=nn.BCEWithLogitsLoss()
optimizer=optim.Adam([{'params':model.parameters(),'initial_lr':learning_rate}],lr=learning_rate)

def test():
    with torch.no_grad():
        model.eval()
        total_ious=[]
        pixel_accs=[]
        val_loss=0
        val_tqdm=tqdm(val_loader,total=len(val_loader),leave=True,desc='Val[{}/{}]'.format(1,1))
        i=0
        for imgs,masks in val_tqdm:
            i=i+1
            imgs=imgs.to(device)
            masks=masks.to(device)

            outputs=model(imgs)
            # if model_name=='unet':
            #     resize=transforms.Resize([masks.shape[2],masks.shape[3]],antialias=True)
            #     outputs=resize(outputs)

            loss=loss_function(outputs,masks)
            val_loss+=loss.item()

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

            val_tqdm.set_postfix(loss='{:.3f}'.format((val_loss)/(i)))

        ious=torch.tensor(total_ious,device=device).transpose(0,1).mean(dim=1)
        pixel_accu=torch.tensor(pixel_accs).mean()
        meaniou=torch.nanmean(ious)
        record_data(val_loss=val_loss/len(val_loader),meaniou=meaniou,pixel_accuracy=pixel_accu,clean=False,IoUs=ious)
        return meaniou,pixel_accu,ious[1]

def iou(pred,target,len_batch):
    ious=[]
    background_iou=[]
    crack_iou=[]
    for i in range(len_batch):
        for cls in range(num_class):
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
    
def pixel_accuracy(pred,target,len_batch):
    accuracy=[]
    for i in range(len_batch):
        correct=(pred[i]==target[i]).sum()
        total=(target[i]==target[i]).sum()
        accuracy.append(correct/total)
        return accuracy

def record_data(clean=None,epoch=None,train_loss=None,val_loss=None,meaniou=None,pixel_accuracy=None,IoUs=None,
                best_meaniou_ep=None,best_pixel_ep=None,start_time=None,lr=None,best_meaniou=None,best_pixel=None,best_crack_iou=None,best_crack_iou_ep=None,
                training_time=None,train_val_time=None):
    now_time=time.localtime()
    if clean:
        c='w'
    else:
        c='a'
    
    with open('crack_model_test/score/{}/{}_information{}.txt'.format(model_name,model_name,train_time),c) as f:
        if epoch:
            f.write('Epoch: {}/{}\n'.format(1,1))
        if train_loss:
            f.write('training loss: {}\n'.format(train_loss))
        if val_loss:
            f.write('validation loss: {}\n'.format(val_loss))
        if meaniou:
            f.write('meaniou: {}\n'.format(meaniou))
        if pixel_accuracy:
            f.write('pixel accuracy: {}\n'.format(pixel_accuracy))
        if IoUs is not None:
            f.write('IoUs: {}\n'.format(IoUs))
        if best_meaniou_ep and best_meaniou:
            f.write('best meaniou: Epoch{}\t{}\n'.format(best_meaniou_ep,best_meaniou))
        if best_crack_iou and best_crack_iou_ep:
            f.write('best crack iou: Epoch{}\t{}\n'.format(best_crack_iou_ep,best_crack_iou))
        if best_pixel_ep and best_pixel:
            f.write('best pixel accuracy: Epoch{}\t{}\n'.format(best_pixel_ep,best_pixel))
        if lr:
            f.write('learning rate: {}\n'.format(lr))
        if start_time:
            f.write(f'\n{now_time.tm_year}/{now_time.tm_mon}/{now_time.tm_mday} {now_time.tm_hour}:{now_time.tm_min}:{now_time.tm_sec}\n')  
        if training_time:
            f.write('training time (without val): {}\n'.format(training_time))
        if train_val_time:
            f.write('training time (with val): {}\n'.format(train_val_time))

if __name__=='__main__':
    test()


