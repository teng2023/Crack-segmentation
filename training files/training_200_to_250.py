if '__file__' in globals(): 
    import sys,os
    sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

from boat.utils.dataset import crack_semantic
import boat.utils.augment as augment

from model import Res18MobileUNet,Res18ShuffleUNet,Res18UNet,Res18SqueezeUNet,Res18GhostUNet,Res18SqueezeSESCUNet

###################################################################################################################
#training conditions
training_begin=True
training_continue=not training_begin
model_name='resnet18+squeeze_se_sc+unet'
best_meaniou=0
best_meaniou_epoch=0
best_crack_iou=0
best_crack_iou_epoch=0
best_pixel_accuracy=0
best_pixel_accuracy_epoch=0
# continue_epoch=0
# total_train_time=3
###################################################################################################################

def main(train_time):
        
    model_dir='crack_model_test/check points/{}{}'.format(model_name,train_time)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    score_dir='crack_model_test/score/{}'.format(model_name)
    if not os.path.exists(score_dir):
        os.makedirs(score_dir)

    best_miou=best_meaniou
    best_miou_ep=best_meaniou_epoch
    best_pix_acc=best_pixel_accuracy
    best_pix_acc_ep=best_pixel_accuracy_epoch
    best_c_iou=best_crack_iou
    best_c_iou_ep=best_crack_iou_epoch

    record_data(clean=False,train_time=train_time)
    
    for epoch in range(previous_epoch+1,total_epochs+1):

        record_data(clean=False,start_time=True,train_time=train_time)

        train(epoch,train_time=train_time)

        meaniou,pixel_accu,crack_iou=val(epoch,train_time=train_time)
        
        if meaniou>best_miou:
            best_miou=meaniou
            best_miou_ep=epoch
        if crack_iou>best_c_iou:
            best_c_iou=crack_iou
            best_c_iou_ep=epoch
        if pixel_accu>best_pix_acc:
            best_pix_acc=pixel_accu
            best_pix_acc_ep=epoch
        record_data(clean=False,best_meaniou_ep=best_miou_ep,best_pixel_ep=best_pix_acc_ep,best_pixel=best_pix_acc,best_meaniou=best_miou,
                    best_crack_iou=best_c_iou,best_crack_iou_ep=best_c_iou_ep,train_time=train_time)
        
def train(epoch,train_time):
    model.train()
    train_tqdm=tqdm(train_loader,total=len(train_loader),leave=True,desc=f'Train[{epoch}/{total_epochs}]')
    train_loss=0
    i=0
    record_data(epoch=epoch,train_time=train_time)
    for imgs,masks in train_tqdm:
        i+=1
        optimizer.zero_grad()

        imgs=imgs.to(device)
        masks=masks.to(device)

        outputs=model(imgs)

        del imgs
        torch.cuda.empty_cache()

        loss=loss_function(outputs,masks)
        train_loss+=loss.item()
        loss.backward()
        optimizer.step()
        
        del masks,loss,outputs
        torch.cuda.empty_cache()

        train_tqdm.set_postfix(loss='{:.3f}'.format((train_loss)/(i)))

    model_keeper(epoch=epoch,mode='save',train_time=train_time)
    record_data(train_loss=(train_loss/len(train_loader)),clean=False,train_time=train_time)

def val(epoch,train_time):
    with torch.no_grad():
        model.eval()
        total_ious=[]
        pixel_accs=[]
        val_loss=0
        val_tqdm=tqdm(val_loader,total=len(val_loader),leave=True,desc='Val[{}/{}]'.format(epoch,total_epochs))
        i=0
        for imgs,masks in val_tqdm:
            i=i+1
            imgs=imgs.to(device)
            masks=masks.to(device)

            outputs=model(imgs) 

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
        record_data(val_loss=val_loss/len(val_loader),meaniou=meaniou,pixel_accuracy=pixel_accu,clean=False,IoUs=ious,train_time=train_time)
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

def record_data(clean=None,epoch=None,train_loss=None,val_loss=None,meaniou=None,pixel_accuracy=None,IoUs=None,train_time=None,
                best_meaniou_ep=None,best_pixel_ep=None,start_time=None,lr=None,best_meaniou=None,best_pixel=None,best_crack_iou=None,best_crack_iou_ep=None):
    now_time=time.localtime()
    if clean:
        c='w'
    else:
        c='a'
    
    with open('crack_model_test/score/{}/{}_information{}.txt'.format(model_name,model_name,train_time),c) as f:
        if epoch:
            f.write('Epoch: {}/{}\n'.format(epoch,total_epochs))
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

def model_keeper(mode=None,epoch=None,train_time=None):
    state={'model':model.state_dict(),'optimizer':optimizer.state_dict()}
    model_path='crack_model_test/check points/{}{}/Epoch{}_of_{}.pth'.format(model_name,train_time,epoch,total_epochs)
    if mode=='save':
        torch.save(state,model_path)
    elif mode=='load':
        weight=torch.load(model_path)
        model.load_state_dict(weight['model'],strict=True)
        optimizer.load_state_dict(weight['optimizer'])
    elif mode=='exist':
        if os.path.exists(model_path):
            return True
        return False

if __name__=='__main__':

    previous_epoch=200
    total_epochs=250
    batch_size=15
    num_workers=2
    learning_rate=1e-3
    init_weight=True
    train_time=1

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

    train_file='boat/database/v_crack/8751crack_400400_50/train.txt'
    val_file='boat/database/v_crack/8751crack_400400_50/val1.txt'

    train_data=crack_semantic(train_file,cattable,transform=augment.Compose(train_transforms,dataset_cat ='semantic'))
    val_data=crack_semantic(val_file,cattable,transform=augment.Compose(validation_transforms,dataset_cat ='semantic'))

    train_loader=DataLoader(train_data,batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader=DataLoader(val_data,batch_size=batch_size,shuffle=False,num_workers=num_workers)

    #choose model
    model_dict={'resnet18+unet':Res18UNet(num_class=num_class,init_weight=init_weight),
                'resnet18+shuffle+unet':Res18ShuffleUNet(num_class=num_class,init_weight=init_weight),
                'resnet18+mobile+unet':Res18MobileUNet(num_class=num_class,init_weight=init_weight),
                'resnet18+squeeze+unet':Res18SqueezeUNet(num_class=num_class,init_weight=init_weight),
                'resnet18+ghost+unet':Res18GhostUNet(num_class=num_class,init_weight=init_weight),
                'resnet18+squeeze_se_sc+unet':Res18SqueezeSESCUNet(num_class=num_class,init_weight=init_weight)
                }

    #gpu problem
    use_gpu=torch.cuda.is_available()
    device=torch.device("cuda:0" if use_gpu else "cpu")

    model=model_dict[model_name]

    if use_gpu:
        model.to(device)

    loss_function=nn.BCEWithLogitsLoss()
    optimizer=optim.Adam([{'params':model.parameters(),'initial_lr':learning_rate}],lr=learning_rate)

    model_path='crack_model_test/check points/{}{}/Epoch{}_of_{}.pth'.format(model_name,train_time,previous_epoch,previous_epoch)
    weight=torch.load(model_path)
    model.load_state_dict(weight['model'],strict=True)
    optimizer.load_state_dict(weight['optimizer'])

    main(train_time=1)
