import  torch
import  numpy as np
import  torch.nn as nn
import  torchvision
from tqdm import tqdm
import random
import os
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torchvision.utils import save_image
import wandb
import datetime
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp


project='MAnet_zhong_6.15'
run_name='MAnet_6.15_mit_b3_MAX'
#device=torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

transform=A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2,p=0.5),
    #A.GaussianBlur(blur_limit=(5, 7), p=0.2),
    #A.RandomCrop(2, 64, p=0.2),
    
    A.Affine(scale=1.2, p=0.2),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
    A.RandomGamma(gamma_limit=(80, 120), p=0.2),
    #A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
    #A.RandomSnow(brightness_coeff=1.5, p=0.2),
    #A.MaskDropout(max_objects=8, p=0.2),
    #A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5),
    
    #transforms.RandomHorizontalFlip(),
    #transforms.RandomVerticalFlip(),
    #transforms.RandomRotation(25),
    #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  
    #transforms.RandomApply([transforms.GaussianBlur(5, scigma=(0.1, 2.0))], p=0.1),
    #ToTensorV2()
   
])
transform2=transforms.Compose([
    
    transforms.ToTensor()
])
transform_test=transforms.Compose([
    
    transforms.ToTensor()
])
'''Cutout：在图像上随机生成一些矩形区域，并将这些区域的像素值设置为0，这种方法被证明对防止过拟合很有帮助。

Cutmix：从两个图像中随机选择一些区域，并将这些区域的像素值混合在一起。

Classmix：这是一种新的数据增强技术，它通过混合来自不同类别的图像来生成新的训练样本。'''

#图像增强对比


#数据集读取处理
class MyDataset(Dataset):
    def __init__(self, path):
        self.mode=('train' if 'mask' in os.listdir(path) else 'test')
        self.path=path
        dir_list=os.listdir(path+'image/')
        self.name=[n for n in dir_list if n[-3:]=='png']
    
    def __len__(self):
        return len(self.name)
    
    def __getitem__(self, idx):
        name=self.name[idx]

        if self.mode=='train':
            img=cv2.imread(self.path+'image/'+name)
            mask=cv2.imread(self.path+'mask/'+name,cv2.IMREAD_GRAYSCALE)
            image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #mask=cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            #print(image.shape, mask.shape)

            #origin_image = Image.fromarray(origin_image)
            #origin_mask = Image.fromarray(origin_mask)

            #augmentation1 = transform(image)
            #augmentation2 = transform(mask)
            augmented = transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
            #return transform(origin_image), transform(origin_mask)
            return transform2(image), transform2(mask)
        
        if self.mode=='test':
            img=cv2.imread(self.path+'image/'+name)
            origin_image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            return transform_test(origin_image)

train_path='train/'
#train_path='fusai_train/'
train_data=MyDataset(train_path)

#current_datetime = datetime.datetime.now().strftime('%m%d_%H%M')
#print("Current Date and Time:", current_datetime)

model_path=f'model{run_name}/'


# 检查目录是否存在，如果不存在则创建
if not os.path.exists(model_path):
    os.makedirs(model_path)

model=smp.MAnet(#encoder_name='resnet50', 
                #encoder_name='mit_b4',

               encoder_name='mit_b3', 
               encoder_weights='imagenet', 
               in_channels=3,
               decoder_channels=(512, 256, 128, 64, 32), 
               decoder_pab_channels=128,
               #decoder_attention_type='scse',
               #encoder_output_stride=8,
               classes=1, )





#model=model.to(device)
'''if torch.cuda.is_available():
    device = torch.device("cuda:2")
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
else:
    device = torch.device("cuda:3")
    print("Training on CPU.")

# 如果有多个GPU，使用DataParallel
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # 指定使用的GPU编号，例如使用编号为0和1的GPU
    device_ids = [2, 3]
    model = nn.DataParallel(model, device_ids=device_ids)
    # 将模型移动到第一个指定的GPU上
    model.to(f'cuda:{device_ids[0]}')
else:
    # 将模型移动到GPU或CPU
    model.to(device)'''
# 检查CUDA是否可用
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    # 指定使用的GPU编号
    device_ids = [2, 3] 
    # 如果有多个GPU，使用DataParallel
    if num_gpus > 1:
        print("Let's use", len(device_ids), "GPUs!")
        model = nn.DataParallel(model, device_ids=device_ids)
        device = torch.device(f"cuda:{device_ids[0]}")
    else:
        device = torch.device(f"cuda:{device_ids[0]}")
else:
    print("Training on CPU.")
    device = torch.device("cpu")
model = model.to(device)

run=wandb.init(project=project, name=run_name,
               config={'batch_size':10, 
                       'lr':1e-4,
                       'epoch':400,
                       'weight_decay':1e-4,
                       'device':device,
                       'transform':transform,
                       'model':model,
                       })
config = wandb.config
wandb.watch(model)




optimizer=torch.optim.AdamW(model.parameters(),lr=config.lr, weight_decay=config.weight_decay)
trainloader=DataLoader(train_data, batch_size=config.batch_size, num_workers=1)

class CombieLoss(nn.Module):
    def __init__(self):
        super(CombieLoss, self).__init__()
        self.dice_loss = smp.losses.DiceLoss(mode='binary')
        self.bce_loss = smp.losses.SoftBCEWithLogitsLoss(reduction='mean', smooth_factor=0.1)
        self.focal_loss=smp.losses.FocalLoss(mode='binary')
        #smp.losses.JaccardLoss
    def forward(self, y_pred, y_true):
        loss_dice = self.dice_loss(y_pred, y_true)
        loss_bce = self.bce_loss(y_pred, y_true)
        loss_focal=self.focal_loss(y_pred, y_true)
        loss = loss_dice*0.6 + loss_bce*0.2+loss_focal*0.2
        return loss
combieLoss=CombieLoss()
'''
def dice_loss(logits, targets):
    smooth = 1
    prob = torch.sigmoid(logits)
    batch = prob.size(0)
    prob = prob.view(batch, 1, -1)
    targets = targets.view(batch, 1, -1)
    intersection = torch.sum(prob * targets, dim=2)
    union = torch.sum(prob, dim=2) + torch.sum(targets, dim=2)
    
    dice = (2 * intersection + smooth) / (union + smooth)
    dice_loss = 1 - dice.mean()
    
    return dice_loss'''

#diceloss = smp.losses.DiceLoss(mode='binary')
#binloss = smp.losses.SoftBCEWithLogitsLoss(reduction='mean', smooth_factor=0.1)
#训练
if __name__ == '__main__':
    loss_last=99999
    best_model_name='x'

    for epoch in range(1,config.epoch+1):
        for step,(inputs,labels) in tqdm(enumerate(trainloader),
                                        desc=f"Epoch {epoch}/{config.epoch}",
                                        ascii=True, total=len(trainloader)):
            #print(f"Inputs shape: {inputs.shape}, Labels shape: {labels.shape}")
            # 将输入数据调整为模型期望的形状
            #inputs = inputs.permute(0, 3, 1, 2).float()  # 将通道维移动到第二个位置


            print(f"Inputs shape: {inputs.shape}, Labels shape: {labels.shape}")
            inputs,labels=inputs.to(device),labels.to(device)
            outputs=model(inputs)
            
            loss=combieLoss(outputs,labels)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            wandb.log({"Epoch": epoch, "Loss": loss.item()})
            #wandb.log({"Epoch": epoch, "DiceLoss": loss.diceloss.item()})
            #wandb.log({"Epoch": epoch, "BinLoss": loss.binloss.item()})
            print(f"\nEpoch: {epoch}/{config.epoch},Loss:{loss}")
        if loss<loss_last:
            loss_last=loss
            torch.save(model.state_dict(), model_path+'model_epoch_{}._loss{}.pth'.format(epoch,loss))
            best_model_name='model_epoch_{}._loss{}.pth'.format(epoch,loss)
            
run.finish()