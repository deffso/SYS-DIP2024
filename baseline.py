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

run_name='baseline_RFLIP'
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    #transforms.RandomVerticalFlip(),
    #transforms.RandomRotation(25),
    #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  
    #transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.1),
    
])
transform_test=transforms.Compose([
    
    transforms.ToTensor()
])
'''Cutout：在图像上随机生成一些矩形区域，并将这些区域的像素值设置为0，这种方法被证明对防止过拟合很有帮助。

Cutmix：从两个图像中随机选择一些区域，并将这些区域的像素值混合在一起。

Classmix：这是一种新的数据增强技术，它通过混合来自不同类别的图像来生成新的训练样本。
ElasticTransform：对图像进行弹性变换，这种变换会模拟图像的非均匀变形。'''
#图像增强对比
'''
# 打开一个图像文件
image = Image.open("my_image.jpg")
transformed_image = transform(image)

transformed_image_pil = transforms.ToPILImage()(transformed_image)

# 创建一个新的matplotlib图像
fig, ax = plt.subplots(1, 2)

# 显示原始图像
ax[0].imshow(image)
ax[0].set_title("Original Image")

# 显示转换后的图像
ax[1].imshow(transformed_image_pil)
ax[1].set_title("Transformed Image")

# 显示图像
plt.show()'''
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
            mask=cv2.imread(self.path+'mask/'+name)
            origin_image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            origin_mask=cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

            #origin_image = Image.fromarray(origin_image)
            #origin_mask = Image.fromarray(origin_mask)

            
            return transform(origin_image), transform(origin_mask)
        
        if self.mode=='test':
            img=cv2.imread(self.path+'image/'+name)
            origin_image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            return transform_test(origin_image)

train_path='train/'
train_data=MyDataset(train_path)

#current_datetime = datetime.datetime.now().strftime('%m%d_%H%M')
#print("Current Date and Time:", current_datetime)

model_path=f'model{run_name}/'


# 检查目录是否存在，如果不存在则创建
if not os.path.exists(model_path):
    os.makedirs(model_path)

device=torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')


run=wandb.init(project='baseline', name=run_name,
               config={'batch_size':10, 
                       'lr':3e-3,
                       'epoch':100,
                       'weight_decay':1e-3,
                       'device':device,
                       'transform':transform,
                       })
config = wandb.config

import segmentation_models_pytorch as smp


model=smp.Unet(encoder_name='resnet50', 
               encoder_weights='imagenet', 
               in_channels=3,
               classes=1, )
wandb.watch(model)
model=model.to(device)

optimizer=torch.optim.Adam(model.parameters(),lr=config.lr, weight_decay=config.weight_decay)
trainloader=DataLoader(train_data, batch_size=config.batch_size, num_workers=1)




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
    
    return dice_loss

#训练
if __name__ == '__main__':
    loss_last=99999
    best_model_name='x'

    for epoch in range(1,config.epoch+1):
        for step,(inputs,labels) in tqdm(enumerate(trainloader),
                                        desc=f"Epoch {epoch}/{config.epoch}",
                                        ascii=True, total=len(trainloader)):
            inputs,labels=inputs.to(device),labels.to(device)
            outputs=model(inputs)
            loss=dice_loss(outputs, labels)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            wandb.log({"Epoch": epoch, "DiceLoss": loss.item()})
            print(f"\nEpoch: {epoch}/{config.epoch},DiceLoss:{loss}")
        if loss<loss_last:
            loss_last=loss
            torch.save(model.state_dict(), model_path+'model_epoch_{}._loss{}.pth'.format(epoch,loss))
            best_model_name='model_epoch_{}._loss{}.pth'.format(epoch,loss)
            
run.finish()