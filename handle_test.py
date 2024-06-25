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
import matplotlib.pyplot as plt
from baseline_MAnet import model,MyDataset,device
from torchvision.utils import save_image
import datetime
from torchvision.transforms import ToPILImage,Resize
from torchvision.transforms.functional import resize
import segmentation_models_pytorch as smp

#device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
# 加载模型状态字典
model_path='modelMAnet_6.15_mit_b3_MAXP/model_epoch_399._loss0.1339769810438156.pth'
#state_dict = torch.load(model_path)

new_state_dict = torch.load(model_path)

# 创建一个新的状态字典，其中的键不包含 'module.' 前缀
#adjusted_state_dict = {k.replace('module.', ''): v for k, v in new_state_dict.items()}
model.load_state_dict(new_state_dict)
# 使用调整后的状态字典加载模型权重
#model.load_state_dict(adjusted_state_dict)
# 新建一个空字典
new_state_dict = {}

# 移除键中的 'module.' 前缀
'''for k, v in state_dict.items():
    name = k.replace('module.', '')  # 更简洁的方式来移除前缀
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
#model.load_state_dict(torch.load(f'{model_path}'))'''

test_path='test/'
#test_path='fusai_image/'

testdata=MyDataset(test_path)

#current_datetime = datetime.datetime.now().strftime('%m%d_%H%M')
#print("Current Date and Time:", current_datetime)
img_save_path='infers/'
# 检查目录是否存在，如果不存在则创建
if not os.path.exists(img_save_path):
    os.makedirs(img_save_path)

import torch
from torchvision.transforms import functional as F
from torchvision import transforms

# 定义图像预处理


def multi_scale_inference(inputs, model, device, scales):
    # 存储每个尺度下的预测结果
    predictions = []
    
    # 原始输入尺寸
    original_size = inputs.size()[2:]
    
    # 对每个尺度进行操作
    for scale in scales:
        # 计算新的尺寸
        scaled_size = [int(dim * scale) for dim in original_size]
        
        # 缩放图像
        resize_transform = Resize(scaled_size)
        inputs_scaled = resize_transform(inputs)
        
        # 确保输入维度匹配模型要求
        if len(inputs_scaled.shape) == 3:
            inputs_scaled = inputs_scaled.unsqueeze(0)
        
        # 将缩放后的图像送入模型
        inputs_scaled = inputs_scaled.to(device)
        with torch.no_grad():
            output_scaled = model(inputs_scaled)
        
        # 将预测结果调整回原始尺寸
        resize_back = Resize(original_size)
        output_resized_back = resize_back(output_scaled)
        
        # 收集预测结果
        predictions.append(output_resized_back)
    
    # 对所有尺度的预测结果取平均
    final_prediction = torch.mean(torch.stack(predictions), 0)
    
    return final_prediction

def sliding_window_inference(inputs, model, device, window_size, step_size):
    # 输入图像的尺寸
    _, _, height, width = inputs.shape
    # 输出预测图初始化为零
    full_output = torch.zeros((1, 1, height, width), device=device)
    # 计数器图，用于平均重叠区域的预测
    count_map = torch.zeros((1, 1, height, width), device=device)

    # 滑动窗口
    for y in range(0, height, step_size):
        for x in range(0, width, step_size):
            # 计算窗口的边界
            x1 = min(width, x + window_size[1])
            y1 = min(height, y + window_size[0])
            x0 = max(0, x1 - window_size[1])
            y0 = max(0, y1 - window_size[0])

            # 提取窗口
            window = inputs[:, :, y0:y1, x0:x1]

            # 如果窗口大小小于window_size，则调整大小
            if window.shape[2] != window_size[0] or window.shape[3] != window_size[1]:
                window = resize(window, window_size)

            # 模型预测
            window = window.to(device)
            with torch.no_grad():
                window_output = model(window)

            # 将预测结果放回全尺寸输出图中的相应位置
            if window_output.shape[2] != window_size[0] or window_output.shape[3] != window_size[1]:
                window_output = resize(window_output, (y1-y0, x1-x0))
            full_output[:, :, y0:y1, x0:x1] += window_output
            count_map[:, :, y0:y1, x0:x1] += 1

    # 使用计数图平均重叠区域
    full_output /= count_map

    return full_output

with torch.no_grad():
    for i,inputs in tqdm(enumerate(testdata)):
        #原始图片和标签
        inputs=inputs.reshape(1,3,320,640).to(device)
        # 输出生成的图像
        inputs_hflip = torch.flip(inputs, [3])  # 水平镜像
        inputs_vflip = torch.flip(inputs, [2])  # 垂直镜像
        
        out = model(inputs.view(1,3,320,640)) # 模型预测
        # 水平镜像图像预测
        #out_hflip = torch.flip(model(inputs_hflip), [3])
        # 垂直镜像图像预测
        #out_vflip = torch.flip(model(inputs_vflip), [2])
        #out_avg = (out + out_hflip + out_vflip) / 3
        #out_mutiscale=multi_scale_inference(inputs, model, device, scales=[0.5, 1.0, 1.5])

        #out_slidewindow=sliding_window_inference(inputs, model, device, window_size=(256, 256), step_size=64)


        threshold=0.5
        #out1= torch.where(out_avg>threshold, torch.tensor(255,dtype=torch.float).to(device),out)
        #out1= torch.where(out1<= threshold, torch.tensor(0,dtype=torch.float).to(device),out)
        #out_mutiscale= torch.where(out_mutiscale >threshold, torch.tensor(255,dtype=torch.float).to(device),out)
        #out_mutiscale= torch.where(out_mutiscale <= threshold, torch.tensor(0,dtype=torch.float).to(device),out)
        #out_slidewindow= torch.where(out_slidewindow >threshold, torch.tensor(255,dtype=torch.float).to(device),out)
        #out_slidewindow= torch.where(out_slidewindow <= threshold, torch.tensor(0,dtype=torch.float).to(device),out)
        #out0=(out_avg+out_mutiscale+out_slidewindow)/3
        out= torch.where(out>threshold, torch.tensor(255,dtype=torch.float).to(device),torch.tensor(0,dtype=torch.float).to(device))
        
        #out_binary = torch.where(out_avg > threshold, torch.tensor(1, dtype=torch.float).to(device), torch.tensor(0, dtype=torch.float).to(device))
        #out_mutiscale_binary = torch.where(out_mutiscale > threshold, torch.tensor(1, dtype=torch.float).to(device), torch.tensor(0, dtype=torch.float).to(device))
        #out_slidewindow_binary = torch.where(out_slidewindow > threshold, torch.tensor(1, dtype=torch.float).to(device), torch.tensor(0, dtype=torch.float).to(device))
        
        # 对每个像素位置进行投票
        #votes = out_binary + out_mutiscale_binary + out_slidewindow_binary

    # 根据投票结果决定最终输出
        #out_final = torch.where(out>= 2, torch.tensor(255, dtype=torch.float).to(device), torch.tensor(0, dtype=torch.float).to(device))
        #out_final = torch.where(votes > 500, torch.tensor(255, dtype=torch.float).to(device), torch.tensor(0, dtype=torch.float).to(device))
        # 如果一个位置的值大于等于2，则该位置的最终输出为255；否则为0
        #final_output = torch.where(votes >= 2, torch.tensor(255, dtype=torch.float).to(device), torch.tensor(0, dtype=torch.float).to(device))
        
        out2= out.detach().cpu().numpy().reshape(1,320,640)
        #保存为1位图提交
        img = Image.fromarray(out2[0].astype(np.uint8))
        img = img.convert('1')
        img.save(img_save_path + testdata.name[i])

    
   

   

#对保存的图像进行打包
import zipfile

def zip_files(file_paths, output_path):
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in file_paths:
            zipf.write(file)
            
#打包图片
file_paths = [img_save_path+i for i in os.listdir(img_save_path) if i[-3:]=='png']
output_path = 'infer.zip'
zip_files(file_paths, output_path)