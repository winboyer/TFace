import os
import os.path as osp
import torch
import torch.nn as nn
import torch.optim as opti
from tqdm import tqdm
import torchvision.transforms as T
from generate_pseudo_labels.extract_embedding.model import model
import numpy as np
from scipy import stats
import pdb
from PIL import Image
import time
import cv2

def read_img(imgPath):     # read image & data pre-process
    data = torch.randn(1, 3, 112, 112)
    transform = T.Compose([
        T.Resize((112, 112)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img = Image.open(imgPath).convert("RGB")
    data[0, :, :, :] = transform(img)
    return data


def network(eval_model, device):
    net = model.R50([112, 112], use_type="Qua").to(device)
    net_dict = net.state_dict()     
    data_dict = {
        key.replace('module.', ''): value for key, value in torch.load(eval_model, map_location=device).items()}
    net_dict.update(data_dict)
    net.load_state_dict(net_dict)
    net.eval()
    return net

if __name__ == "__main__":
#     device = 'cpu'                                        # 'cpu' or 'cuda:x'
    device = 'cuda:1'                                        # 'cpu' or 'cuda:x'
#     eval_model = './model/SDD_FIQA_checkpoints_r50.pth'   # checkpoint
    eval_model = '/root/jinyfeng/models/faceQuality/SDD_FIQA_checkpoints_r50.pth'   # checkpoint
    net = network(eval_model, device)
#     img_folder = '/root/jinyfeng/datas/20230801_v2_align_faces_v2'
#     save_folder = '/root/jinyfeng/datas/20230801_v2_align_faces_v2_TFace'
    
    img_folder = '/root/jinyfeng/datas/20230801_v2_crop_faces'
    save_folder = '/root/jinyfeng/datas/20230801_v2_crop_faces_TFace'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    filelist = os.listdir(img_folder)
    print(len(filelist))
    total_time = 0.0
    blur_scores_list = []
    for idx, filename in enumerate(filelist):
#         print(idx, filename)
        if not filename.endswith(".jpg"):
            continue
            
        if (idx+1)%1000 == 0:
            print("processed {} images".format(idx+1))
            
        image_path = os.path.join(img_folder, filename)
        frame = cv2.imread(image_path)
        input_data = read_img(image_path)
        time_s = time.time()
        if device == 'cpu':
            pred_score = net(input_data).data.cpu().numpy().squeeze()
        else:
            input_data = input_data.to(device)
            pred_score = net(input_data).data.cpu().numpy().squeeze()
        time_e = time.time()
        if idx!=0:
            total_time +=(time_e-time_s)
#         print('cost time ============ ', time_e-time_s)
        
        blur_label = f"{pred_score:.3f}"
        save_filepath = os.path.join(save_folder, blur_label+'_'+filename)
        cv2.imwrite(save_filepath, frame)
        
#         print(f"Quality score = {blur_label}")
        blur_label = float(blur_label)
        blur_scores_list.append(blur_label)
        
    print('total_time=========',total_time)
    print('avg_time=========',total_time/(idx-1))
    print(type(blur_scores_list))        
    list_max = np.max(blur_scores_list)
    list_min = np.min(blur_scores_list)
    list_mean = np.mean(blur_scores_list)
    blur_scores_list.sort()
    list_mid = blur_scores_list[len(blur_scores_list)//2]
    print('list_min, list_mean, list_mid, list_max===========', list_min, list_mean, list_mid, list_max)
        
