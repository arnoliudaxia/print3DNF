import argparse
import os
from glob import glob
import cv2
import numpy as np
from tqdm import tqdm
import random

import json


kw = 3.93 # 测量值
kk = 3.24 # 测量值
dz = 0.014 # 层高

def rgba_to_cmykwa_image(image):
    
    # 提取 RGBA 通道
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]
    
    # Step 1: Normalize RGB
    # 已经在前面通过 / 255.0 做了归一化

    # Step 2: Calculate initial CMY
    C = 1 - R
    M = 1 - G
    Y = 1 - B

    # Step 3: Determine K (Key/Black)
    K = np.minimum(np.minimum(C, M), Y)

    # Step 4: Adjust CMY based on K
    non_zero_mask = (1 - K) != 0
    C[non_zero_mask] = (C[non_zero_mask] - K[non_zero_mask]) / (1 - K[non_zero_mask])
    M[non_zero_mask] = (M[non_zero_mask] - K[non_zero_mask]) / (1 - K[non_zero_mask])
    Y[non_zero_mask] = (Y[non_zero_mask] - K[non_zero_mask]) / (1 - K[non_zero_mask])

    # Step 5: Calculate W (using max method)
    W = 1 - np.maximum(np.maximum(C, M), np.maximum(Y, K))

    # Step 7: Clamp all values to [0, 1]
    C = np.clip(C, 0, 1)
    M = np.clip(M, 0, 1)
    Y = np.clip(Y, 0, 1)
    K = np.clip(K, 0, 1)
    W = np.clip(W, 0, 1)
    
    # 创建最终的五通道图像
    result = np.stack([C, M, Y, K, W], axis=-1)
    
    
    return result





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, default='workspace/lego/volume/ngp_300')
    parser.add_argument('--output_folder', type=str, default='workspace/lego/print_volume_s_1225/ngp_300')
    opt = parser.parse_args()


    os.makedirs(opt.output_folder, exist_ok=True)

    
    image_paths = glob(os.path.join(opt.input_folder, '*.npy'))
    
    with open(os.path.join(opt.input_folder, 'params.json'), "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    
    delta = data['delta']
    
    
    # 排序确保按照文件名的顺序读取图片
    image_paths = sorted(image_paths)
    
    print("Loading Images...")
    volume_images = []
    for path in tqdm(image_paths):
        # 读取图片
        img = np.load(path)
        
        # 如果图片为空（可能是损坏的图片），跳过
        if img is None:
            print(f"Warning: {path} could not be loaded.")
            continue
        
        # 将图片归一化到0-1之间
        img_normalized = img
        
        image_cmykw = rgba_to_cmykwa_image(img_normalized[:, :, :3])
        image_cmykwa = np.concatenate([image_cmykw, img_normalized[:, :, [3]]], axis=2)
        
        volume_images.append(image_cmykwa)
        
        
    print("Calculating...")

    volume_images = np.stack(volume_images, axis=0)

    print_images = np.zeros((volume_images.shape[0], volume_images.shape[1], volume_images.shape[2], 4))


    pure_cmykw_colors = [
        [1, 0, 0, 0, 0],  # 纯青色
        [0, 1, 0, 0, 0],  # 纯品红
        [0, 0, 1, 0, 0],  # 纯黄色
        [0, 0, 0, 1, 0],  # 纯黑色
        [0, 0, 0, 0, 1],  # 纯白色
    ]

    cmykw_save_color = [
        [0, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [0, 0, 0, 1],
        [1, 1, 1, 1],
    ]

    # for z in tqdm(range(0, volume_images.shape[0])):
    #     for x in range(0, volume_images.shape[1]):
    #         for y in range(0, volume_images.shape[2]):
                
    #             k_rate = volume_images[z, x, y, 3] / (volume_images[z, x, y, 3] + volume_images[z, x, y, 4] + 1e-10)
    #             w_rate = volume_images[z, x, y, 4] / (volume_images[z, x, y, 3] + volume_images[z, x, y, 4] + 1e-10)
    #             concentration = (delta * volume_images[z, x, y, 5]) / (dz * (k_rate * kk + w_rate * kw + 1e-10))
    #             k_concentration, w_concentration = k_rate * concentration, w_rate * concentration
    #             # print(k_concentration, w_concentration, k_rate, w_rate)
    #             # continue
    #             is_clear_random = random.random()
    #             if is_clear_random < w_concentration:
    #                 is_clear = False
    #                 print_images[z, x, y] = np.array(cmykw_save_color[4])
    #             elif w_concentration < is_clear_random and is_clear_random < w_concentration + k_concentration:
    #                 is_clear = False
    #                 print_images[z, x, y] = np.array(cmykw_save_color[3])
    #             else:
    #                 is_clear = True
                
    #             # is_clear = False if random.random() < volume_images[z, x, y, 5] else True
    #             if is_clear or volume_images[z, x, y, 0:3].sum() == 0:
    #                 print_images[z, x, y] = np.array([0, 0, 0, 0])
    #             else:
    #                 normalized_weights = volume_images[z, x, y, 0:3] / volume_images[z, x, y, 0:3].sum()
    #                 cmykw_color = np.random.choice(len(normalized_weights), p=normalized_weights)
    #                 print_images[z, x, y] = np.array(cmykw_save_color[cmykw_color])
    

    z_size, x_size, y_size, _ = volume_images.shape

    # 预计算 k_rate 和 w_rate
    denominator = volume_images[..., 3] + volume_images[..., 4] + 1e-10
    k_rate = volume_images[..., 3] / denominator
    w_rate = volume_images[..., 4] / denominator

    # 计算浓度
    kw_concentration = (delta * volume_images[..., 5]) / (dz * (k_rate * kk + w_rate * kw + 1e-10))
    kw_concentration[kw_concentration > 0.8] = 0.8 # clip_magicnum
    k_concentration = k_rate * kw_concentration
    w_concentration = w_rate * kw_concentration

    # 生成随机数矩阵
    is_clear_random = np.random.rand(z_size, x_size, y_size)

    # 判断是否 clear
    is_clear = (is_clear_random >= w_concentration) & (is_clear_random >= w_concentration + k_concentration)

    # 初始化结果数组
    print_images = np.zeros((z_size, x_size, y_size, 4), dtype=np.uint8)

    # 填充非 clear 部分
    not_clear_mask = ~is_clear & (volume_images[..., 0:3].sum(axis=-1) > 0)
    normalized_weights = volume_images[..., 0:3] / (volume_images[..., 0:3].sum(axis=-1, keepdims=True) + 1e-10)
    
    denominator_cmy = volume_images[..., 0] + volume_images[..., 1] + volume_images[..., 2] + 1e-10
    c_rate = volume_images[..., 0] / denominator_cmy
    m_rate = volume_images[..., 1] / denominator_cmy
    y_rate = volume_images[..., 2] / denominator_cmy
    
    
    cmy_concentration = (volume_images[..., 0] + volume_images[..., 1] + volume_images[..., 2]) / (volume_images[..., 3] + volume_images[..., 4] + 1e-10) * kw_concentration
    
    cmy_concentration[(cmy_concentration + kw_concentration) > 1] = (1 - kw_concentration[(cmy_concentration + kw_concentration) > 1]) # clip_magicnum
    cmy_concentration[cmy_concentration > 2*kw_concentration] = 2*kw_concentration[cmy_concentration > 2*kw_concentration] # clip_magicnum
    cmy_concentration[cmy_concentration < 0] = 0                                         # clip_magicnum
    c_concentration = cmy_concentration * c_rate
    m_concentration = cmy_concentration * m_rate
    y_concentration = cmy_concentration * y_rate
    
    
    color_random = np.random.rand(z_size, x_size, y_size)
    
    c_mask = (color_random < c_concentration)
    m_mask = (c_concentration <= color_random) & (color_random < c_concentration + m_concentration)
    y_mask = (c_concentration + m_concentration <= color_random) & (color_random < c_concentration + m_concentration + y_concentration)
    w_mask = (c_concentration + m_concentration + y_concentration <= color_random) & (color_random < c_concentration + m_concentration + y_concentration + w_concentration)
    k_mask = (c_concentration + m_concentration + y_concentration + w_concentration <= color_random) & (color_random < c_concentration + m_concentration + y_concentration + w_concentration + k_concentration)
    
    
    # random_indices = np.random.choice(len(cmykw_save_color), size=(z_size, x_size, y_size), p=normalized_weights.reshape(-1, 3))

    # 填充颜色
    # print_images[not_clear_mask] = np.array(cmykw_save_color)[random_indices[not_clear_mask]]

    # 填充 clear 部分
    # w_mask = (is_clear_random < w_concentration)
    # k_mask = (is_clear_random >= w_concentration) & (is_clear_random < w_concentration + k_concentration)
    # cl_mask = (w_concentration + k_concentration <= is_clear_random)

    print_images[c_mask] = np.array(cmykw_save_color[0])
    print_images[m_mask] = np.array(cmykw_save_color[1])
    print_images[y_mask] = np.array(cmykw_save_color[2])
    print_images[k_mask] = np.array(cmykw_save_color[3])
    print_images[w_mask] = np.array(cmykw_save_color[4])



    print("Saving Images...")

    for i, path in enumerate(tqdm(image_paths)):
        base_name = os.path.basename(path).replace('npy','png')
        save_path = os.path.join(opt.output_folder, base_name)
        cv2.imwrite(save_path, print_images[i]*255)
        