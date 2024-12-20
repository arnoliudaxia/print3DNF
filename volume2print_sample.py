import argparse
import os
from glob import glob
import cv2
import numpy as np
from tqdm import tqdm
import random


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
    parser.add_argument('--output_folder', type=str, default='workspace/lego/print_volume_s/ngp_300')
    opt = parser.parse_args()


    os.makedirs(opt.output_folder, exist_ok=True)

    
    image_paths = glob(os.path.join(opt.input_folder, '*.png'))
    
    # 排序确保按照文件名的顺序读取图片
    image_paths = sorted(image_paths)
    
    volume_images = []
    for path in image_paths:
        # 读取图片
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        
        # 如果图片为空（可能是损坏的图片），跳过
        if img is None:
            print(f"Warning: {path} could not be loaded.")
            continue
        
        # 将图片归一化到0-1之间
        img_normalized = img.astype(np.float32) / 255.0
        
        image_cmykw = rgba_to_cmykwa_image(img_normalized[:, :, :3])
        image_cmykwa = np.concatenate([image_cmykw, img_normalized[:, :, [3]]], axis=2)
        
        volume_images.append(image_cmykwa)
        print(f'Read and normalized: {path}')

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

    for z in tqdm(range(0, volume_images.shape[0])):
        for x in range(0, volume_images.shape[1]):
            for y in range(0, volume_images.shape[2]):
                is_clear = False if random.random() < volume_images[z, x, y, 5] else True
                if is_clear:
                    print_images[z, x, y] = np.array([0, 0, 0, 0])
                else:
                    normalized_weights = volume_images[z, x, y, 0:5] / volume_images[z, x, y, 0:5].sum()
                    cmykw_color = np.random.choice(len(normalized_weights), p=normalized_weights)
                    print_images[z, x, y] = np.array(cmykw_save_color[cmykw_color])
    




    for i, path in enumerate(image_paths):
        base_name = os.path.basename(path)
        save_path = os.path.join(opt.output_folder, base_name)
        cv2.imwrite(save_path, print_images[i]*255)
        