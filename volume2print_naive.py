import argparse
import os
from glob import glob
import cv2
import numpy as np
from tqdm import tqdm
import random

voxel_group = [2, 4, 12]



def cmykw_to_rgb(cmykw):
    """
    将单个 cmykw 颜色转为 RGB 颜色
    """
    C, M, Y, K, W = cmykw
    R = (1 - C) * (1 - K) + W
    G = (1 - M) * (1 - K) + W
    B = (1 - Y) * (1 - K) + W
    return np.array([R, G, B])



def greedy_fill_pure_cmykw(target_rgb, pure_cmykw_colors, n):
    """
    使用贪心算法填充 N 个纯 cmykw 颜色组合
    输入:
        target_rgb: 目标 RGB 颜色 (长度为 3 的数组，值域为 [0, 1])
        pure_cmykw_colors: 纯 cmykw 颜色列表，每个元素是长度为 6 的数组
        n: 必须选择的颜色数量
    输出:
        selected_indices: 被选中的颜色索引 (允许重复)
        final_rgb: 最终混合后的 RGB 颜色
    """
    rgb_colors = np.array([cmykw_to_rgb(c) for c in pure_cmykw_colors])
    current_rgb = np.zeros(3)  # 初始化混合颜色
    selected_indices = []  # 已选颜色索引

    for j in range(n):
        best_index = -1
        min_error = float('inf')

        # 遍历所有纯 cmykw 颜色，找到使误差最小的颜色
        for i in range(len(rgb_colors)):
            # 计算添加该颜色后的混合结果
            candidate_rgb = ((current_rgb * i) + rgb_colors[i]) / (i+1)
            error = np.abs(candidate_rgb - target_rgb).sum()

            # 更新最佳选择
            if error < min_error:
                min_error = error
                best_index = i

        # 选择最佳颜色并更新当前混合结果
        selected_indices.append(best_index)
        current_rgb = (current_rgb * j + rgb_colors[best_index]) / (j+1)
    return selected_indices, current_rgb


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, default='workspace/lego/volume/ngp_300')
    parser.add_argument('--output_folder', type=str, default='workspace/lego/print_volume/ngp_300')
    opt = parser.parse_args()


    os.makedirs(opt.output_folder, exist_ok=True)

    
    image_paths = glob(os.path.join(opt.input_folder, '*.png'))
    
    # 排序确保按照文件名的顺序读取图片
    image_paths = sorted(image_paths)
    
    volume_images = []
    print_images = []
    for path in image_paths:
        # 读取图片
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        
        # 如果图片为空（可能是损坏的图片），跳过
        if img is None:
            print(f"Warning: {path} could not be loaded.")
            continue
        
        # 将图片归一化到0-1之间
        img_normalized = img.astype(np.float32) / 255.0
        
        volume_images.append(img_normalized)
        print_images.append(np.zeros_like(img_normalized))
        print(f'Read and normalized: {path}')

    volume_images = np.stack(volume_images, axis=0)
    print_images = np.stack(print_images, axis=0)


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

    for z in tqdm(range(0, volume_images.shape[0], voxel_group[2])):
        for x in range(0, volume_images.shape[1], voxel_group[0]):
            for y in range(0, volume_images.shape[2], voxel_group[1]):
                # if x < 240 or x > 280:
                #     continue
                # if y < 220 or y > 250:
                #     continue
                voxel = volume_images[z:z+voxel_group[2], x:x+voxel_group[0], y:y+voxel_group[1]]
                mean_color = np.sum(voxel[..., :3] * voxel[..., 3:4], axis=(0, 1, 2)) / np.sum(voxel[..., 3], axis=(0, 1, 2)) if np.sum(voxel[..., 3]) > 0 else np.zeros(3)  # 计算 RGB 的平均值
                mean_alpha = np.mean(voxel[..., 3])  
                
                fill_count = int(np.floor(mean_alpha * np.prod(voxel.shape[:3])))
                



                selected_indices, final_rgb = greedy_fill_pure_cmykw(mean_color, pure_cmykw_colors, fill_count)

                fill_region = np.zeros_like(voxel)

                # print(final_rgb, mean_color)

                fill_region = fill_region.reshape(-1, voxel.shape[-1])
                fill_list = list(range(len(fill_region)))
                random.shuffle(fill_list)
                for i in range(fill_count):
                    fill_region[fill_list[i], :] = cmykw_save_color[selected_indices[i]]
                fill_region = fill_region.reshape(voxel.shape)
                
                # 将填充结果写入到 print_images 的对应区域
                print_images[z:z+voxel_group[2], x:x+voxel_group[0], y:y+voxel_group[1]] = fill_region
                


    for i, path in enumerate(image_paths):
        base_name = os.path.basename(path)
        save_path = os.path.join(opt.output_folder, base_name)
        cv2.imwrite(save_path, print_images[i]*255)
        