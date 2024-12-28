import cv2
import numpy as np
import glob
import os
import json
from tqdm import tqdm
import mixbox

def read_images_from_folder(folder_path):
    # 获取文件夹中所有图片路径，使用glob获取所有png或jpg文件
    image_paths = glob.glob(os.path.join(folder_path, '*.png'))
    
    # 排序确保按照文件名的顺序读取图片
    image_paths = sorted(image_paths)

    images = []
    for path in tqdm(image_paths):
        # 读取图片
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        
        # 如果图片为空（可能是损坏的图片），跳过
        if img is None:
            print(f"Warning: {path} could not be loaded.")
            continue
        
        
        
        images.append(img)

    return images

# 使用示例
folder_path = '/media/vrlab/rabbit/print3dingp/print_ngp/workspace/fox/print_volume/ngp_50'

color_calib_file = '/media/vrlab/rabbit/print3dingp/print_ngp/color_calib.json'
with open(color_calib_file, 'r') as f:
    color_calib_data = json.load(f)

print(color_calib_data)

images = read_images_from_folder(os.path.join(folder_path, 'index'))
images = images[::-1]

result_image = np.zeros((images[0].shape[0], images[0].shape[1], 4), dtype=np.float32)
for i in tqdm(range(len(images))):
    color = np.zeros((images[0].shape[0], images[0].shape[1], 3), dtype=np.float32)
    alpha = np.zeros((images[0].shape[0], images[0].shape[1], 1), dtype=np.float32)
    
    color_map = np.array(color_calib_data['color'], dtype=np.float32)
    color = color_map[images[i]]
    density_map = np.array(color_calib_data['density'], dtype=np.float32)
    alpha = 1 - np.exp(-1 * density_map[images[i]] * 0.014)
    
    result_image[:, :, :3] += color * (1-result_image[:, :, [3]]) * alpha
    result_image[:, :, [3]] = result_image[:, :, [3]] + (1 - result_image[:, :, [3]]) * alpha
    cv2.imwrite('temp.png', result_image[:, :, [2,1,0]]*255)




save_path = os.path.join(folder_path, 'preview.png')
result_image[:, :, 0:3] = cv2.cvtColor(result_image[:, :, 0:3], cv2.COLOR_RGB2BGR)
cv2.imwrite(save_path, result_image[:, :, 0:3]*255)
print("saved to", save_path)