import cv2
import numpy as np
import glob
import os

def read_images_from_folder(folder_path):
    # 获取文件夹中所有图片路径，使用glob获取所有png或jpg文件
    image_paths = glob.glob(os.path.join(folder_path, '*.png'))
    
    # 排序确保按照文件名的顺序读取图片
    image_paths = sorted(image_paths)

    images = []
    for path in image_paths:
        # 读取图片
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        
        # 如果图片为空（可能是损坏的图片），跳过
        if img is None:
            print(f"Warning: {path} could not be loaded.")
            continue
        
        # 将图片归一化到0-1之间
        img_normalized = img.astype(np.float32) / 255.0
        
        images.append(img_normalized)
        print(f'Read and normalized: {path}')

    return images

# 使用示例
folder_path = '/media/vrlab/rabbit/print3dingp/torch-ngp/workspace/fox/print_volume_s/ngp_613'
# folder_path = '/media/vrlab/rabbit/print3dingp/torch-ngp/workspace/lego/print_volume_s/ngp_300'
images = read_images_from_folder(folder_path)
images = images[::-1]

result_image = np.zeros_like(images[0])
print(result_image.shape)
for i in range(len(images)):
    result_image[:, :, :3] += images[i][:, :, :3] * (1-result_image[:, :, [3]]) * images[i][:, :, [3]]
    result_image[:, :, [3]] = result_image[:, :, [3]] + (1 - result_image[:, :, [3]]) * images[i][:, :, [3]]


cv2.imwrite("workspace/fox_p_s.png", result_image*255,)