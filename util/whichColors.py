"""
图像颜色分析工具

这个脚本用于分析 TIF 图像中的唯一颜色组合。它可以：
- 读取 TIF 格式的图像文件
- 将图像转换为 numpy 数组进行处理
- 识别并统计图像中所有唯一的颜色组合
- 输出图像的基本信息和颜色分析结果

用法:
    直接运行脚本，确保指定的 TIF 文件路径正确。

依赖:
    - numpy
    - PIL (Python Imaging Library)
"""

import numpy as np
from PIL import Image
import sys

if len(sys.argv) != 2:
    print("Usage: python whichColors.py <path_to_tif_file>")
    sys.exit(1)

tif_file_path = sys.argv[1]
img = Image.open(tif_file_path)

# 转换为numpy数组
img_array = np.array(img)
# 将所有 [255, 255, 255, *] 转换为 [255, 255, 255, 0]
# 这里假设 * 是任何值
mask = (img_array[:, :, 0] == 255) & (img_array[:, :, 1] == 255) & (img_array[:, :, 2] == 255)
img_array[mask] = [255, 255, 255, 0]
# 将图像数组重塑为二维数组，每行代表一个像素的所有通道值
pixels = img_array.reshape(-1, img_array.shape[-1])

# 获取唯一的像素组合
unique_pixels = np.unique(pixels, axis=0)

# 定义颜色映射
def get_color_name(pixel):
    """将像素值映射到对应的颜色名称"""
    pixel_tuple = tuple(pixel)
    color_map = {
        (0, 0, 0, 0): 'W',      # White
        (0, 0, 255, 0): 'Y',    # Yellow
        (0, 255, 0, 0): 'M',    # Magenta
        (255, 0, 0, 0): 'C',    # Cyan
        (255, 255, 255, 0): 'B', # Black
        (0, 0, 0, 255): 'B' # Black
    }
    return color_map.get(pixel_tuple, '')

# 打印结果
print(f"图像形状: {img_array.shape}")
print(f"唯一像素组合数量: {len(unique_pixels)}")
print("唯一像素值:")
for pixel in unique_pixels:
    color_name = get_color_name(pixel)
    if color_name:
        print(f"{pixel} - {color_name}")
    else:
        print(pixel)