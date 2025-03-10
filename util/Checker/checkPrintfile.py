# 检查打印尺寸，检查颜色是否只有CMYK
from colorama import Fore
import numpy as np
from PIL import Image
import sys

if len(sys.argv) != 2:
    print("Usage: python whichColors.py <path_to_tif_file>")
    sys.exit(1)
    
    
tif_file_path = sys.argv[1]
img = Image.open(tif_file_path)
img_array = np.array(img)
img_width, img_height = img.size
# 将所有 [255, 255, 255, *] 转换为 [255, 255, 255, 0]
# 这里假设 * 是任何值
mask = (img_array[:, :, 0] == 255) & (img_array[:, :, 1] == 255) & (img_array[:, :, 2] == 255)
img_array[mask] = [0, 0, 0, 255]
pixels = img_array.reshape(-1, img_array.shape[-1])

unique_pixels = np.unique(pixels, axis=0)

if len(unique_pixels) != 5:
    print()
    print(Fore.RED +"Error: 颜色数量不对: " + str(len(unique_pixels)))
    sys.exit(1)

print(Fore.GREEN +"颜色数量正确")

z = 0.014
y = z * 2
x = y * 3

print(f"x: {img_width*x:.2f} mm, y: {img_height*y:.2f} mm")



