import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import sys
import os
from matplotlib.colors import ListedColormap
import argparse
import multiprocessing
from multiprocessing import Pool
from functools import partial

# 将绝对路径改为相对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(os.path.join(parent_dir, "util"))

import printer.printerICM as icm
from Graph.drawMarker import draw_mark

# 定义处理单个切片的函数
def process_slice(slice_data, srgb_profile_path, cmyk_profile_path):
    rgb_slice = slice_data.copy()
    
    rgb_slice[:, :, 0] = slice_data[:, :, 2]  # R = B
    rgb_slice[:, :, 2] = slice_data[:, :, 0]  # B = R
    # G通道保持不变
    slice_uint8 = (rgb_slice * 255).astype(np.uint8)
    img = Image.fromarray(slice_uint8)
    output_files = icm.convert_color_profile(
        img,
        modes=[1],
        srgb_profile_path=srgb_profile_path,
        cmyk_profile_path=cmyk_profile_path,
        output_mode="CMYK",
        save_files=False
    )
    return np.array(output_files[0])

# 添加命令行参数解析
parser = argparse.ArgumentParser(description='体积数据切片，注意是RGB还是BGR ')
parser.add_argument('--volume_path', type=str, required=True,
                    help='体积数据的路径 (.npz 或 .npy 格式)')
parser.add_argument('--mark_layer', choices=['1', '-1', 'none'], default='none',
                    help='在哪一层添加标记（默认不添加）：1=首层，-1=最后一层，none=不添加')
parser.add_argument('--mark_type', choices=['box', 'circle'], default='box',
                    help='标记类型（默认方框）：box=方框，circle=圆形')
parser.add_argument('--mark_size', type=float, default=0.25,
                    help='标记大小（相对于图像尺寸的比例，默认0.25）')
parser.add_argument('--mark_width', type=int, default=10,
                    help='标记线条宽度（像素）')
parser.add_argument('--num_processes', type=int, default=None,
                    help='使用的进程数（默认为CPU核心数）')

args = parser.parse_args()

# 使用命令行参数加载数据
if args.volume_path.endswith('.npz'):
    loaded_data = np.load(args.volume_path)
    volume = loaded_data['volume']
elif args.volume_path.endswith('.npy'):
    volume = np.load(args.volume_path)
else:
    raise ValueError("不支持的文件格式，请使用.npz或.npy文件")

# 添加标记（如果需要）
if (args.mark_layer != 'none'):
    print(f"在第 {args.mark_layer} 层添加{args.mark_type}标记")
    layer_index = int(args.mark_layer)
    volume[layer_index] = draw_mark(
        volume[layer_index],
        mark_type=args.mark_type,
        size_ratio=args.mark_size,
        line_width=args.mark_width
    )
    
volume_icc = np.zeros((volume.shape[0], volume.shape[1], volume.shape[2], 4))

# 设置进程数
num_processes = args.num_processes if args.num_processes is not None else multiprocessing.cpu_count()
print(f"使用 {num_processes} 个进程进行并行处理")

# 准备ICC配置文件路径
srgb_profile_path = os.path.join(parent_dir, os.path.join(parent_dir, "util", "printer", "icc", "AdobeRGB1998.icc"))
cmyk_profile_path = os.path.join(parent_dir, os.path.join(parent_dir, "util", "printer", "icc", "Stratasys_J8_7xx_VeroUltraWhite_HT3_VividCMYK.icm"))

# 使用进程池进行并行处理
with Pool(num_processes) as pool:
    # 创建偏函数，固定ICC配置文件路径参数
    process_slice_partial = partial(process_slice, 
                                  srgb_profile_path=srgb_profile_path,
                                  cmyk_profile_path=cmyk_profile_path)
    
    # 使用imap处理切片并显示进度条
    for i, result in enumerate(tqdm(
        pool.imap(process_slice_partial, [volume[i] for i in range(volume.shape[0])]),
        total=volume.shape[0],
        desc="icc correcting..."
    )):
        volume_icc[i] = result

    
volume_icc_halftone = np.zeros((volume_icc.shape[0], volume_icc.shape[1], volume_icc.shape[2]))

def white_weight_func(cmyk_sum):
    # return np.exp(-cmyk_sum * 5)  # 指数衰减函数
    white_threshold = 0.8*255  # 可调整的阈值
    white_weight = np.maximum(0, white_threshold - cmyk_sum)
    return white_weight

for i in tqdm(range(volume_icc.shape[0]), desc="halftone sampling..."):
    cmyk_slice = volume_icc[i]
    # 创建采样结果数组
    height, width, channels = cmyk_slice.shape
    sampled_image = np.zeros((height, width), dtype=np.uint8)
    # 确保所有值都是非负的
    cmyk_values = np.maximum(cmyk_slice, 0)
    # 计算CMYK的总和
    cmyk_sum = np.sum(cmyk_values, axis=-1)
    
    #! 检查是否在前10层或最后10层
    # is_boundary_layer = (i < 20) or (i >= volume_icc.shape[0] - 20)
    is_boundary_layer=False
    
    
    if is_boundary_layer:
        # 前10层和最后10层：只使用CMYK通道，不使用白色
        cmyk_w_values = cmyk_values  # 只有4个通道 [C, M, Y, K]
        
        # 计算总和（只包括CMYK通道）
        totals = np.sum(cmyk_w_values, axis=-1)
        
        # 生成随机数数组
        r = np.random.random((height, width)) * totals
        
        # 计算累积和
        cumulative = np.cumsum(cmyk_w_values, axis=-1)
        
        # 初始化采样结果为全0
        sampled_image = np.zeros((height, width), dtype=np.uint8)
        for k in range(1, 5):  # 从通道1开始，因为通道0是默认值
            sampled_image = np.where(r >= cumulative[:, :, k-1], k, sampled_image)
        
        # 处理总和为0的情况（将其分配为5，即白色）
        zero_totals = (totals == 0)
        if np.any(zero_totals):
            # 对于总和为0的像素，分配为5（白色）
            sampled_image = np.where(zero_totals, 5, sampled_image)

    else:
        # 中间层：使用CMYK和白色通道
        # 定义白色通道的权重
        white_weight = white_weight_func(cmyk_sum)
        
        # 创建包含白色通道的5通道数组 [C, M, Y, K, White]
        cmyk_w_values = np.zeros((height, width, 5))
        cmyk_w_values[:, :, :4] = cmyk_values  # 前4个通道是CMYK
        cmyk_w_values[:, :, 4] = white_weight  # 第5个通道是白色
        
        # 计算总和（包括白色通道）
        totals = np.sum(cmyk_w_values, axis=-1)
        
        # 生成随机数数组 (0到1之间的随机数乘以总和)
        r = np.random.random((height, width)) * totals
        
        # 计算累积和
        cumulative = np.cumsum(cmyk_w_values, axis=-1)
        
        # 初始化采样结果为全0
        sampled_image = np.zeros((height, width), dtype=np.uint8)
        for k in range(1, 6):  # 从通道1开始，因为通道0是默认值
            sampled_image = np.where(r >= cumulative[:, :, k-1], k, sampled_image)
        
        # 处理总和为0的情况
        zero_totals = (totals == 0)
        if np.any(zero_totals):
            print(f"Layer {i}: ZERO!!")
    
    volume_icc_halftone[i] = sampled_image

# 可视化采样结果
def previewHalftone(sampled_image, cmyk_slice): 
    # 更新颜色映射以包含白色
    cmap = plt.cm.colors.ListedColormap(['cyan', 'magenta', 'yellow', 'black', 'white', "grey"])
    bounds = [0, 1, 2, 3, 4, 5, 6]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(12, 8))
    plt.imshow(sampled_image, cmap=cmap, norm=norm)
    plt.colorbar(ticks=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], label='')
    plt.clim(-0.5, 5.5)
    plt.title('CMYKW Halftone')
    plt.axis('off')
    # 保存halftone图
    halftone_path = os.path.join(SaveFolder, 'halftone_preview.png')
    plt.savefig(halftone_path)
    print(f"已保存Halftone预览图到: {halftone_path}")
    plt.show()

    # 显示原始CMYK值和计算的白色通道值的分布
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    channel_names = ['C (River)', 'M (Flower)', 'Y (Yellow)', 'K (Black)', "White"]

    # 原始CMYK切片
    cmyk_values = np.maximum(cmyk_slice, 0)

    # 计算CMYK的总和
    cmyk_sum = np.sum(cmyk_values, axis=-1)

    # 显示CMYK通道
    for i in range(4):
        im = axes[i].imshow(cmyk_values[:, :, i], cmap='viridis')
        axes[i].set_title(f'{channel_names[i]}')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i])

    # 显示白色通道
    white_weight = white_weight_func(cmyk_sum)
    im = axes[4].imshow(white_weight, cmap='viridis')
    axes[4].set_title(f'{channel_names[4]}')
    axes[4].axis('off')
    plt.colorbar(im, ax=axes[4])

    plt.tight_layout()
    # 保存channels图
    channels_path = os.path.join(SaveFolder, 'channels_preview.png')
    plt.savefig(channels_path)
    print(f"已保存通道预览图到: {channels_path}")
    plt.show()

previewHalftone(volume_icc_halftone[-1], volume_icc[-1])

# 设置保存文件夹为volume_path同目录下的print子文件夹
volume_dir = os.path.dirname(args.volume_path)
SaveFolder = os.path.join(volume_dir, "print")
print("保存到:",SaveFolder)

# 确保目标目录存在
os.makedirs(SaveFolder, exist_ok=True)
# 删除其中所有文件
if os.path.exists(SaveFolder):
    for file in os.listdir(SaveFolder):
        file_path = os.path.join(SaveFolder, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

# 创建自定义颜色映射: 0=C, 1=M, 2=Y, 3=K, 4=W, 5=透明
# 使用CMYK对应的RGB值
colors = [
    [0, 1, 1],    # C - 青色 (RGB: 0, 255, 255)
    [1, 0, 1],    # M - 品红 (RGB: 255, 0, 255)
    [1, 1, 0],    # Y - 黄色 (RGB: 255, 255, 0)
    [0, 0, 0],    # K - 黑色 (RGB: 0, 0, 0)
    [1, 1, 1],    # W - 白色 (RGB: 255, 255, 255)
    [0.7, 0.7, 0.7],    # 边缘 - 灰色 (RGB: 255, 255, 255)
]

# 创建自定义颜色映射
cmap = ListedColormap(colors[:5])  # 不包括透明色，我们会单独处理透明度

for i in tqdm(range(volume_icc_halftone.shape[0]), desc="saving png slices..."):
    slice_data = volume_icc_halftone[i].copy()
    
    # 创建RGBA图像
    rgba_image = np.ones((slice_data.shape[0], slice_data.shape[1], 4))
    
    # 设置RGB颜色
    for val in range(5):  # 处理0-4的值
        mask = (slice_data == val)
        rgba_image[mask, 0] = colors[val][0]
        rgba_image[mask, 1] = colors[val][1]
        rgba_image[mask, 2] = colors[val][2]
    
    # 设置透明度 - 只有值为5的像素是透明的
    rgba_image[slice_data == 5, 3] = 0  # 设置透明度为0
    
    # 保存图像
    plt.imsave(f"{SaveFolder}/{i:04d}.png", rgba_image)
