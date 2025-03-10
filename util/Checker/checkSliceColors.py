import os
import glob
from PIL import Image
from collections import defaultdict
import time

def get_unique_colors(image_path):
    """获取单个PNG图像中的所有唯一颜色值"""
    try:
        with Image.open(image_path) as img:
            # 转换为RGBA模式以统一处理
            img = img.convert('RGBA')
            # 获取像素数据
            pixels = list(img.getdata())
            # 使用集合去重
            unique_colors = set(pixels)
            return unique_colors
    except Exception as e:
        print(f"处理图像 {image_path} 时出错: {e}")
        return set()

def analyze_png_colors(directory_path):
    """分析目录中所有PNG文件的唯一颜色值"""
    # 确保路径末尾有斜杠
    if not directory_path.endswith(('/', '\\')):
        directory_path += os.path.sep
    
    # 获取所有PNG文件路径
    png_files = glob.glob(directory_path + "**/*.png", recursive=True)
    
    if not png_files:
        print(f"在路径 {directory_path} 中未找到PNG文件")
        return
    
    print(f"找到 {len(png_files)} 个PNG文件")
    
    # 存储每个文件的唯一颜色
    file_colors = {}
    # 存储所有文件中出现的颜色及其出现的文件
    all_colors = defaultdict(list)
    
    start_time = time.time()
    
    # 处理每个PNG文件
    for i, png_file in enumerate(png_files):
        if i > 0 and i % 10 == 0:
            elapsed = time.time() - start_time
            print(f"已处理 {i}/{len(png_files)} 个文件 (用时: {elapsed:.2f}秒)")
        
        file_name = os.path.basename(png_file)
        unique_colors = get_unique_colors(png_file)
        
        file_colors[file_name] = unique_colors
        
        # 更新全局颜色统计
        for color in unique_colors:
            all_colors[color].append(file_name)
    
    # 输出统计结果
    print("\n===== 分析结果 =====")
    print(f"总共分析了 {len(png_files)} 个PNG文件")
    print(f"发现 {len(all_colors)} 种唯一颜色")
    

    print("\n颜色值及其出现的文件:")
    for color, files in all_colors.items():
        print(f"颜色 {color}: 出现在 {len(files)} 个文件中")
        if len(files) < 5:  # 如果文件数量少，显示所有文件名
            print(f"  文件: {', '.join(files)}")
        else:  # 否则只显示部分
            print(f"  文件示例: {', '.join(files[:5])}...")
    
    # 输出每个文件包含的颜色数量
    # print("\n每个文件的唯一颜色数量:")
    # for file_name, colors in file_colors.items():
    #     print(f"{file_name}: {len(colors)} 种颜色")

if __name__ == "__main__":
    # 获取用户输入的目录路径
    directory_path = "print/ProNew/white0.014mm"
    analyze_png_colors(directory_path)
