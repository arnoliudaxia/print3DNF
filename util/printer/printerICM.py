from PIL import Image, ImageCms
import io
import os
import argparse
from typing import Union, List, Optional

def convert_color_profile(
    input_image: Union[str, Image.Image], 
    output_prefix: Optional[str] = None, 
    modes: Optional[List[int]] = None, 
    srgb_profile_path: str = "AdobeRGB1998.icc", 
    cmyk_profile_path: str = "Stratasys_J8_7xx_VeroUltraWhite_HT3_VividCMYK.icm",
    output_mode: str = 'CMYK',
    save_files: bool = True
) -> Union[List[str], List[Image.Image]]:
    """
    将输入图像从sRGB颜色配置文件转换为CMYK颜色配置文件。
    
    参数:
        input_image (str or PIL.Image.Image): 输入图像的路径或PIL图像对象
        output_prefix (str, optional): 输出文件名前缀，默认使用输入文件名
        modes (list, optional): 渲染意图模式列表，默认为[0,1,2,3]
        srgb_profile_path (str): sRGB配置文件路径
        cmyk_profile_path (str): CMYK配置文件路径
        output_mode (str): 输出模式，'CMYK'或'RGB'
        save_files (bool): 是否保存文件到磁盘，默认为True
    
    返回:
        list: 如果save_files为True，返回生成的输出文件路径列表；
              否则，返回转换后的PIL图像对象列表
    """
    if modes is None:
        modes = [0, 1, 2, 3]
    
    # 确定是否需要关闭图像
    need_close = False
    
    # 处理输入图像
    if isinstance(input_image, str):
        img = Image.open(input_image)
        need_close = True
        
        if output_prefix is None:
            # 从输入文件名中提取基本名称（不含扩展名）
            output_prefix = os.path.splitext(os.path.basename(input_image))[0]
    else:
        # 输入是PIL图像对象
        img = input_image
        
        if output_prefix is None:
            # 使用默认前缀
            output_prefix = "converted_image"
    
    # 准备结果列表
    output_files = []
    output_images = []
    
    try:
        input_profile = ImageCms.ImageCmsProfile(srgb_profile_path)
        output_profile = ImageCms.ImageCmsProfile(cmyk_profile_path)
        
        for mode in modes:
            # 执行颜色配置文件转换
            converted_image = ImageCms.profileToProfile(
                img,
                inputProfile=input_profile,
                outputProfile=output_profile,
                renderingIntent=mode,
                outputMode=output_mode
            )
            
            # 保存转换后的图像
            if save_files:
                output_image_path = f"{output_prefix}_{mode}.tif"
                converted_image.save(output_image_path)
                print(f"Mode {mode} conversion completed: {output_image_path}")
                output_files.append(output_image_path)
            
            # 保存图像对象
            output_images.append(converted_image)
    
    finally:
        # 如果我们打开了图像文件，确保关闭它
        if need_close:
            img.close()
    
    # 根据save_files参数返回相应的结果
    return output_files if save_files else output_images

def main():
    parser = argparse.ArgumentParser(description='Convert image color profiles')
    parser.add_argument('input_image', help='Path to the input image')
    parser.add_argument('--output-prefix', help='Prefix for output filenames')
    parser.add_argument('--modes', type=int, nargs='+', default=[0, 1, 2, 3],
                        help='Rendering intent modes (0-3)')
    parser.add_argument('--srgb-profile', default='AdobeRGB1998.icc',
                        help='Path to sRGB color profile')
    parser.add_argument('--cmyk-profile', 
                        default='Stratasys_J8_7xx_VeroUltraWhite_HT3_VividCMYK.icm',
                        help='Path to CMYK color profile')
    parser.add_argument('--output-mode', choices=['CMYK', 'RGB'], default='CMYK',
                        help='Output color mode')
    
    args = parser.parse_args()
    
    convert_color_profile(
        args.input_image,
        args.output_prefix,
        args.modes,
        args.srgb_profile,
        args.cmyk_profile,
        args.output_mode
    )

if __name__ == "__main__":
    main()

"""
作为命令行工具使用
# 基本用法
python printerICM.py color_palette.png

# 指定输出前缀
python printerICM.py color_palette.png --output-prefix my_converted

# 只使用特定的渲染意图模式
python printerICM.py color_palette.png --modes 1 3

# 指定不同的配置文件
python printerICM.py color_palette.png --srgb-profile custom_rgb.icc --cmyk-profile custom_cmyk.icm

# 指定输出为RGB模式
python printerICM.py color_palette.png --output-mode RGB


作为模块导入使用
from printerICM import convert_color_profile

# 基本用法
convert_color_profile("color_palette.png")

# 自定义参数
output_files = convert_color_profile(
    "color_palette.png",
    output_prefix="custom_output",
    modes=[1, 3],
    srgb_profile_path="custom_rgb.icc",
    cmyk_profile_path="custom_cmyk.icm",
    output_mode="RGB"
)

print(f"Generated files: {output_files}")

"""
