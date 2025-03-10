from PIL import Image
from pathlib import Path
from typing import Union, Optional, List, Tuple
import logging
import numpy as np
from PIL import ImageDraw
import random

def resize_byVoxelSize(
    image_path: Union[str, Path], 
    output_path: Union[str, Path],
    scale_factor: float = 3.0
) -> Optional[Image.Image]:
    """根据打印机体素大小调整图像尺寸。
    
    由于打印机的体素不是正立方体，需要按照打印机voxel的物理尺寸调整图像。
    默认情况下，打印机在xy方向的比例是x = y * 3，图片发送后y方向会拉伸3倍。
    
    Args:
        image_path (Union[str, Path]): 输入图像的路径
        output_path (Union[str, Path]): 输出图像的保存路径
        scale_factor (float, optional): 缩放因子，默认为3.0
        
    Returns:
        Optional[Image.Image]: 返回处理后的PIL Image对象，如果处理失败则返回None
        
    Raises:
        FileNotFoundError: 当输入图像文件不存在时
        IOError: 当图像处理或保存过程中发生错误时
    """
    try:
        # 确保输入路径存在
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"输入图像不存在: {image_path}")
            
        # 打开并处理图像
        with Image.open(image_path) as img:
            # 获取原始尺寸
            original_width, original_height = img.size
            # 计算新的高度
            new_height = int(original_height / scale_factor)
            
            # 使用LANCZOS重采样方法进行缩放
            resized_img = img.resize(
                (original_width, new_height), 
                Image.Resampling.LANCZOS
            )
            
            # 确保输出目录存在
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存结果
            resized_img.save(output_path, quality=95, optimize=True)
            
            return resized_img
            
    except Exception as e:
        logging.error(f"处理图像时发生错误: {str(e)}")
        return None
    
class Halftone:
    """半色调图像处理类
    
    提供两种不同的半色调处理方法：
    1. 简单CMYK转换方法
    2. 高级半色调处理方法（支持点大小和角度控制）
    """
    
    def __init__(self):
        self.angles = {
            'C': 15,  # Cyan 青色角度
            'M': 75,  # Magenta 品红角度
            'Y': 45,   # Yellow 黄色角度
            'K': 0   # Key(Black) 黑色角度
        }
    
    def simple_halftone(self, image: Image.Image, output_path: Optional[str] = None) -> Image.Image:
        """简单的半色调处理方法
        
        Args:
            image (Image.Image): 输入图像
            output_path (Optional[str]): 输出文件路径，如果不指定则不保存
            
        Returns:
            Image.Image: 处理后的CMYK半色调图像
        """
        # 转换为CMYK
        cmyk = image.convert('CMYK')
        cmyk_channels = cmyk.split()
        
        # 对每个通道进行二值化处理
        processed_channels = [
            channel.convert('1').convert('L')
            for channel in cmyk_channels
        ]
        
        # 合并通道
        new_cmyk = Image.merge('CMYK', processed_channels)
        
        if output_path:
            new_cmyk.save(output_path)
            
        return new_cmyk
    
    @staticmethod
    def gcr(image: Image.Image, percentage: int = 100) -> Image.Image:
        """灰色分量替换（Gray Component Replacement）
        
        Args:
            image (Image.Image): 输入图像
            percentage (int): GCR处理的百分比，默认100%
            
        Returns:
            Image.Image: 处理后的CMYK图像
        """
        cmyk_im = image.convert('CMYK')
        cmyk_channels = cmyk_im.split()
        
        # 转换为numpy数组处理
        channels = [np.array(channel) for channel in cmyk_channels]
        cmyk_array = np.stack(channels, axis=2)
        
        # 计算灰色分量
        gray = np.min(cmyk_array[:,:,:3], axis=2) * (percentage / 100)
        
        # 更新CMYK值
        for i in range(3):  # CMY通道
            cmyk_array[:,:,i] = np.maximum(0, cmyk_array[:,:,i] - gray)
        cmyk_array[:,:,3] = gray  # K通道
        
        # 确保值在有效范围内
        cmyk_array = np.clip(cmyk_array, 0, 255).astype(np.uint8)
        
        # 转回PIL图像
        new_channels = [Image.fromarray(cmyk_array[:,:,i]) for i in range(4)]
        return Image.merge('CMYK', new_channels)
    
    def advanced_halftone(
        self,
        image: Image.Image,
        sample: int = 10,
        scale: float = 2.0,
        output_path: Optional[str] = None
    ) -> Union[Image.Image, List[Image.Image]]:
        """高级半色调处理方法
        
        Args:
            image (Image.Image): 输入图像
            sample (int): 采样大小（像素）
            scale (float): 缩放因子
            output_path (Optional[str]): 输出文件路径
            
        Returns:
            Union[Image.Image, List[Image.Image]]: 如果指定output_path则返回合并后的图像，
                                                否则返回各个通道的半色调图像列表
        """
        # 首先进行GCR处理
        cmyk = self.gcr(image)
        dots = []
        
        for i, channel in enumerate(cmyk.split()):
            angle = list(self.angles.values())[i]
            # 旋转通道
            channel = channel.rotate(angle, expand=1)
            channel_array = np.array(channel)
            
            # 计算输出尺寸
            h, w = channel_array.shape
            size = (int(w * scale), int(h * scale))
            
            # 创建采样网格
            x_coords = np.arange(0, w, sample)
            y_coords = np.arange(0, h, sample)
            X, Y = np.meshgrid(x_coords, y_coords)
            
            # 计算采样区域平均值
            means = np.zeros((len(y_coords), len(x_coords)))
            for i, y in enumerate(y_coords):
                for j, x in enumerate(x_coords):
                    y_end = min(y + sample, h)
                    x_end = min(x + sample, w)
                    means[i, j] = np.mean(channel_array[y:y_end, x:x_end])
            
            # 创建半色调图像
            half_tone = Image.new('L', size)
            draw = ImageDraw.Draw(half_tone)
            
            # 计算和绘制点
            diameters = np.sqrt(means / 255.0)
            edges = 0.5 * (1 - diameters)
            
            for i in range(len(y_coords)):
                for j in range(len(x_coords)):
                    if diameters[i, j] > 0:
                        x_pos = (X[i, j] + edges[i, j]) * scale
                        y_pos = (Y[i, j] + edges[i, j]) * scale
                        box_edge = sample * diameters[i, j] * scale
                        draw.ellipse(
                            (x_pos, y_pos, x_pos + box_edge, y_pos + box_edge),
                            fill=255
                        )
            
            # 旋转回原始角度并裁剪
            half_tone = half_tone.rotate(-angle, expand=1)
            width_half, height_half = half_tone.size
            xx = (width_half - image.size[0] * scale) // 2
            yy = (height_half - image.size[1] * scale) // 2
            half_tone = half_tone.crop((
                xx, yy,
                xx + image.size[0] * scale,
                yy + image.size[1] * scale
            ))
            
            dots.append(half_tone)
        
        if output_path:
            result = Image.merge('CMYK', dots)
            result.save(output_path)
            return result
            
        return dots
    
class ColorSeparator:
    """CMYK颜色分离处理类
    
    提供将CMYK图像进行颜色分离的功能，支持多种分离模式：
    1. 2x2模式：将每个像素分离为2x2的网格
    2. 1x4模式：将每个像素水平分离为4个点
    3. 4x1模式：将每个像素垂直分离为4个点
    """
    
    @staticmethod
    def separate_colors(pixel: Tuple[int, int, int, int]) -> List[List[int]]:
        """将一个CMYK像素分解为基础色
        
        Args:
            pixel (Tuple[int, int, int, int]): CMYK像素值
            
        Returns:
            List[List[int]]: 分离后的4个基础色值列表
        """
        c, m, y, k = pixel
        base_colors = []
        
        # 添加基础色
        if c == 255: base_colors.append([255,0,0,0])  # C
        if m == 255: base_colors.append([0,255,0,0])  # M
        if y == 255: base_colors.append([0,0,255,0])  # Y
        if k == 255: base_colors.append([0,0,0,255])  # K
        
        # 如果没有任何颜色，返回白色
        if not base_colors:
            return [[0,0,0,0]] * 4
        
        # 如果基础色少于4个，重复填充
        while len(base_colors) < 4:
            base_colors.append(random.choice(base_colors))
        
        # 随机打乱顺序
        random.shuffle(base_colors)
        return base_colors
    
    @classmethod
    def upsample_cmyk_image(
        cls,
        image: Union[str, Path, Image.Image],
        output_path: Optional[Union[str, Path]] = None,
        mode: str = '2x2'
    ) -> Image.Image:
        """将CMYK图像进行颜色分离处理
        
        Args:
            image (Union[str, Path, Image.Image]): 输入图像或图像路径
            output_path (Optional[Union[str, Path]]): 输出图像路径
            mode (str): 分离模式，可选 '2x2', '1x4', '4x1'
            
        Returns:
            Image.Image: 处理后的图像
            
        Raises:
            ValueError: 当mode参数不合法时
            FileNotFoundError: 当输入图像文件不存在时
        """
        # 处理输入图像
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"输入图像不存在: {image_path}")
            img = Image.open(image_path)
        else:
            img = image
            
        # 确保图像是CMYK模式
        if img.mode != 'CMYK':
            img = img.convert('CMYK')
            
        width, height = img.size
        img_array = np.array(img)
        
        if mode == '2x2':
            # 创建新图像，尺寸是原来的2倍宽和2倍高
            new_image = Image.new('CMYK', (width*2, height*2))
            
            for y in range(height):
                for x in range(width):
                    # 获取原始像素的CMYK值
                    pixel = tuple(img_array[y,x])
                    # 分解为基础色
                    separated_colors = cls.separate_colors(pixel)
                    
                    # 在2x2区域填充分离后的颜色
                    for i, color in enumerate(separated_colors):
                        new_x = (x * 2) + (i % 2)
                        new_y = (y * 2) + (i // 2)
                        new_image.putpixel((new_x, new_y), tuple(color))
        
        elif mode == '1x4':
            # 创建新图像，宽度是原来的4倍，高度不变
            new_image = Image.new('CMYK', (width*4, height))
            
            for y in range(height):
                for x in range(width):
                    # 获取原始像素的CMYK值
                    pixel = tuple(img_array[y,x])
                    # 分解为基础色
                    separated_colors = cls.separate_colors(pixel)
                    
                    # 在1x4区域填充分离后的颜色
                    for i, color in enumerate(separated_colors):
                        new_x = (x * 4) + i
                        new_image.putpixel((new_x, y), tuple(color))
        
        elif mode == '4x1':
            # 创建新图像，高度是原来的4倍，宽度不变
            new_image = Image.new('CMYK', (width, height*4))
            
            for y in range(height):
                for x in range(width):
                    # 获取原始像素的CMYK值
                    pixel = tuple(img_array[y,x])
                    # 分解为基础色
                    separated_colors = cls.separate_colors(pixel)
                    
                    # 在4x1区域填充分离后的颜色
                    for i, color in enumerate(separated_colors):
                        new_y = (y * 4) + i
                        new_image.putpixel((x, new_y), tuple(color))
        
        else:
            raise ValueError("模式必须是 '2x2', '1x4' 或 '4x1'")
        
        # 保存结果（如果指定了输出路径）
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            new_image.save(output_path)
        
        return new_image
    
    
    
    
