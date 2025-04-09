"""展示模型的贴图效果"""

import numpy as np
import trimesh
import pyvista as pv
from PIL import Image
import os
import glob
import argparse

def render_with_texture_mapping(obj_path):
    """使用PyVista的纹理映射功能直接渲染"""
    # 加载OBJ文件
    mesh = pv.read(obj_path)
    
    # 获取材质文件路径
    obj_dir = os.path.dirname(obj_path)
    tex_name = os.path.splitext(os.path.basename(obj_path))[0] + '_TEX.png'
    tex_path = os.path.join(obj_dir, tex_name)
    
    # 如果指定命名格式的贴图不存在，尝试查找目录下唯一的PNG文件
    if not os.path.exists(tex_path):
        png_files = glob.glob(os.path.join(obj_dir, "*.png"))
        if len(png_files) == 1:
            tex_path = png_files[0]
            print(f"找不到默认命名的贴图文件，使用目录中唯一的PNG文件: {tex_path}")
        elif len(png_files) > 1:
            print(f"找不到默认命名的贴图文件，且目录中有多个PNG文件，无法确定使用哪一个")
            # 尝试查找与OBJ文件同名的PNG
            base_name = os.path.splitext(os.path.basename(obj_path))[0]
            for png_file in png_files:
                if base_name.lower() in os.path.basename(png_file).lower():
                    tex_path = png_file
                    print(f"使用包含模型名称的PNG文件: {tex_path}")
                    break
            else:
                # 如果没有找到匹配的，使用第一个PNG
                tex_path = png_files[0]
                print(f"使用目录中的第一个PNG文件: {tex_path}")
        else:
            raise FileNotFoundError(f"找不到贴图文件： 既没有 {tex_name}，目录中也没有其他PNG文件")
    else:
        print(f"从文件加载贴图: {tex_path}")
    
    # 加载纹理
    texture = pv.read_texture(tex_path)
    
    # 创建渲染窗口
    p = pv.Plotter()
    p.add_mesh(mesh, texture=texture, smooth_shading=True)
    p.show_grid()
    p.show()
    
    return mesh


parser = argparse.ArgumentParser(description='渲染带贴图的模型')
parser.add_argument('--obj_path', type=str, required=True, help='OBJ文件路径')

args = parser.parse_args()

obj_file = args.obj_path

# 使用PyVista的纹理映射功能直接渲染
print("\n使用PyVista的纹理映射功能直接渲染")
try:
    pv_mesh = render_with_texture_mapping(obj_file)
except Exception as e:
    print(f"渲染失败： {e}")
