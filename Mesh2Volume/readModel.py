"""展示模型的贴图效果"""

import numpy as np
import trimesh
import pyvista as pv
from PIL import Image
import os

def render_with_texture_mapping(obj_path):
    """使用PyVista的纹理映射功能直接渲染"""
    # 加载OBJ文件
    mesh = pv.read(obj_path)
    
    # 获取材质文件路径
    obj_dir = os.path.dirname(obj_path)
    tex_name = os.path.splitext(os.path.basename(obj_path))[0] + '_TEX.png'
    tex_path = os.path.join(obj_dir, tex_name)
    
    if os.path.exists(tex_path):
        print(f"从文件加载贴图: {tex_path}")
        # 加载纹理
        texture = pv.read_texture(tex_path)
        
        # 创建渲染窗口
        p = pv.Plotter()
        p.add_mesh(mesh, texture=texture, smooth_shading=True)
        p.show_grid()
        p.show()
        
        return mesh
    else:
        raise FileNotFoundError(f"找不到贴图文件: {tex_path}")

# 使用示例
obj_file = "/home/arno/Projects/Pint3D/print_ngp/Mesh2Volume/ExampleMesh/Minecraft_Grass_Block_OBJ/Grass_Block.obj"

# 方法2: 使用PyVista的纹理映射功能直接渲染
print("\n方法2: 使用PyVista的纹理映射功能直接渲染")
pv_mesh2 = render_with_texture_mapping(obj_file)
