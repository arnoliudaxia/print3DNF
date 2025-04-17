"""在模型的贴图上展示对应的vertices

Returns:
    _type_: _description_
"""
import argparse
import glob
import numpy as np
import trimesh
import pyvista as pv
import os
from PIL import Image
import matplotlib.pyplot as plt

def visualize_mesh_structure(obj_path, tex_path, show_wireframe=True, show_points=True, point_size=5, 
                             show_edges=True, edge_color='black', show_faces=True, 
                             opacity=0.5, show_normals=False, normal_scale=0.1):
    """
    可视化模型的网格结构
    
    参数:
    obj_path: OBJ文件路径
    show_wireframe: 是否显示线框
    show_points: 是否显示顶点
    point_size: 顶点大小
    show_edges: 是否显示边
    edge_color: 边的颜色
    show_faces: 是否显示面
    opacity: 面的透明度
    show_normals: 是否显示法线
    normal_scale: 法线长度缩放因子
    """
    print(f"加载模型: {obj_path}")
    
    # 使用trimesh加载模型
    mesh_trimesh = trimesh.load(obj_path)
    print(f"模型信息:")
    print(f"  顶点数量: {len(mesh_trimesh.vertices)}")
    print(f"  面数量: {len(mesh_trimesh.faces)}")
    print(f"  边数量: {len(mesh_trimesh.edges)}")
    print(f"  边界框: {mesh_trimesh.bounds}")
    
    # 使用PyVista加载模型
    mesh = pv.read(obj_path)
    
    # 创建渲染窗口
    p = pv.Plotter()
    
    # 添加面
    if show_faces:
        # 尝试获取贴图
        try:
            
            if os.path.exists(tex_path):
                print(f"使用贴图: {tex_path}")
                texture = pv.read_texture(tex_path)
                p.add_mesh(mesh, texture=texture, opacity=opacity, show_edges=show_wireframe)
            else:
                print(f"找不到贴图文件，使用随机颜色")
                p.add_mesh(mesh, opacity=opacity, show_edges=show_wireframe)
        except Exception as e:
            print(f"加载贴图失败: {str(e)}")
            p.add_mesh(mesh, opacity=opacity, show_edges=show_wireframe)
    
    # 添加边
    if show_edges and not show_wireframe:
        edges = mesh.extract_all_edges()
        p.add_mesh(edges, color=edge_color, line_width=1, render_lines_as_tubes=True)
    
    # 添加顶点
    if show_points:
        vertices = mesh.points
        p.add_points(vertices, color='red', point_size=point_size, render_points_as_spheres=True)
    
    # 添加法线
    if show_normals:
        # 计算面法线
        mesh.compute_normals(cell_normals=True, point_normals=True, inplace=True)
        
        # 显示面法线
        centers = mesh.cell_centers().points
        normals = mesh.cell_normals
        p.add_arrows(centers, normals, mag=normal_scale, color='blue')
        
        # 显示顶点法线
        p.add_arrows(mesh.points, mesh.point_normals, mag=normal_scale, color='green')
    
    # 添加坐标轴和网格
    p.show_grid()
    p.show_axes()
    
    # 显示渲染窗口
    p.show()

def visualize_mesh_components(obj_path):
    """
    分别可视化模型的顶点、边和面
    """
    # 使用trimesh加载模型
    mesh_trimesh = trimesh.load(obj_path)
    
    # 使用PyVista加载模型
    mesh = pv.read(obj_path)
    
    # 创建三个子图
    p = pv.Plotter(shape=(1, 3))
    
    # 1. 显示顶点
    p.subplot(0, 0)
    p.add_title("顶点")
    vertices = mesh.points
    p.add_points(vertices, color='red', point_size=15, render_points_as_spheres=True)
    p.show_grid()
    
    # 2. 显示边
    p.subplot(0, 1)
    p.add_title("边")
    edges = mesh.extract_all_edges()
    p.add_mesh(edges, color='black', line_width=5, render_lines_as_tubes=True)
    p.show_grid()
    
    # 3. 显示面
    p.subplot(0, 2)
    p.add_title("面")
    # 尝试获取贴图
    try:
        # 获取材质文件路径
        obj_dir = os.path.dirname(obj_path)
        tex_name = os.path.splitext(os.path.basename(obj_path))[0] + '_TEX.png'
        tex_path = os.path.join(obj_dir, tex_name)
        
        if os.path.exists(tex_path):
            texture = pv.read_texture(tex_path)
            p.add_mesh(mesh, texture=texture)
        else:
            p.add_mesh(mesh)
    except Exception as e:
        p.add_mesh(mesh)
    p.show_grid()
    
    # 显示渲染窗口
    p.show()

def visualize_mesh_uv_mapping(obj_path):
    """
    可视化模型的UV映射
    """
    # 使用trimesh加载模型
    mesh = trimesh.load(obj_path)
    
    # 检查是否有UV坐标
    if not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None:
        print("模型没有UV坐标")
        return
    
    # 获取UV坐标
    uv = mesh.visual.uv
    
    # 获取贴图
    texture = None
    if hasattr(mesh.visual, 'material') and hasattr(mesh.visual.material, 'image') and mesh.visual.material.image is not None:
        texture = np.array(mesh.visual.material.image)
    else:
        # 手动寻找和加载贴图
        obj_dir = os.path.dirname(obj_path)
        tex_name = os.path.splitext(os.path.basename(obj_path))[0] + '_TEX.png'
        tex_path = os.path.join(obj_dir, tex_name)
        if os.path.exists(tex_path):
            texture = np.array(Image.open(tex_path))
        else:
            print("找不到贴图文件")
            # 创建一个简单的棋盘格贴图
            texture = np.zeros((512, 512, 3), dtype=np.uint8)
            for i in range(8):
                for j in range(8):
                    if (i + j) % 2 == 0:
                        texture[i*64:(i+1)*64, j*64:(j+1)*64] = [200, 200, 200]
                    else:
                        texture[i*64:(i+1)*64, j*64:(j+1)*64] = [100, 100, 100]
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # 显示贴图
    ax1.imshow(texture)
    ax1.set_title("贴图")
    ax1.axis('off')
    
    # 显示UV坐标
    ax2.scatter(uv[:, 0], 1 - uv[:, 1], s=1, c='blue', alpha=0.5)  # 注意UV坐标y轴通常是翻转的
    
    # 获取面信息
    faces = mesh.faces
    
    # 绘制UV三角形
    for face in faces:
        # 获取面的UV坐标
        face_uv = uv[face]
        # 翻转y坐标
        face_uv_y_flipped = np.copy(face_uv)
        face_uv_y_flipped[:, 1] = 1 - face_uv_y_flipped[:, 1]
        # 绘制三角形
        ax2.plot([face_uv_y_flipped[0, 0], face_uv_y_flipped[1, 0]], 
                 [face_uv_y_flipped[0, 1], face_uv_y_flipped[1, 1]], 'k-', linewidth=0.1)
        ax2.plot([face_uv_y_flipped[1, 0], face_uv_y_flipped[2, 0]], 
                 [face_uv_y_flipped[1, 1], face_uv_y_flipped[2, 1]], 'k-', linewidth=0.1)
        ax2.plot([face_uv_y_flipped[2, 0], face_uv_y_flipped[0, 0]], 
                 [face_uv_y_flipped[2, 1], face_uv_y_flipped[0, 1]], 'k-', linewidth=0.1)
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_aspect('equal')
    ax2.set_title("UV映射")
    
    plt.tight_layout()
    plt.show()


def visualize_obj_vt_mapping_overlay(obj_path, tex_path):
    """
    直接从OBJ文件中读取vt信息，并将其叠加在纹理图上展示
    
    参数:
    obj_path: OBJ文件路径
    """
    print(f"从OBJ文件直接读取vt信息并叠加在纹理上: {obj_path}")
    
    # 读取OBJ文件
    with open(obj_path, 'r') as f:
        lines = f.readlines()
    
    # 解析顶点和纹理坐标
    vertices = []
    vts = []
    faces = []
    
    for line in lines:
        line = line.strip()
        if line.startswith('v '):
            # 解析顶点坐标
            parts = line.split()
            if len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif line.startswith('vt '):
            # 解析纹理坐标
            parts = line.split()
            if len(parts) >= 3:
                vts.append([float(parts[1]), float(parts[2])])
        elif line.startswith('f '):
            # 解析面信息
            parts = line.split()[1:]
            face_vertices = []
            face_vts = []
            
            for part in parts:
                indices = part.split('/')
                if len(indices) >= 2 and indices[0] and indices[1]:
                    # OBJ索引从1开始，所以要减1
                    v_idx = int(indices[0]) - 1
                    vt_idx = int(indices[1]) - 1
                    face_vertices.append(v_idx)
                    face_vts.append(vt_idx)
            
            if len(face_vertices) >= 3:
                faces.append((face_vertices, face_vts))
    
    print(f"解析结果:")
    print(f"  顶点数量: {len(vertices)}")
    print(f"  纹理坐标数量: {len(vts)}")
    print(f"  面数量: {len(faces)}")
    
    if len(vts) == 0:
        print("OBJ文件中没有纹理坐标(vt)信息")
        return
    
    # 将vts转换为numpy数组
    vts = np.array(vts)
    
    # 获取贴图
    texture = None
    
    if os.path.exists(tex_path):
        print(f"从文件加载贴图: {tex_path}")
        texture = np.array(Image.open(tex_path))
    else:
        print("找不到贴图文件，创建棋盘格贴图")
        # 创建一个简单的棋盘格贴图
        texture = np.zeros((512, 512, 3), dtype=np.uint8)
        for i in range(8):
            for j in range(8):
                if (i + j) % 2 == 0:
                    texture[i*64:(i+1)*64, j*64:(j+1)*64] = [200, 200, 200]
                else:
                    texture[i*64:(i+1)*64, j*64:(j+1)*64] = [100, 100, 100]
    
    # 创建图形 - 只有一个图来叠加显示
    plt.figure(figsize=(10, 10))
    
    # 显示贴图作为背景
    plt.imshow(texture)
    
    # 纹理高度和宽度
    tex_height, tex_width = texture.shape[:2]
    
    # 在贴图上显示所有vt点
    # 注意将坐标映射到图像像素位置
    scatter_x = vts[:, 0] * tex_width
    scatter_y = (1 - vts[:, 1]) * tex_height  # 纹理坐标y轴需要翻转
    plt.scatter(scatter_x, scatter_y, s=20, c='blue', alpha=0.7, label='Texture Coordinates (vt)')
    
    # 绘制UV三角形映射到贴图上
    for face_vertices, face_vts in faces:
        if len(face_vts) >= 3:  # 确保至少有3个点形成三角形
            # 获取面的UV坐标
            uv_coords = np.array([vts[idx] for idx in face_vts])
            
            # 转换为图像坐标
            uv_image_coords = np.copy(uv_coords)
            uv_image_coords[:, 0] *= tex_width  # x坐标映射到图像宽度
            uv_image_coords[:, 1] = (1 - uv_image_coords[:, 1]) * tex_height  # y坐标映射到图像高度并翻转
            
            # 绘制三角形边缘
            for i in range(len(uv_image_coords)):
                j = (i + 1) % len(uv_image_coords)
                plt.plot([uv_image_coords[i, 0], uv_image_coords[j, 0]], 
                         [uv_image_coords[i, 1], uv_image_coords[j, 1]], 
                         'r-', linewidth=0.7, alpha=0.8)
    
    plt.title("Texture Coordinate (vt) Mapping Overlay")
    plt.axis('off')  # 不显示坐标轴
    plt.tight_layout()
    plt.savefig('uv_mapping_overlay.png', dpi=300)
    print("已保存叠加UV映射图像到: uv_mapping_overlay.png")
    plt.show()



# 使用示例
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='展示模型的贴图对应的vertices，faces')
    parser.add_argument('--obj_path', type=str, required=True, help='OBJ文件路径')
    args = parser.parse_args()
    
    obj_file = args.obj_path
    
    png_files = glob.glob(os.path.join( os.path.dirname(obj_file), "*.png"))
    print(f"找到的PNG文件: {png_files}")
    if len(png_files) == 1:
        tex_path = png_files[0]
        print(f"找不到默认命名的贴图文件，使用目录中唯一的PNG文件: {tex_path}")
    

    # 可视化mesh
    visualize_mesh_structure(obj_file, tex_path, show_wireframe=True, show_points=True, 
                         show_edges=True, show_faces=True, opacity=0.9)

    # 分别可视化顶点、边和面
    visualize_mesh_components(obj_file)

        
    # 将vt坐标叠加到材质图上显示
    try:
        visualize_obj_vt_mapping_overlay(obj_file, tex_path)
    except Exception as e:
        print(f"将vt坐标叠加到材质图上失败: {str(e)}")# 计算纹理到3D坐标的映射
        