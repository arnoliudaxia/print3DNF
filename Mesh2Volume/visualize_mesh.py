"""在模型的贴图上展示对应的vertices

Returns:
    _type_: _description_
"""
import numpy as np
import trimesh
import pyvista as pv
import os
from PIL import Image
import matplotlib.pyplot as plt

def visualize_mesh_structure(obj_path, show_wireframe=True, show_points=True, point_size=5, 
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
            # 获取材质文件路径
            obj_dir = os.path.dirname(obj_path)
            tex_name = os.path.splitext(os.path.basename(obj_path))[0] + '_TEX.png'
            tex_path = os.path.join(obj_dir, tex_name)
            
            if os.path.exists(tex_path):
                print(f"使用贴图: {tex_path}")
                texture = pv.read_texture(tex_path)
                p.add_mesh(mesh, texture=texture, opacity=opacity, show_edges=show_wireframe)
            else:
                # 如果没有贴图，使用随机颜色
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

def visualize_obj_vt_mapping(obj_path):
    """
    直接从OBJ文件中读取vt信息，并在纹理图上展示出来
    
    参数:
    obj_path: OBJ文件路径
    """
    print(f"从OBJ文件直接读取vt信息: {obj_path}")
    
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
    obj_dir = os.path.dirname(obj_path)
    tex_name = os.path.splitext(os.path.basename(obj_path))[0] + '_TEX.png'
    tex_path = os.path.join(obj_dir, tex_name)
    
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
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # 显示贴图
    ax1.imshow(texture)
    ax1.set_title("贴图")
    ax1.axis('off')
    
    # 显示所有vt点
    ax2.scatter(vts[:, 0], 1 - vts[:, 1], s=5, c='blue', alpha=0.5, label='纹理坐标(vt)')
    
    # 绘制UV三角形
    for face_vertices, face_vts in faces:
        if len(face_vts) >= 3:  # 确保至少有3个点形成三角形
            # 获取面的UV坐标
            uv_coords = np.array([vts[idx] for idx in face_vts])
            # 翻转y坐标
            uv_coords_y_flipped = np.copy(uv_coords)
            uv_coords_y_flipped[:, 1] = 1 - uv_coords_y_flipped[:, 1]
            
            # 绘制三角形边缘
            for i in range(len(uv_coords_y_flipped)):
                j = (i + 1) % len(uv_coords_y_flipped)
                ax2.plot([uv_coords_y_flipped[i, 0], uv_coords_y_flipped[j, 0]], 
                         [uv_coords_y_flipped[i, 1], uv_coords_y_flipped[j, 1]], 
                         'k-', linewidth=0.2, alpha=0.7)
    
    # 设置坐标轴范围和标题
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_aspect('equal')
    ax2.set_title("OBJ文件中的纹理坐标(vt)映射")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('uv_mapping_from_obj.png', dpi=300)
    print("已保存UV映射图像到: uv_mapping_from_obj.png")
    plt.show()

def visualize_obj_vt_mapping_overlay(obj_path):
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
    obj_dir = os.path.dirname(obj_path)
    tex_name = os.path.splitext(os.path.basename(obj_path))[0] + '_TEX.png'
    tex_path = os.path.join(obj_dir, tex_name)
    
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


def texture_to_3d_coordinates(obj_path, output_path=None):
    """
    将纹理图上的每个像素映射到3D模型坐标
    
    参数:
    obj_path: OBJ文件路径
    output_path: 输出结果的路径，如果为None则不保存
    
    返回：
    texture_3d_map: 形状为(height, width, 3)的numpy数组，存储每个纹理像素对应的3D坐标
    """
    print(f"从OBJ文件读取信息并计算纹理到3D映射: {obj_path}")
    
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
    print(f"  面数量： {len(faces)}")
    
    if len(vts) == 0:
        print("OBJ文件中没有纹理坐标(vt)信息")
        return None
    
    # 将数据转换为numpy数组
    vertices = np.array(vertices)
    vts = np.array(vts)
    
    # 获取贴图
    texture = None
    obj_dir = os.path.dirname(obj_path)
    tex_name = os.path.splitext(os.path.basename(obj_path))[0] + '_TEX.png'
    tex_path = os.path.join(obj_dir, tex_name)
    
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
    
    # 纹理高度和宽度
    tex_height, tex_width = texture.shape[:2]
    
    # 创建一个与纹理图同样大小的3D坐标映射数组
    texture_3d_map = np.zeros((tex_height, tex_width, 3), dtype=np.float32)
    # 创建一个掩码，标记哪些像素已经被映射
    mapped_mask = np.zeros((tex_height, tex_width), dtype=bool)
    
    # 定义一个函数计算点在三角形内的重心坐标
    def barycentric_coordinates(p, a, b, c):
        """
        计算点p在三角形abc中的重心坐标
        
        参数:
        p, a, b, c: 2D点坐标
        
        返回:
        (u, v, w): 重心坐标，如果点在三角形外，则某些坐标可能为负
        """
        v0 = b - a
        v1 = c - a
        v2 = p - a
        
        # 计算点积
        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        d20 = np.dot(v2, v0)
        d21 = np.dot(v2, v1)
        
        # 计算重心坐标
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-10:
            return (-1, -1, -1)  # 退化三角形
        
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        
        return (u, v, w)
    
    # 定义一个函数判断点是否在三角形内
    def point_in_triangle(p, a, b, c):
        """
        判断点p是否在三角形abc内部
        
        参数:
        p, a, b, c: 2D点坐标
        
        返回:
        bool: 如果点在三角形内（包括边界）则为True
        """
        u, v, w = barycentric_coordinates(p, a, b, c)
        # 允许一点点数值误差
        epsilon = 1e-5
        return (u >= -epsilon) and (v >= -epsilon) and (w >= -epsilon) and (abs(u + v + w - 1.0) < epsilon)
    
    print("开始计算纹理到3D映射...")
    
    # 对每个三角形面进行处理
    for face_idx, (face_vertices, face_vts) in enumerate(faces):
        if len(face_vertices) < 3 or len(face_vts) < 3:
            continue
        
        # 获取面的顶点和纹理坐标
        v_indices = face_vertices[:3]  # 只处理前三个点（如果是四边形，拆分为三角形）
        vt_indices = face_vts[:3]
        
        # 获取三角形的3D顶点
        tri_vertices = vertices[v_indices]
        
        # 获取三角形的纹理坐标并转换为图像坐标
        tri_vts = vts[vt_indices]
        tri_vts_img = np.copy(tri_vts)
        tri_vts_img[:, 0] *= tex_width
        tri_vts_img[:, 1] = (1 - tri_vts_img[:, 1]) * tex_height  # 翻转y坐标
        
        # 计算三角形的包围盒
        min_x = max(0, int(np.floor(np.min(tri_vts_img[:, 0]))))
        max_x = min(tex_width - 1, int(np.ceil(np.max(tri_vts_img[:, 0]))))
        min_y = max(0, int(np.floor(np.min(tri_vts_img[:, 1]))))
        max_y = min(tex_height - 1, int(np.ceil(np.max(tri_vts_img[:, 1]))))
        
        # 对包围盒内的每个像素进行处理
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                # 如果这个像素已经被映射，跳过
                if mapped_mask[y, x]:
                    continue
                
                # 检查像素是否在三角形内
                pixel_pos = np.array([x, y])
                if point_in_triangle(pixel_pos, tri_vts_img[0], tri_vts_img[1], tri_vts_img[2]):
                    # 计算重心坐标
                    u, v, w = barycentric_coordinates(pixel_pos, tri_vts_img[0], tri_vts_img[1], tri_vts_img[2])
                    
                    # 使用重心坐标插值计算3D坐标
                    pos_3d = u * tri_vertices[0] + v * tri_vertices[1] + w * tri_vertices[2]
                    
                    # 存储3D坐标
                    texture_3d_map[y, x] = pos_3d
                    mapped_mask[y, x] = True
        
        # 每处理100个面打印一次进度
        if (face_idx + 1) % 100 == 0 or face_idx == len(faces) - 1:
            print(f"处理进度: {face_idx + 1}/{len(faces)} 面 ({(face_idx + 1) / len(faces) * 100:.1f}%)")
    
    # 计算映射覆盖率
    coverage = np.sum(mapped_mask) / (tex_height * tex_width) * 100
    print(f"纹理映射覆盖率: {coverage:.2f}%")
    
    # 可视化结果
    # 创建一个RGB图像，显示3D坐标的映射
    # 将3D坐标归一化到[0,1]范围用于可视化
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    range_coords = max_coords - min_coords
    
    # 避免除以零
    range_coords[range_coords < 1e-10] = 1.0
    
    # 创建归一化的3D坐标可视化
    normalized_map = np.zeros_like(texture_3d_map)
    normalized_map[mapped_mask] = (texture_3d_map[mapped_mask] - min_coords) / range_coords
    
    # 创建可视化图像
    plt.figure(figsize=(15, 10))
    
    # 显示原始纹理
    plt.subplot(2, 2, 1)
    plt.imshow(texture)
    plt.title("原始纹理")
    plt.axis('off')
    
    # 显示映射掩码
    plt.subplot(2, 2, 2)
    plt.imshow(mapped_mask, cmap='gray')
    plt.title(f"映射覆盖 ({coverage:.2f}%)")
    plt.axis('off')
    
    # 显示X坐标映射
    plt.subplot(2, 2, 3)
    plt.imshow(normalized_map[:, :, 0], cmap='jet')
    plt.title("X坐标映射")
    plt.axis('off')
    plt.colorbar()
    
    # 显示Y坐标映射
    plt.subplot(2, 2, 4)
    plt.imshow(normalized_map[:, :, 1], cmap='jet')
    plt.title("Y坐标映射")
    plt.axis('off')
    plt.colorbar()
    
    plt.tight_layout()
    
    # 保存结果
    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"已保存可视化结果到: {output_path}")
        
        # 保存3D坐标映射为numpy数组
        np_output_path = os.path.splitext(output_path)[0] + "_3d_map.npy"
        np.save(np_output_path, texture_3d_map)
        print(f"已保存3D坐标映射数据到: {np_output_path}")
        
        # 保存掩码
        mask_output_path = os.path.splitext(output_path)[0] + "_mask.npy"
        np.save(mask_output_path, mapped_mask)
        print(f"已保存映射掩码到: {mask_output_path}")
    
    plt.show()
    
    return texture_3d_map, mapped_mask


# 使用示例
if __name__ == "__main__":
    obj_file = "/home/arno/Projects/Pint3D/print_ngp/Mesh2Volume/ExampleMesh/Minecraft_Grass_Block_OBJ/Grass_Block.obj"

    # 可视化网格结构
    # visualize_mesh_structure(obj_file, show_wireframe=True, show_points=True, 
    #                      show_edges=True, show_faces=True, opacity=0.9)

    # 分别可视化顶点、边和面
    # visualize_mesh_components(obj_file)

        
    # 将vt坐标叠加到材质图上显示
    try:
        visualize_obj_vt_mapping_overlay(obj_file)
    except Exception as e:
        print(f"将vt坐标叠加到材质图上失败: {str(e)}")# 计算纹理到3D坐标的映射
    texture_3d_map, mapped_mask = texture_to_3d_coordinates(obj_file, output_path="texture_to_3d_mapping.png")