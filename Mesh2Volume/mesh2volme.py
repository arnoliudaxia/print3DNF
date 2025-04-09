import glob
import sys
import numpy as np
import trimesh
import polyscope as ps
import os
from PIL import Image
import torch
import time
from scipy.spatial import KDTree
from joblib import Memory
mem = Memory(".cache")


script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(os.path.join(parent_dir, "util"))

from printer.printerHelper import getVoxelSize

def load_obj_and_texture(obj_path):
    """
    加载OBJ模型和对应的贴图
    
    参数:
    obj_path: OBJ文件路径
    
    返回:
    mesh: trimesh网格对象
    texture: 贴图图像（如果存在）
    texture_path: 贴图文件路径
    """
    print(f"加载OBJ模型: {obj_path}")
    
    # 使用trimesh加载OBJ文件
    mesh = trimesh.load(obj_path)
    
    # 获取贴图
    texture = None
    texture_path = None
    obj_dir = os.path.dirname(obj_path)
    tex_name = os.path.splitext(os.path.basename(obj_path))[0] + '_TEX.png'
    tex_path = os.path.join(obj_dir, tex_name)
    
    if os.path.exists(tex_path):
        print(f"从文件加载贴图: {tex_path}")
        texture = np.array(Image.open(tex_path))
        texture_path = tex_path
    else:
        # 如果指定命名格式的贴图不存在，尝试查找目录下唯一的PNG文件
        png_files = glob.glob(os.path.join(obj_dir, "*.png"))
        if len(png_files) == 1:
            tex_path = png_files[0]
            print(f"找不到默认命名的贴图文件，使用目录中唯一的PNG文件: {tex_path}")
        elif len(png_files) > 1:
            print(f"找不到默认命名的贴图文件，且目录中有多个PNG文件，无法确定使用哪一个")
            

        if os.path.exists(tex_path):
            print(f"从文件加载贴图: {tex_path}")
            texture = np.array(Image.open(tex_path))
            texture_path = tex_path
        
        if texture is None:
            print("找不到贴图文件")
    
    return mesh, texture, texture_path

def create_voxel_grid(mesh, voxel_size):
    """
    创建一个体素网格，基于模型的边界框
    
    参数:
    mesh: trimesh网格对象
    voxel_size: 体素尺寸 [x, y, z]
    
    返回:
    bound_low: 网格最小边界点
    bound_high: 网格最大边界点
    grid_dims: 网格维度 (nx, ny, nz)
    voxel_size: 体素尺寸 [x, y, z]
    """
    # 计算模型的边界框
    bounds = mesh.bounds
    min_bound = bounds[0]
    max_bound = bounds[1]
    
    # 计算体素尺寸
    voxel_size = np.array(voxel_size)
    
    # 确保边界框完全覆盖模型
    min_bound = min_bound - voxel_size*2
    max_bound = max_bound + voxel_size*2
    
    # 计算每个维度的体素数量
    grid_dims = np.ceil((max_bound - min_bound) / voxel_size).astype(int)
    
    print(f"体素网格信息:")
    print(f"  体素尺寸: {voxel_size}")
    print(f"  网格维度: {grid_dims}")
    print(f"  总体素数: {np.prod(grid_dims)}")
    
    return min_bound, max_bound, tuple(grid_dims)

def get_vertices_texture_mapping(obj_path, texture):
    """
    获取OBJ模型中每个顶点在纹理上的坐标
    
    参数:
    obj_path: OBJ文件路径
    texture: 纹理图像
    
    返回:
    vertex_to_texture: 字典，键为顶点索引，值为该顶点在纹理上的坐标列表(可能有多个)
    faces: 面信息列表，每个面包含顶点索引和对应的纹理坐标索引
    """
    print(f"从OBJ文件读取顶点-纹理映射关系: {obj_path}")
    
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
    
    # 转换为numpy数组
    vertices = np.array(vertices)
    vts = np.array(vts)
    
    print(f"解析结果:")
    print(f"  顶点数量: {len(vertices)}")
    print(f"  纹理坐标数量: {len(vts)}")
    print(f"  面数量: {len(faces)}")
    
    # 创建顶点到纹理坐标的映射
    vertex_to_texture = {}  # 键: 顶点索引, 值: 该顶点使用的纹理坐标列表
    
    # 纹理高度和宽度
    tex_height, tex_width = texture.shape[:2] if texture is not None else (1, 1)
    
    # 遍历所有面，建立映射关系
    for face_vertices, face_vts in faces:
        for v_idx, vt_idx in zip(face_vertices, face_vts):
            # 添加到顶点->纹理映射
            if v_idx not in vertex_to_texture:
                vertex_to_texture[v_idx] = []
                
            # 将纹理坐标转换为像素坐标
            if vt_idx < len(vts):
                vt = vts[vt_idx]
                pixel_x = vt[0] * tex_width
                pixel_y = (1 - vt[1]) * tex_height  # 翻转y坐标
                
                if (pixel_x, pixel_y) not in vertex_to_texture[v_idx]:
                    vertex_to_texture[v_idx].append((pixel_x, pixel_y))
    
    # 统计映射情况
    vertices_with_texture = len(vertex_to_texture)
    avg_textures_per_vertex = sum(len(textures) for textures in vertex_to_texture.values()) / max(1, vertices_with_texture)
    
    print(f"映射统计:")
    print(f"  有纹理映射的顶点数量: {vertices_with_texture}/{len(vertices)} ({vertices_with_texture/len(vertices)*100:.1f}%)")
    print(f"  每个顶点平均使用的纹理坐标数量: {avg_textures_per_vertex:.2f}")
    
    return vertex_to_texture, faces, vertices, vts

def ray_triangle_intersection_batch(ray_origins, ray_directions, triangles):
    """
    使用PyTorch在GPU上批量计算射线与三角形的交点
    
    参数:
    ray_origins: 形状为(N, 3)的张量，表示N条射线的起点
    ray_directions: 形状为(N, 3)的张量，表示N条射线的方向
    triangles: 形状为(M, 3, 3)的张量，表示M个三角形，每个三角形有3个顶点，每个顶点有xyz坐标
    
    返回:
    intersect: 形状为(N, M)的布尔张量，表示每条射线是否与每个三角形相交
    t: 形状为(N, M)的张量，表示每条射线与每个三角形的交点距离
    """
    # 确保输入是张量
    if not isinstance(ray_origins, torch.Tensor):
        ray_origins = torch.tensor(ray_origins, dtype=torch.float32)
    if not isinstance(ray_directions, torch.Tensor):
        ray_directions = torch.tensor(ray_directions, dtype=torch.float32)
    if not isinstance(triangles, torch.Tensor):
        triangles = torch.tensor(triangles, dtype=torch.float32)
    
    # 将数据移动到GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ray_origins = ray_origins.to(device)
    ray_directions = ray_directions.to(device)
    triangles = triangles.to(device)
    
    # 获取三角形的三个顶点
    v0 = triangles[:, 0]  # (M, 3)
    v1 = triangles[:, 1]  # (M, 3)
    v2 = triangles[:, 2]  # (M, 3)
    
    # 计算三角形的两条边
    edge1 = v1 - v0  # (M, 3)
    edge2 = v2 - v0  # (M, 3)
    
    # 准备批量计算
    N = ray_origins.shape[0]
    M = triangles.shape[0]
    
    # 扩展维度以便批量计算
    ray_origins = ray_origins.unsqueeze(1).expand(-1, M, -1)  # (N, M, 3)
    ray_directions = ray_directions.unsqueeze(1).expand(-1, M, -1)  # (N, M, 3)
    
    edge1 = edge1.unsqueeze(0).expand(N, -1, -1)  # (N, M, 3)
    edge2 = edge2.unsqueeze(0).expand(N, -1, -1)  # (N, M, 3)
    v0 = v0.unsqueeze(0).expand(N, -1, -1)  # (N, M, 3)
    
    # 计算Möller–Trumbore算法中的h
    h = torch.cross(ray_directions, edge2, dim=2)  # (N, M, 3)
    
    # 计算a
    a = torch.sum(edge1 * h, dim=2)  # (N, M)
    
    # 如果a接近0，则射线与三角形平行，没有交点
    epsilon = 1e-10
    mask = torch.abs(a) > epsilon  # (N, M)
    
    # 初始化结果
    t = torch.ones((N, M), device=device) * float('inf')
    intersect = torch.zeros((N, M), device=device, dtype=torch.bool)
    
    # 只对非平行的情况进行计算
    if torch.any(mask):
        # 计算f = 1/a
        f = 1.0 / a  # (N, M)
        
        # 计算s = ray_origin - v0
        s = ray_origins - v0  # (N, M, 3)
        
        # 计算u = f * (s · h)
        u = f * torch.sum(s * h, dim=2)  # (N, M)
        
        # 如果u在[0,1]范围外，则没有交点
        u_mask = (u >= 0.0) & (u <= 1.0) & mask  # (N, M)
        
        if torch.any(u_mask):
            # 计算q = s × edge1
            q = torch.cross(s, edge1, dim=2)  # (N, M, 3)
            
            # 计算v = f * (ray_direction · q)
            v = f * torch.sum(ray_directions * q, dim=2)  # (N, M)
            
            # 如果v在[0,1]范围外或u+v>1，则没有交点
            v_mask = (v >= 0.0) & (u + v <= 1.0) & u_mask  # (N, M)
            
            if torch.any(v_mask):
                # 计算t = f * (edge2 · q)
                t_values = f * torch.sum(edge2 * q, dim=2)  # (N, M)
                
                # 如果t>0，则有交点
                t_mask = (t_values > 0.0) & v_mask  # (N, M)
                
                # 更新结果
                t = torch.where(t_mask, t_values, t)
                intersect = t_mask
    
    return intersect, t

@mem.cache
def find_intersecting_voxels_gpu(mesh, bound_low, bound_high, grid_dims, voxel_size, batch_size=1000):
    """
    使用GPU加速查找与模型相交的体素
    
    参数:
    mesh: trimesh网格对象
    bound_low: 网格最小边界点
    bound_high: 网格最大边界点
    grid_dims: 网格维度 (nx, ny, nz)
    voxel_size: 体素尺寸 [x, y, z]
    batch_size: 每批处理的体素数量
    
    返回：
    intersection_grid: volume mask，标记每个体素是否与模型相交
    intersecting_voxels: 相交体素的索引列表 [(i,j,k), ...]
    
    
    目前的算法：
    对每个体素，检查其12条边是否与网格模型的任何三角形相交
    如果体素的任何一条边与模型相交，则认为该体素与模型相交
    """
    print("使用GPU加速查找与模型相交的体素...")
    start_time = time.time()
    
    # 检查是否有可用的GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 准备模型数据
    mesh_vertices = np.array(mesh.vertices)
    mesh_faces = np.array(mesh.faces)
    
    # 创建三角形数组
    triangles = mesh_vertices[mesh_faces]  # (num_faces, 3, 3)
    triangles_tensor = torch.tensor(triangles, dtype=torch.float32, device=device)
    
    # 创建体素的8个顶点模板
    corners_template = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1]
    ])
    
    # 创建体素的12条边
    edges = [
        (0, 1), (0, 2), (1, 3), (2, 3),  # 底面
        (4, 5), (4, 6), (5, 7), (6, 7),  # 顶面
        (0, 4), (1, 5), (2, 6), (3, 7)   # 连接边
    ]
    
    # 初始化结果网格 - 使用0表示不相交，1表示相交
    intersection_grid = np.zeros(grid_dims, dtype=np.float32)
    
    # 存储相交体素的索引和相交的三角形索引
    intersecting_voxels = []
    voxel_to_triangles = {}  # 键: (i,j,k), 值: [三角形索引列表]
    
    # 计算总体素数
    total_voxels = np.prod(grid_dims)
    
    # 分批处理体素
    num_batches = (total_voxels + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, total_voxels)
        
        # 获取当前批次的体素索引
        voxel_indices = np.arange(batch_start, batch_end)
        
        # 将线性索引转换为3D索引
        ix = voxel_indices // (grid_dims[1] * grid_dims[2])
        iy = (voxel_indices % (grid_dims[1] * grid_dims[2])) // grid_dims[2]
        iz = voxel_indices % grid_dims[2]
        
        # 计算体素原点坐标
        origins = np.stack([
            bound_low[0] + ix * voxel_size[0],
            bound_low[1] + iy * voxel_size[1],
            bound_low[2] + iz * voxel_size[2]
        ], axis=1)
        
        # 创建批量体素的所有边
        all_ray_origins = []
        all_ray_directions = []
        all_ray_lengths = []
        all_voxel_indices = []
        all_3d_indices = []
        
        for i, origin in enumerate(origins):
            voxel_idx = voxel_indices[i]
            voxel_3d_idx = (ix[i], iy[i], iz[i])
            
            # 计算体素的8个顶点
            corners = corners_template * voxel_size + origin
            
            # 为每条边创建射线
            for start_idx, end_idx in edges:
                start = corners[start_idx]
                end = corners[end_idx]
                
                # 创建射线
                direction = end - start
                length = np.linalg.norm(direction)
                if length < 1e-10:
                    continue
                    
                direction = direction / length
                
                all_ray_origins.append(start)
                all_ray_directions.append(direction)
                all_ray_lengths.append(length)
                all_voxel_indices.append(voxel_idx)
                all_3d_indices.append(voxel_3d_idx)
        
        if not all_ray_origins:
            continue
        
        # 转换为张量
        ray_origins = torch.tensor(all_ray_origins, dtype=torch.float32, device=device)
        ray_directions = torch.tensor(all_ray_directions, dtype=torch.float32, device=device)
        ray_lengths = torch.tensor(all_ray_lengths, dtype=torch.float32, device=device)
        voxel_indices = torch.tensor(all_voxel_indices, dtype=torch.int64, device=device)
        
        # 计算射线与三角形的交点
        intersect, t = ray_triangle_intersection_batch(ray_origins, ray_directions, triangles_tensor)
        
        # 检查是否有交点在射线长度范围内
        valid_intersect = intersect & (t <= ray_lengths.unsqueeze(1))
        
        # 获取每条射线相交的三角形
        for ray_idx in range(len(all_ray_origins)):
            ray_intersect = valid_intersect[ray_idx]
            if torch.any(ray_intersect):
                voxel_3d_idx = all_3d_indices[ray_idx]
                
                # 获取相交的三角形索引
                triangle_indices = torch.where(ray_intersect)[0].cpu().numpy()
                
                # 更新相交网格
                i, j, k = voxel_3d_idx
                intersection_grid[i, j, k] = 1.0
                
                # 添加到相交体素列表
                if voxel_3d_idx not in intersecting_voxels:
                    intersecting_voxels.append(voxel_3d_idx)
                
                # 记录体素相交的三角形
                if voxel_3d_idx not in voxel_to_triangles:
                    voxel_to_triangles[voxel_3d_idx] = []
                voxel_to_triangles[voxel_3d_idx].extend(triangle_indices)
        
        # 打印进度
        elapsed_time = time.time() - start_time
        print(f"处理进度： {batch_end}/{total_voxels} "
              f"({batch_end / total_voxels * 100:.1f}%) "
              f"- 已用时间: {elapsed_time:.2f}秒")
    
    # 去除重复的三角形索引
    for voxel_idx in voxel_to_triangles:
        voxel_to_triangles[voxel_idx] = list(set(voxel_to_triangles[voxel_idx]))
    
    # 统计相交体素数量
    num_intersecting = len(intersecting_voxels)
    print(f"找到 {num_intersecting} 个与模型相交的体素")
    print(f"总用时: {time.time() - start_time:.2f}秒")
    
    return intersection_grid, intersecting_voxels, voxel_to_triangles

def project_point_to_triangle(point, triangle):
    """
    将点投影到三角形上
    
    参数:
    point: 要投影的点 [x, y, z]
    triangle: 三角形的三个顶点 [[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]]
    
    返回：
    projected_point: 投影点 [x, y, z]
    barycentric: 重心坐标 [u, v, w]
    """
    # 获取三角形的三个顶点
    v0, v1, v2 = triangle
    
    # 计算三角形的法向量
    normal = np.cross(v1 - v0, v2 - v0)
    normal = normal / np.linalg.norm(normal)
    
    # 计算点到三角形平面的距离
    d = np.dot(normal, v0)
    dist = np.dot(normal, point) - d
    
    # 将点投影到三角形平面上
    projected_point = point - dist * normal
    
    # 计算重心坐标
    # 使用面积法计算重心坐标
    area_triangle = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
    
    area_0 = 0.5 * np.linalg.norm(np.cross(v1 - projected_point, v2 - projected_point))
    area_1 = 0.5 * np.linalg.norm(np.cross(v2 - projected_point, v0 - projected_point))
    area_2 = 0.5 * np.linalg.norm(np.cross(v0 - projected_point, v1 - projected_point))
    
    # 计算重心坐标
    u = area_0 / area_triangle if area_triangle > 0 else 0
    v = area_1 / area_triangle if area_triangle > 0 else 0
    w = area_2 / area_triangle if area_triangle > 0 else 0
    
    # 处理数值误差，确保重心坐标和为1
    total = u + v + w
    if total > 0:
        u, v, w = u/total, v/total, w/total
    
    return projected_point, np.array([u, v, w])

def get_color_from_texture(pixel_coords, texture):
    """
    从纹理图像中获取指定像素坐标的颜色
    
    参数:
    pixel_coords: 像素坐标 [x, y]
    texture: 纹理图像
    
    返回：
    color: RGB颜色值 [r, g, b]
    """
    if texture is None:
        return np.array([128, 128, 128])  # 默认灰色
    
    # 获取纹理图像的尺寸
    height, width = texture.shape[:2]
    
    # 确保像素坐标在有效范围内
    x = int(pixel_coords[0]) % width
    y = int(pixel_coords[1]) % height
    
    # 获取颜色
    color = texture[y, x]
    
    return color

def create_colored_point_cloud(mesh, texture, obj_path, intersecting_voxels, voxel_to_triangles, bound_low, voxel_size):
    """
    为相交的体素创建带颜色的点云
    
    参数:
    mesh: trimesh网格对象
    texture: 纹理图像
    obj_path: OBJ文件路径
    intersecting_voxels: 相交体素的索引列表 [(i,j,k), ...]
    voxel_to_triangles: 体素到三角形的映射 {(i,j,k): [三角形索引列表]}
    bound_low: 网格最小边界点
    voxel_size: 体素尺寸 [x, y, z]
    
    返回:
    points: 点云坐标 Nx3
    colors: 点云颜色 Nx3
    """
    print("创建带颜色的点云...")
    
    # 获取顶点-纹理映射关系
    vertex_to_texture, faces, vertices, vts = get_vertices_texture_mapping(obj_path, texture)
    # vertex_to_texture v的index到texture图片2D坐标
    
    # 获取模型的面和顶点
    mesh_vertices = np.array(mesh.vertices)
    mesh_faces = np.array(mesh.faces)
    triangles = mesh_vertices[mesh_faces]
    
    # 初始化点云和颜色
    points = []
    colors = []
    
    # 处理每个相交的体素
    for i, j, k in intersecting_voxels:
        # 计算体素中心
        voxel_center = np.array([
            bound_low[0] + (i + 0.5) * voxel_size[0],
            bound_low[1] + (j + 0.5) * voxel_size[1],
            bound_low[2] + (k + 0.5) * voxel_size[2]
        ])
        
        # 获取该体素相交的三角形
        triangle_indices = voxel_to_triangles.get((i, j, k), [])
        
        if not triangle_indices:
            continue
        
        # 选择第一个相交的三角形
        triangle_idx = triangle_indices[0]
        triangle = triangles[triangle_idx]
        
        # 将体素中心投影到三角形上
        projected_point, barycentric = project_point_to_triangle(voxel_center, triangle)
        
        # 获取三角形的顶点索引
        v_indices = mesh_faces[triangle_idx]
        
        # 获取三角形顶点的纹理坐标
        pixel_coords = []
        for v_idx in v_indices:
            if v_idx in vertex_to_texture and vertex_to_texture[v_idx]:
                # 使用第一个纹理坐标
                pixel_coords.append(vertex_to_texture[v_idx][0])
            else:
                # 如果没有纹理坐标，使用默认值
                pixel_coords.append((0, 0))
        
        # 使用重心坐标插值计算纹理坐标
        if len(pixel_coords) == 3:
            interpolated_pixel = (
                barycentric[0] * np.array(pixel_coords[0]) +
                barycentric[1] * np.array(pixel_coords[1]) +
                barycentric[2] * np.array(pixel_coords[2])
            )
            
            # 获取纹理颜色
            color = get_color_from_texture(interpolated_pixel, texture)
            
            # 添加到点云
            points.append(projected_point)
            colors.append(color)
    
    return np.array(points), np.array(colors)

def visualize_with_polyscope(mesh, bound_low, bound_high, grid_dims, intersection_grid, points, colors):
    """
    使用polyscope可视化模型、体素网格和带颜色的点云
    
    参数:
    mesh: trimesh网格对象
    bound_low: 网格最小边界点
    bound_high: 网格最大边界点
    grid_dims: 网格维度 (nx, ny, nz)
    intersection_grid: 形状为grid_dims的3D数组，标记每个体素是否与模型相交
    points: 点云坐标 Nx3
    colors: 点云颜色 Nx3 或 Nx4 (RGBA)
    """
    print("使用polyscope可视化结果...")
    
    # 初始化polyscope
    ps.init()
    
    # 注册网格
    vertices = mesh.vertices
    faces = mesh.faces
    ps_mesh = ps.register_surface_mesh("model", vertices, faces)
    ps_mesh.set_color((0.8, 0.8, 0.8))
    ps_mesh.set_transparency(0.5)
    
    # 注册体素网格
    ps_grid = ps.register_volume_grid("voxel_grid", grid_dims, bound_low, bound_high)
    
    # 添加标量场，用于区分相交和不相交的体素
    ps_grid.add_scalar_quantity(
        "intersection", 
        intersection_grid,
        defined_on='nodes',  # 定义在节点上
        enabled=True,
        vminmax=(0.0, 1.0),  # 值的范围
        cmap="coolwarm"      # 颜色映射
    )
    
    # 注册带颜色的点云
    if len(points) > 0:
        ps_points = ps.register_point_cloud("projected_points", points)
        
        # 设置点云颜色
        if colors.shape[1] >= 3:  # RGB或RGBA颜色
            # 将RGB颜色值归一化到[0,1]范围
            normalized_rgb = colors[:, :3].astype(float) / 255.0
            ps_points.add_color_quantity("colors", normalized_rgb, enabled=True)
            
            # 如果有Alpha通道，设置透明度
            if colors.shape[1] == 4:
                # 将Alpha值归一化到[0,1]范围
                alpha = colors[:, 3].astype(float) / 255.0
                ps_points.add_scalar_quantity("alpha", alpha, enabled=False)
                ps_points.set_transparency_quantity("alpha")
        
        # 设置点云大小
        ps_points.set_radius(0.01)  # 调整点的大小
    
    # 显示polyscope界面
    ps.show()

def main(obj_path, use_gpu=True, batch_size=1000):
    """
    主函数
    
    参数：
    obj_path: OBJ文件路径
    use_gpu: 是否使用GPU加速
    batch_size: GPU批处理大小
    """
    # 加载OBJ模型和贴图
    mesh, texture, texture_path = load_obj_and_texture(obj_path) 
    # mesh->Trimesh, texture->numpy.ndarray, texture_path->str
    
    # 创建体素网格
    voxel_size=getVoxelSize()
    bound_low, bound_high, grid_dims = create_voxel_grid(mesh, voxel_size=getVoxelSize())
    # bound_low, bound_high, grid_dims = create_voxel_grid(mesh, voxel_size=[5e-2]*3)
    # bound_low, bound_high 是体素网格的边界点(3D position)，grid_dims是体素每个维度的voxel数量
    
    # 查找与模型相交的体素
    if use_gpu and torch.cuda.is_available():
        intersection_grid, intersecting_voxels, voxel_to_triangles = find_intersecting_voxels_gpu(
            mesh, bound_low, bound_high, grid_dims, voxel_size, batch_size)
        # intersection_grid->volume mask 1 代表相交, intersecting_voxels->相交体素的索引列表 [(i,j,k), ...], voxel_to_triangles->dict
    else:
        if use_gpu and not torch.cuda.is_available():
            raise ValueError("请求使用GPU但没有可用的CUDA设备")
    
    # 创建带颜色的点云
    points, colors = create_colored_point_cloud(
        mesh, texture, obj_path, intersecting_voxels, voxel_to_triangles, bound_low, voxel_size)
    breakpoint()
    # 使用polyscope可视化结果
    visualize_with_polyscope(mesh, bound_low, bound_high, grid_dims, intersection_grid, points, colors)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='体素化3D模型并查找与模型相交的体素')
    parser.add_argument('--obj_path', type=str, default=None, help='OBJ文件路径')
    parser.add_argument('--cpu', action='store_true', help='强制使用CPU多线程而不是GPU')
    parser.add_argument('--batch-size', type=int, default=1000, help='GPU批处理大小')
    
    args = parser.parse_args()
    
    
    main( args.obj_path, not args.cpu, args.batch_size)
