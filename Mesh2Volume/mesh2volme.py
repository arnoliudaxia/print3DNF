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
import pyvista as pv
mem = Memory(".cache")


script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(os.path.join(parent_dir, "util"))

from printer.printerHelper import getVoxelSize

def load_obj_with_texture_mapping(obj_path, scale=1.0):
    """
    加载OBJ模型、对应的贴图以及顶点和纹理坐标
    
    参数:
    obj_path: OBJ文件路径
    scale: 模型缩放系数，默认为1.0
    
    返回:
    mesh: trimesh网格对象
    texture: 贴图图像（如果存在）
    texture_path: 贴图文件路径
    faces: 面信息列表，每个面包含顶点索引
    vertices: 顶点坐标列表
    vts: 纹理坐标列表，与顶点一一对应
    """
    print(f"加载OBJ模型: {obj_path}")
    
    # 使用trimesh加载OBJ文件
    mesh = trimesh.load(obj_path)
    
    # 应用缩放
    if scale != 1.0:
        print(f"应用模型缩放: {scale}")
        mesh.apply_scale(scale)
    
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
    
    # 使用trimesh API获取顶点和面
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    
    # 使用trimesh的视觉属性获取纹理坐标
    vts = []
    if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'uv'):
        vts = np.array(mesh.visual.uv)
        print(f"从trimesh获取到 {len(vts)} 个纹理坐标")
    else:
        print("mesh没有纹理坐标信息")
        vts = np.array([])
    
    print(f"解析结果:")
    print(f"  顶点数量: {len(vertices)}")
    print(f"  纹理坐标数量: {len(vts)}")
    print(f"  面数量: {len(faces)}")
    
    # 检查顶点和纹理坐标是否一一对应
    if len(vts) > 0 and len(vts) != len(vertices):
        print(f"警告: 顶点数量 ({len(vertices)}) 与纹理坐标数量 ({len(vts)}) 不匹配")
    
    # 如果使用trimesh API无法获取完整的纹理坐标，回退到手动解析OBJ文件
    if len(vts) == 0 and texture is not None:
        print("使用trimesh无法获取纹理坐标，回退到手动解析OBJ文件...")
        # 读取OBJ文件
        with open(obj_path, 'r') as f:
            lines = f.readlines()
        
        # 解析顶点和纹理坐标
        obj_vertices = []
        obj_vts = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('v '):
                # 解析顶点坐标
                parts = line.split()
                if len(parts) >= 4:
                    obj_vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('vt '):
                # 解析纹理坐标
                parts = line.split()
                if len(parts) >= 3:
                    obj_vts.append([float(parts[1]), float(parts[2])])
        
        # 确保顶点和纹理坐标数量一致
        if len(obj_vts) > 0:
            print(f"手动解析结果:")
            print(f"  纹理坐标数量: {len(obj_vts)}")
            
            # 如果纹理坐标和顶点数量相同，直接使用
            if len(obj_vts) == len(vertices):
                vts = np.array(obj_vts)
                print(f"使用手动解析的纹理坐标 (顶点和纹理坐标数量相同)")
            else:
                print(f"警告: 手动解析后顶点数量 ({len(vertices)}) 与纹理坐标数量 ({len(obj_vts)}) 不匹配")
                # 这里可以添加额外的处理代码...
    
    # 纹理坐标转换为像素坐标
    if texture is not None and len(vts) > 0:
        tex_height, tex_width = texture.shape[:2]
        pixel_vts = np.zeros_like(vts)
        pixel_vts[:, 0] = vts[:, 0] * tex_width
        pixel_vts[:, 1] = (1 - vts[:, 1]) * tex_height  # 翻转y坐标
        vts = pixel_vts
    
    return mesh, texture, texture_path, faces, vertices, vts

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
    voxel_centers: 体素中心坐标张量，形状为[grid_dims, 3]
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
    
    # 创建体素中心坐标张量
    x = np.linspace(min_bound[0] + voxel_size[0]/2, max_bound[0] - voxel_size[0]/2, grid_dims[0])
    y = np.linspace(min_bound[1] + voxel_size[1]/2, max_bound[1] - voxel_size[1]/2, grid_dims[1])
    z = np.linspace(min_bound[2] + voxel_size[2]/2, max_bound[2] - voxel_size[2]/2, grid_dims[2])
    
    # 使用meshgrid创建坐标网格
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    
    # 将坐标组合成[N,3]形状的张量
    voxel_centers = np.stack([xx, yy, zz], axis=-1)
    
    return min_bound, max_bound, tuple(grid_dims), voxel_centers


# @mem.cache
def find_intersecting_voxels_gpu(mesh, voxel_centers, grid_dims, voxel_size):
    """
    使用GPU加速查找与模型相交的体素
    
    参数:
    mesh: trimesh网格对象
    voxel_centers: 体素中心坐标张量，形状为[N, 3]，其中N是总体素数
    grid_dims: 网格维度 (nx, ny, nz)
    voxel_size: 体素尺寸 [x, z, y]
    
    返回：
    intersection_grid: volume mask，标记每个体素是否与模型相交
    intersecting_voxels: 相交体素的索引列表 [(i,j,k), ...]
    voxel_to_triangles: 字典，键为体素3D索引(i,j,k)，值为相交的三角形索引列表
    voxel_intersection_points: 字典，键为体素3D索引(i,j,k)，值为该体素与三角形面的相交点列表
    voxel_barycentric_data: 字典，键为体素3D索引(i,j,k)，值为字典{triangle_idx: (vertices_indices, barycentric_weights)}
    
    算法：
    1. 首先进行bbox的快速相交检测
    2. 然后对潜在相交的三角形和体素进行精确相交检测
    3. 计算交点的平均中心，并求出相对于三角形顶点的权重
    """
    print("使用GPU加速查找与模型相交的体素...")
    start_time = time.time()
    voxel_centers=voxel_centers.reshape(-1,3)
    
    # 检查是否有可用的GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 准备模型数据
    mesh_vertices = np.array(mesh.vertices)
    mesh_faces = np.array(mesh.faces)
    
    # 创建三角形数组
    triangles = mesh_vertices[mesh_faces]  # (num_faces, 3, 3)
    triangles_tensor = torch.tensor(triangles, dtype=torch.float32, device=device)
    
    # 计算每个三角形的边界框
    # 获取每个三角形的最大和最小坐标
    face_min = torch.min(triangles_tensor, dim=1)[0]  # (num_faces, 3)
    face_max = torch.max(triangles_tensor, dim=1)[0]  # (num_faces, 3)
    
    # 计算每个体素的边界框
    voxel_size_tensor = torch.tensor(voxel_size, dtype=torch.float32, device=device)
    voxel_half_size = voxel_size_tensor / 2.0
    
    # 将体素中心转换为张量
    voxel_centers_tensor = torch.tensor(voxel_centers, dtype=torch.float32, device=device)
    
    # 计算体素的边界框
    voxel_min = voxel_centers_tensor - voxel_half_size  # (N, 3)
    voxel_max = voxel_centers_tensor + voxel_half_size  # (N, 3)
    
    # 扩展维度以便进行批量比较
    # face_min: (num_faces, 3) -> (1, num_faces, 3)
    # voxel_max: (N, 3) -> (N, 1, 3)
    face_min_expanded = face_min.unsqueeze(0)  # (1, num_faces, 3)
    face_max_expanded = face_max.unsqueeze(0)  # (1, num_faces, 3)
    voxel_min_expanded = voxel_min.unsqueeze(1)  # (N, 1, 3)
    voxel_max_expanded = voxel_max.unsqueeze(1)  # (N, 1, 3)
    
    # 检查边界框是否相交
    # 两个边界框相交的条件是：
    # 在每个维度上，一个框的最小值小于另一个框的最大值
    # 且一个框的最大值大于另一个框的最小值
    intersect_x = (voxel_min_expanded[..., 0] < face_max_expanded[..., 0]) & (voxel_max_expanded[..., 0] > face_min_expanded[..., 0])
    intersect_y = (voxel_min_expanded[..., 1] < face_max_expanded[..., 1]) & (voxel_max_expanded[..., 1] > face_min_expanded[..., 1])
    intersect_z = (voxel_min_expanded[..., 2] < face_max_expanded[..., 2]) & (voxel_max_expanded[..., 2] > face_min_expanded[..., 2])
    
    # 在所有维度上都相交
    bbox_intersect = intersect_x & intersect_y & intersect_z  # (N, num_faces)
    
    # 获取与任何三角形边界框相交的体素
    voxel_intersect = torch.any(bbox_intersect, dim=1)  # (N,)
    
    # 获取相交体素的索引
    intersecting_indices = torch.where(voxel_intersect)[0].cpu().numpy()
    
    # 使用传入的grid_dims计算3D索引
    nx, ny, nz = grid_dims
    # 将线性索引转换为3D索引
    ix = intersecting_indices // (ny * nz)
    iy = (intersecting_indices % (ny * nz)) // nz
    iz = intersecting_indices % nz
    
    intersecting_voxels = list(zip(ix, iy, iz))
    
    # 创建相交网格
    bboxInterCounter=0
    intersection_grid = np.zeros(grid_dims, dtype=np.float32)
    for i, j, k in intersecting_voxels:
        intersection_grid[i, j, k] = 1.0
        bboxInterCounter+=1
    print(f"bbox相交的voxel数量: {bboxInterCounter}")
    
    # 记录每个体素相交的三角形
    voxel_to_triangles = {}
    # 记录每个体素与三角形的相交点
    voxel_intersection_points = {}
    # 记录每个体素与三角形的重心坐标数据
    voxel_barycentric_data = {}
    
    print("计算三角形面与体素的精确相交...")
    
    # 对可能相交的体素进行精确的相交检测
    for idx, voxel_idx in enumerate(intersecting_indices):
        # 获取与该体素相交的三角形索引
        triangle_indices = torch.where(bbox_intersect[voxel_idx])[0].cpu().numpy()
        voxel_3d_idx = intersecting_voxels[idx]
        
        # 获取该体素的中心点和边界
        center = voxel_centers[voxel_idx]
        vmin = center - voxel_half_size.cpu().numpy()
        vmax = center + voxel_half_size.cpu().numpy()
        
        # 创建体素的8个顶点
        voxel_vertices = np.array([
            [vmin[0], vmin[1], vmin[2]],  # 0
            [vmax[0], vmin[1], vmin[2]],  # 1
            [vmin[0], vmax[1], vmin[2]],  # 2
            [vmax[0], vmax[1], vmin[2]],  # 3
            [vmin[0], vmin[1], vmax[2]],  # 4
            [vmax[0], vmin[1], vmax[2]],  # 5
            [vmin[0], vmax[1], vmax[2]],  # 6
            [vmax[0], vmax[1], vmax[2]]   # 7
        ])
        
        # 创建体素的12条边（起点，终点）
        voxel_edges = [
            (0, 1), (0, 2), (1, 3), (2, 3),  # 底面
            (4, 5), (4, 6), (5, 7), (6, 7),  # 顶面
            (0, 4), (1, 5), (2, 6), (3, 7)   # 连接边
        ]
        
        # 存储该体素相交的三角形和相交点
        intersecting_triangle_indices = []
        intersection_points = []
        # 存储每个三角形的相交点
        triangle_to_points = {}
        
        for tri_idx in triangle_indices:
            # 获取三角形的顶点
            tri_vertices = triangles[tri_idx]
            v0, v1, v2 = tri_vertices
            
            # 计算三角形的法向量和平面方程
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            normal = normal / np.linalg.norm(normal)
            d = -np.dot(normal, v0)
            
            # 检查体素的每条边是否与三角形相交
            has_intersection = False
            tri_intersection_points = []
            
            for start_idx, end_idx in voxel_edges:
                start = voxel_vertices[start_idx]
                end = voxel_vertices[end_idx]
                
                # 计算射线与平面的交点
                direction = end - start
                denom = np.dot(normal, direction)
                
                # 如果射线与平面平行，则无交点
                if abs(denom) < 1e-6:
                    continue
                
                t = -(np.dot(normal, start) + d) / denom
                
                # 如果交点不在线段上，则跳过
                if t < 0 or t > 1:
                    continue
                
                # 计算交点
                intersection = start + t * direction
                
                # 检查交点是否在三角形内
                # 使用重心坐标判断
                edge1 = v1 - v0
                edge2 = v2 - v0
                h = np.cross(direction, edge2)
                a = np.dot(edge1, h)
                
                if abs(a) < 1e-6:  # 平行情况
                    continue
                
                f = 1.0 / a
                s = intersection - v0
                u = f * np.dot(s, h)
                
                if u < 0.0 or u > 1.0:
                    continue
                
                q = np.cross(s, edge1)
                v = f * np.dot(direction, q)
                
                if v < 0.0 or u + v > 1.0:
                    continue
                
                # 找到了有效的交点
                has_intersection = True
                intersection_points.append(intersection)
                tri_intersection_points.append(intersection)
            
            if has_intersection:
                intersecting_triangle_indices.append(tri_idx)
                triangle_to_points[tri_idx] = tri_intersection_points
        
        if intersecting_triangle_indices:
            voxel_to_triangles[voxel_3d_idx] = intersecting_triangle_indices
            voxel_intersection_points[voxel_3d_idx] = intersection_points
            
            # 为每个相交的三角形计算重心坐标数据
            barycentric_data = {}
            for tri_idx in intersecting_triangle_indices:
                # 获取三角形顶点索引
                v_indices = mesh_faces[tri_idx]
                
                # 获取三角形顶点坐标
                tri_vertices = triangles[tri_idx]
                
                # 计算该三角形所有交点的平均中心
                if tri_idx in triangle_to_points and triangle_to_points[tri_idx]:
                    center_point = np.mean(triangle_to_points[tri_idx], axis=0)
                    
                    # 计算平均中心到三角形的投影和重心坐标
                    projected_point, barycentric_weights = project_point_to_triangle(center_point, tri_vertices)
                    
                    # 存储三角形顶点索引和重心坐标
                    barycentric_data[tri_idx] = (v_indices, barycentric_weights)
            
            voxel_barycentric_data[voxel_3d_idx] = barycentric_data
    
    # 更新相交网格，只保留精确相交的体素
    intersection_grid = np.zeros(grid_dims, dtype=np.float32)
    for voxel_idx in voxel_to_triangles:
        i, j, k = voxel_idx
        intersection_grid[i, j, k] = 1.0
    
    # 更新相交体素列表
    intersecting_voxels = list(voxel_to_triangles.keys())
    
    print(f"找到 {len(intersecting_voxels)} 个与模型精确相交的体素")
    print(f"总用时: {time.time() - start_time:.2f}秒")
    
    return intersection_grid, intersecting_voxels, voxel_to_triangles, voxel_intersection_points, voxel_barycentric_data

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

def visualize_texture_mapping(texture, texture_points, vts=None, faces=None, output_path=None):
    """
    将纹理坐标点叠加在纹理图像上，用于可视化插值点的位置
    
    参数:
    texture: 纹理图像，numpy数组
    texture_points: 插值的纹理坐标点列表，像素坐标 [(x1, y1), (x2, y2), ...]
    vts: 原始纹理坐标列表，UV坐标 [[u1, v1], [u2, v2], ...]，可选
    faces: 面信息列表，每个面包含顶点索引和对应的纹理坐标索引，可选
    output_path: 输出图像的保存路径，如果为None则使用默认路径
    """
    if texture is None:
        print("无法可视化纹理坐标：纹理图像为空")
        return
    
    import matplotlib.pyplot as plt
    
    # 创建图形
    plt.figure(figsize=(10, 10))
    
    # 显示贴图作为背景
    plt.imshow(texture)
    
    # 纹理高度和宽度
    tex_height, tex_width = texture.shape[:2]
    
    # 如果有原始纹理坐标和面信息，绘制UV网格
    if vts is not None and len(vts) > 0:
        # 在贴图上显示所有vt点
        scatter_x = [vt[0] for vt in vts]
        scatter_y = [vt[1] for vt in vts]
        plt.scatter(scatter_x, scatter_y, s=10, c='blue', alpha=0.5, label='Original Texture Coordinates (vt)')
        
        # 如果有面信息，绘制UV三角形
        if faces is not None and len(faces) > 0:
            for face_vertices, face_vts in faces:
                if len(face_vts) >= 3:  # 确保至少有3个点形成三角形
                    # 获取面的UV坐标
                    uv_coords = np.array([vts[idx] for idx in face_vts])
                    
                    # 绘制三角形边缘
                    for i in range(len(uv_coords)):
                        j = (i + 1) % len(uv_coords)
                        plt.plot([uv_coords[i][0], uv_coords[j][0]], 
                                [uv_coords[i][1], uv_coords[j][1]], 
                                'g-', linewidth=0.5, alpha=0.4)
    
    # 在贴图上显示插值的纹理坐标点
    if len(texture_points) > 0:
        interp_x = [pt[0] for pt in texture_points]
        interp_y = [pt[1] for pt in texture_points]
        plt.scatter(interp_x, interp_y, s=20, c='red', alpha=0.7, label='Interpolated Texture Points')
    
    plt.title("Texture Mapping Visualization")
    plt.legend()
    
    # 保存图像
    if output_path is None:
        output_path = 'texture_mapping_visualization.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"纹理坐标可视化图像已保存到: {output_path}")
    
    # 显示图像
    plt.show()
    
    return

def create_colored_point_cloud(mesh, texture, vertices, vts, voxel_centers, intersecting_voxels, voxel_to_triangles, voxel_barycentric_data):
    """
    为相交的体素创建带颜色的点云
    
    参数:
    mesh: trimesh网格对象
    texture: 纹理图像
    vertices: 顶点坐标列表
    vts: 纹理坐标列表，与顶点一一对应
    voxel_centers: 体素中心坐标, 形状为[grid_dims, 3]
    intersecting_voxels: 相交体素的索引列表 [(i,j,k), ...]
    voxel_to_triangles: 体素到三角形的映射 {(i,j,k): [三角形索引列表]}
    voxel_barycentric_data: 字典，键为体素3D索引(i,j,k)，值为字典{triangle_idx: (vertices_indices, barycentric_weights)}
    
    返回:
    points: 点云坐标 Nx3
    colors: 点云颜色 Nx3
    interpolated_texture_points: 所有插值后的纹理坐标点 [(x1, y1), (x2, y2), ...]
    """
    print("创建带颜色的点云...")
    
    # 获取模型的面和顶点
    mesh_vertices = np.array(mesh.vertices)
    mesh_faces = np.array(mesh.faces)
    triangles = mesh_vertices[mesh_faces]
    
    # 初始化点云和颜色
    points = []
    colors = []
    interpolated_texture_points = []
    
    # 检查顶点和纹理坐标是否一一对应
    has_texture_coords = len(vts) > 0 and len(vts) == len(vertices)
    if not has_texture_coords:
        print("警告: 顶点和纹理坐标不匹配，无法正确映射纹理")
    
    # 处理每个相交的体素
    for voxel_idx in intersecting_voxels:
        # 如果该体素没有重心坐标数据，则跳过
        if voxel_idx not in voxel_barycentric_data:
            continue
        
        # 获取该体素的重心坐标数据
        barycentric_dict = voxel_barycentric_data[voxel_idx]
        
        # 对于每个相交的三角形
        for tri_idx, (v_indices, barycentric_weights) in barycentric_dict.items():
            # 获取三角形的重心投影点（使用barycentric_weights和三角形顶点）
            triangle = triangles[tri_idx]
            projected_point = (
                barycentric_weights[0] * triangle[0] +
                barycentric_weights[1] * triangle[1] +
                barycentric_weights[2] * triangle[2]
            )
            
            # 如果有纹理坐标，计算颜色
            if has_texture_coords and texture is not None:
                # 获取该三角形三个顶点的纹理坐标
                pixel_coords = [vts[v_idx] for v_idx in v_indices]
                
                # 使用重心坐标插值计算纹理坐标
                interpolated_pixel = (
                    barycentric_weights[0] * np.array(pixel_coords[0]) +
                    barycentric_weights[1] * np.array(pixel_coords[1]) +
                    barycentric_weights[2] * np.array(pixel_coords[2])
                )
                
                # 保存插值后的纹理坐标点
                interpolated_texture_points.append((interpolated_pixel[0], interpolated_pixel[1]))
                
                # 从纹理中获取颜色
                color = get_color_from_texture(interpolated_pixel, texture)
            else:
                # 如果没有纹理坐标或纹理，使用默认灰色
                color = np.array([128, 128, 128])
            
            # 添加到点云
            points.append(projected_point)
            colors.append(color)
            
            # 每个体素只处理一个三角形（通常是最接近的那个）
            break
    
    return np.array(points), np.array(colors), interpolated_texture_points

def visualize_with_polyscope(mesh, bound_low, bound_high, grid_dims, intersection_grid, pointsW, colors, pointColouds=None):
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
    
    colors=colors[:,:3]
    
    # 初始化polyscope
    ps.init()
    
    if pointColouds is not None:
        for name, points in pointColouds:
            ps.register_point_cloud(name, points)
            # ps.set_enabled(False)
    
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
    
    ps_grid.set_enabled(False)
    
    # 注册带颜色的点云
    if len(pointsW) > 0:
        ps_points = ps.register_point_cloud("projected_points", pointsW)
        
        # 设置点云颜色
        if len(colors) > 0 and colors.shape[1] >= 3:  # RGB或RGBA颜色
            # 将RGB颜色值归一化到[0,1]范围
            ps_points.add_color_quantity("colors", colors, enabled=True)
            
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

def main(obj_path, use_gpu=True, batch_size=1000, scale=1.0):
    """
    主函数
    
    参数：
    obj_path: OBJ文件路径
    use_gpu: 是否使用GPU加速
    batch_size: GPU批处理大小
    scale: 模型缩放系数
    """
    # 加载OBJ模型和贴图
    mesh, texture, texture_path, faces, vertices, vts = load_obj_with_texture_mapping(obj_path, scale) 
    # mesh->Trimesh, texture->numpy.ndarray, texture_path->str
    # Trimesh 的坐标系是xzy
    
    # 创建体素网格
    x,y,z=getVoxelSize()
    x,y,z=5e-2,5e-2,5e-2
    bound_low, bound_high, grid_dims, voxel_centers = create_voxel_grid(mesh, voxel_size=[x,z,y])
    # pyvista 可视化voxel_centers
    # p = pv.Plotter()
    # p.add_points(voxel_centers.reshape(-1, 3), render_points_as_spheres=True, point_size=10, color='red')
    # p.show()

    # bound_low, bound_high 是体素网格的边界点(3D position)，grid_dims是体素每个维度的voxel数量
    
    # 查找与模型相交的体素
    if use_gpu and torch.cuda.is_available():
        intersection_grid, intersecting_voxels, voxel_to_triangles, voxel_intersection_points, voxel_barycentric_data = find_intersecting_voxels_gpu(
            mesh, voxel_centers, grid_dims,voxel_size=[x,z,y])
        # intersection_grid->volume mask 1 代表相交, intersecting_voxels->相交体素的索引列表 [(i,j,k), ...], voxel_to_triangles->dict 每一个voxel和哪些faces相交, voxel_intersection_points-> dic[list] 每一个voxel和相交face的交点list
    else:
        if use_gpu and not torch.cuda.is_available():
            raise ValueError("请求使用GPU但没有可用的CUDA设备")
    visInterPoints=[]
    for interpoints in list(voxel_intersection_points.values()):
        for point in interpoints:
            visInterPoints.append(point)
    visInterPoints=np.array(visInterPoints)
    
    # 创建带颜色的点云
    points, colors = [], []
    points, colors, interpolated_texture_points = create_colored_point_cloud(
        mesh, texture, vertices, vts, voxel_centers, intersecting_voxels, voxel_to_triangles, voxel_barycentric_data)

    # 可视化纹理坐标
    if texture is not None and len(interpolated_texture_points) > 0:
        # 获取OBJ文件中的面信息，用于绘制UV网格
        obj_faces = []
        
        # 尝试手动解析OBJ文件获取面-纹理映射信息
        try:
            with open(obj_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if line.startswith('f '):
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
                        obj_faces.append((face_vertices, face_vts))
        except Exception as e:
            print(f"解析OBJ文件获取面信息时出错: {e}")
            obj_faces = []
        
        # 生成输出路径
        texture_vis_path = None
        if texture_path:
            texture_dir = os.path.dirname(texture_path)
            texture_basename = os.path.basename(texture_path)
            texture_name, texture_ext = os.path.splitext(texture_basename)
            texture_vis_path = os.path.join(texture_dir,'mapping',  f"{texture_name}_mapping_vis.png")
        
        # 可视化插值的纹理坐标点
        visualize_texture_mapping(texture, interpolated_texture_points, vts, obj_faces, texture_vis_path)
    colors=colors.astype(np.float32)/255.0
    # 使用polyscope可视化结果
    visualize_with_polyscope(mesh, bound_low, bound_high, grid_dims, intersection_grid, points, colors, pointColouds=[
        ("voxel_intersection_points", visInterPoints),
    ])

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='体素化3D模型并查找与模型相交的体素')
    parser.add_argument('--obj_path', type=str, default=None, help='OBJ文件路径')
    parser.add_argument('--cpu', action='store_true', help='强制使用CPU多线程而不是GPU')
    parser.add_argument('--batch-size', type=int, default=1000, help='GPU批处理大小')
    parser.add_argument('--scale', type=float, default=1.0, help='模型缩放系数，默认为1.0')
    
    args = parser.parse_args()
    
    
    main(args.obj_path, not args.cpu, args.batch_size, args.scale)
