import numpy as np
import trimesh
import polyscope as ps
import os
from PIL import Image
import torch
import time

def load_obj_and_texture(obj_path):
    """
    加载OBJ模型和对应的贴图
    
    参数:
    obj_path: OBJ文件路径
    
    返回:
    mesh: trimesh网格对象
    texture: 贴图图像（如果存在）
    """
    print(f"加载OBJ模型: {obj_path}")
    
    # 使用trimesh加载OBJ文件
    mesh = trimesh.load(obj_path)
    
    # 获取贴图
    texture = None
    obj_dir = os.path.dirname(obj_path)
    tex_name = os.path.splitext(os.path.basename(obj_path))[0] + '_TEX.png'
    tex_path = os.path.join(obj_dir, tex_name)
    
    if os.path.exists(tex_path):
        print(f"从文件加载贴图: {tex_path}")
        texture = np.array(Image.open(tex_path))
    else:
        print("找不到贴图文件")
    
    return mesh, texture

def create_voxel_grid(mesh, voxel_size_z=0.014):
    """
    创建一个体素网格，基于模型的边界框
    
    参数:
    mesh: trimesh网格对象
    voxel_size_z: Z方向的体素大小
    
    返回:
    bound_low: 网格最小边界点
    bound_high: 网格最大边界点
    grid_dims: 网格维度 [nx, ny, nz]
    voxel_size: 体素尺寸 [x, y, z]
    """
    # 计算模型的边界框
    bounds = mesh.bounds
    min_bound = bounds[0]
    max_bound = bounds[1]
    
    # 计算体素尺寸
    voxel_size = np.array([
        voxel_size_z * 2 * 3,  # x = y * 3
        voxel_size_z * 2,      # y = z * 2
        voxel_size_z           # z
    ])
    
    # 确保边界框完全覆盖模型
    min_bound = min_bound - voxel_size
    max_bound = max_bound + voxel_size
    
    # 计算每个维度的体素数量
    grid_dims = np.ceil((max_bound - min_bound) / voxel_size).astype(int)
    
    print(f"体素网格信息:")
    print(f"  体素尺寸: {voxel_size}")
    print(f"  网格维度: {grid_dims}")
    print(f"  总体素数: {np.prod(grid_dims)}")
    
    return min_bound, max_bound, tuple(grid_dims), voxel_size

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
    
    返回:
    intersection_grid: 形状为grid_dims的3D数组，标记每个体素是否与模型相交
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
        
        for i, origin in enumerate(origins):
            voxel_idx = voxel_indices[i]
            
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
        has_intersect = torch.any(valid_intersect, dim=1)
        
        # 获取相交的体素索引
        intersecting_rays = torch.where(has_intersect)[0]
        intersecting_voxel_indices = voxel_indices[intersecting_rays].unique().cpu().numpy()
        
        # 更新相交网格
        for idx in intersecting_voxel_indices:
            i = idx // (grid_dims[1] * grid_dims[2])
            j = (idx % (grid_dims[1] * grid_dims[2])) // grid_dims[2]
            k = idx % grid_dims[2]
            intersection_grid[i, j, k] = 1.0
        
        # 打印进度
        elapsed_time = time.time() - start_time
        print(f"处理进度： {batch_end}/{total_voxels} "
              f"({batch_end / total_voxels * 100:.1f}%) "
              f"- 已用时间: {elapsed_time:.2f}秒")
    
    # 统计相交体素数量
    num_intersecting = np.sum(intersection_grid > 0.5)
    print(f"找到 {num_intersecting} 个与模型相交的体素")
    print(f"总用时: {time.time() - start_time:.2f}秒")
    
    return intersection_grid

def visualize_with_polyscope_volume_grid(mesh, bound_low, bound_high, grid_dims, intersection_grid):
    """
    使用polyscope的register_volume_grid可视化模型和相交的体素
    
    参数:
    mesh: trimesh网格对象
    bound_low: 网格最小边界点
    bound_high: 网格最大边界点
    grid_dims: 网格维度 (nx, ny, nz)
    intersection_grid: 形状为grid_dims的3D数组，标记每个体素是否与模型相交
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
    mesh, texture = load_obj_and_texture(obj_path)
    
    # 创建体素网格
    bound_low, bound_high, grid_dims, voxel_size = create_voxel_grid(mesh, voxel_size_z=0.014)
    
    # 查找与模型相交的体素
    if use_gpu and torch.cuda.is_available():
        intersection_grid = find_intersecting_voxels_gpu(
            mesh, bound_low, bound_high, grid_dims, voxel_size, batch_size)
    else:
        if use_gpu and not torch.cuda.is_available():
            raise ValueError("请求使用GPU但没有可用的CUDA设备")
    
    # 使用polyscope的register_volume_grid可视化结果
    visualize_with_polyscope_volume_grid(mesh, bound_low, bound_high, grid_dims, intersection_grid)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='体素化3D模型并查找与模型相交的体素')
    parser.add_argument('--obj_path', type=str, default=None, help='OBJ文件路径')
    parser.add_argument('--cpu', action='store_true', help='强制使用CPU多线程而不是GPU')
    parser.add_argument('--batch-size', type=int, default=1000, help='GPU批处理大小')
    
    args = parser.parse_args()
    
    # 如果没有提供OBJ文件路径，使用默认路径
    if args.obj_path is None:
        obj_file = "/home/arno/Projects/Pint3D/print_ngp/Mesh2Volume/ExampleMesh/Minecraft_Grass_Block_OBJ/Grass_Block.obj"
    else:
        obj_file = args.obj_path
    
    main(obj_file, not args.cpu, args.batch_size)
