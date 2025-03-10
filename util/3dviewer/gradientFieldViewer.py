import numpy as np
from tqdm import tqdm
import polyscope as ps

volume=np.load("/home/arno/Projects/Pint3D/print_data/fern/volume/ngp_471/array/allData-onlytree.npy" )
# volume=np.load("/home/arno/Projects/Pint3D/print_data/test_slice/array/allData.npy" )
z_scale, y_scale, x_scale = 0.014, 0.0846666, 0.042333
density = volume[..., 3]
# 计算梯度场
def compute_gradient_field(density):
    # 使用numpy的gradient函数计算三个方向的梯度
    # 注意：需要考虑实际物理尺度
    # dz, dy, dx = np.gradient(density, z_scale, y_scale, x_scale)
    dx, dy, dz = np.gradient(density)
    return np.stack([dz, dy, dx], axis=-1)

density_field=compute_gradient_field(density)
breakpoint()
# 创建用于可视化的点云数据
z_coords, y_coords, x_coords = np.meshgrid(np.arange(density.shape[0]) * z_scale,
                                          np.arange(density.shape[1]) * y_scale,
                                          np.arange(density.shape[2]) * x_scale,
                                          indexing='ij')
points = np.stack([x_coords.flatten(), y_coords.flatten(), z_coords.flatten()], axis=1)
vectors = density_field.reshape(-1, 3)

# 初始化 polyscope
ps.init()

# 注册点云和向量场
ps_cloud = ps.register_point_cloud("gradient_field", points)
ps_cloud.add_vector_quantity("gradients", vectors, enabled=True)

# 显示可视化窗口
ps.show()
