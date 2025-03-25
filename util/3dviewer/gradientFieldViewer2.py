import numpy as np
import pyvista as pv
import argparse
from tqdm import tqdm

# 解析命令行参数
parser = argparse.ArgumentParser(description='使用PyVista显示体积数据的梯度场')
parser.add_argument('--volume_path', type=str, help='体积数据的路径(.npy文件)')
parser.add_argument('--sliceIndex', type=int, help='切片索引', default=-1)
parser.add_argument('--stride', type=int, help='采样步长', default=5)
parser.add_argument('--scale', type=float, help='向量缩放因子', default=0.5)
parser.add_argument('--threshold', type=float, help='梯度大小阈值', default=0.01)
args = parser.parse_args()

# 加载体积数据
print("加载体积数据...")
volume = np.load(args.volume_path)

# 处理切片
if args.sliceIndex != -1:
    print(f"提取切片 {args.sliceIndex} 到 {args.sliceIndex+100}...")
    volume = volume[:,:,args.sliceIndex:args.sliceIndex+100, :]

# 设置物理尺度
z_scale, y_scale, x_scale = 0.014, 0.0846666, 0.042333
density = volume[..., 3]

# 计算梯度场
print("计算梯度场...")
def compute_gradient_field(density):
    # 使用numpy的gradient函数计算三个方向的梯度
    # 考虑物理尺度
    dz, dy, dx = np.gradient(density, z_scale, y_scale, x_scale)
    return np.stack([dx, dy, dz], axis=-1)

gradient_field = compute_gradient_field(density)

# 创建结构化网格 - 使用正确的方法创建网格
print("创建结构化网格...")
# 使用 pv.ImageData 替代 UniformGrid
grid = pv.ImageData()
grid.dimensions = np.array(density.shape) + 1
grid.spacing = (x_scale, y_scale, z_scale)
grid.origin = (0, 0, 0)

# 将密度数据添加到网格
grid.cell_data["density"] = density.flatten(order='F')

# 计算梯度场的大小
gradient_magnitude = np.linalg.norm(gradient_field, axis=-1)
grid.cell_data["gradient_magnitude"] = gradient_magnitude.flatten(order='F')

# 创建用于显示的点和向量
print("准备向量场数据...")
z_coords, y_coords, x_coords = np.meshgrid(
    np.arange(density.shape[0]) * z_scale,
    np.arange(density.shape[1]) * y_scale,
    np.arange(density.shape[2]) * x_scale,
    indexing='ij'
)

# 使用步长对数据进行采样，减少显示的向量数量
stride = args.stride
mask = gradient_magnitude > args.threshold  # 只显示大于阈值的梯度

# 安全地应用掩码和步长
sampled_mask = mask[::stride, ::stride, ::stride]
sampled_x = x_coords[::stride, ::stride, ::stride]
sampled_y = y_coords[::stride, ::stride, ::stride]
sampled_z = z_coords[::stride, ::stride, ::stride]
sampled_vectors = gradient_field[::stride, ::stride, ::stride]

# 提取满足条件的点和向量
points = np.vstack([
    sampled_x[sampled_mask].flatten(),
    sampled_y[sampled_mask].flatten(),
    sampled_z[sampled_mask].flatten()
]).T

vectors = sampled_vectors[sampled_mask]

# 创建 PyVista 绘图器
print("创建可视化...")
p = pv.Plotter()

# 添加体积渲染
p.add_volume(grid, cmap="viridis", opacity="linear", clim=[0, np.max(density)])

# 添加梯度场 (如果有足够的点)
if len(points) > 0:
    p.add_arrows(points, vectors, scale=args.scale)
else:
    print("警告: 没有满足阈值条件的梯度向量")

# 添加三个正交切片
p.add_mesh(grid.slice_orthogonal(), opacity=0.5)

# 添加等值面
if np.max(density) > 0:
    contour_value = np.max(density) * 0.5
    contours = grid.contour([contour_value])
    p.add_mesh(contours, color="white", opacity=0.3)

# 添加坐标轴
p.add_axes()
p.add_bounding_box()

# 设置相机位置
p.camera_position = 'xy'
p.camera.zoom(1.5)

# 显示可视化窗口
print("显示可视化窗口...")
p.show()
