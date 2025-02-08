import numpy as np
import os
from tqdm import tqdm
import polyscope as ps
import polyscope.imgui as psim
import miniball


datapath = "/media/vrlab/rabbit/print3dingp/print_ngp_lyf/🐶NOMASK/ficus_d-1/volume/ngp_180/pred_rgbd"

# 获取目录中所有 .npy 文件
npy_files = sorted([os.path.join(datapath, f) for f in os.listdir(datapath) if f.endswith('.npy')])

# 检查是否有文件
if not npy_files:
    raise ValueError("No .npy files found in the directory!")

# 读取所有 .npy 文件并合并为一个张量
data_list = []
for file in tqdm(npy_files, desc="Loading .npy files", unit="file"):
    data = np.load(file)
    data_list.append(data)

# 将所有数组堆叠到一个大张量
big_tensor = np.stack(data_list, axis=0)  # 形状 (N, 273, 944, 4)，N 是文件数量

# 提取 r, g, b, alpha 通道
r = big_tensor[..., 0]  # 取第 1 通道
g = big_tensor[..., 1]  # 取第 2 通道
b = big_tensor[..., 2]  # 取第 3 通道
alpha = big_tensor[..., 3]  # 取第 4 通道

# 计算 alpha * (r + g + b)
pixel_values = alpha * (r + g + b)  # 保留原始形状

# 找到满足条件的索引
threshold = 0.4
indices = np.where(pixel_values > threshold)  # 返回满足条件的 (z, x, y) 坐标

# 组合三维坐标
z, x, y = indices
z = z.astype(np.float64) * 0.014
x = x.astype(np.float64) * 0.0846666
y = y.astype(np.float64) * 0.042333

coordinates = np.stack([z, x, y], axis=-1)  # 形状 (M, 3)，M 是满足条件的点的数量

# 输出结果
print(f"Number of coordinates: {coordinates.shape[0]}")


# 使用 miniball 计算最小外接球
mb = miniball.Miniball(coordinates)

# 获取球心和半径
center = mb.center()  # 球心
radius = np.sqrt(mb.squared_radius())  # 半径

print(f"Minimum Sphere Center: {center}")
print(f"Minimum Sphere Radius: {radius}")


# 可视化点云和最小外接球
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 50)
u, v = np.meshgrid(u, v)

x = radius * np.cos(u) * np.sin(v)
y = radius * np.sin(u) * np.sin(v)
z = radius * np.cos(v)

sphere_points = np.stack([x, y, z], axis=-1).reshape(-1, 3) + center

# 球心坐标
physical_center = np.array(center)  # 球心在物理空间中的位置

# 物理空间到图像空间的比例因子
z_factor = 1 / 0.014
x_factor = 1 / 0.0846666
y_factor = 1 / 0.042333

# 将球心从物理空间映射到图像空间
image_center = np.array([
    physical_center[0] * z_factor,
    physical_center[1] * x_factor,
    physical_center[2] * y_factor
])
# 四舍五入到最近的像素索引
image_center_index = np.round(image_center).astype(int)

# 确保索引在 `big_tensor` 的有效范围内
image_center_index[0] = np.clip(image_center_index[0], 0, big_tensor.shape[0] - 1)
image_center_index[1] = np.clip(image_center_index[1], 0, big_tensor.shape[1] - 1)
image_center_index[2] = np.clip(image_center_index[2], 0, big_tensor.shape[2] - 1)
print(f"球心对应的image stack index 为{image_center_index}")

# 获取最近点的值和索引
nearest_value = big_tensor[image_center_index[0], image_center_index[1], image_center_index[2]]
nearest_index = tuple(image_center_index)

# 输出结果
print(f"Physical Sphere Center: {physical_center}")
print(f"Image Center Index: {nearest_index}")
print(f"Value at Nearest Index: {nearest_value}")
# 初始化 Polyscope
# ps.init()
# # 注册点云数据
# ps.register_point_cloud("3D Points", coordinates, radius=0.005)


# ps.register_point_cloud("Bounding Sphere", sphere_points, radius=0.005)

# ps.show()

# # 椭球体参数
# ellipsoid_scale = [0.1, 0.1, 0.1]  # 椭球体的缩放参数
# ellipsoid_color = [1.0, 0.0, 0.0, 0.5]  # 椭球体颜色（RGBA）

# ui_x, ui_y ,ui_z= 0.5, 0.5, 0.5


# # 添加椭球体的 UI 控件和更新逻辑
# def update_ellipsoid(ellipsoid_position):
#     """更新和绘制椭球体的逻辑"""
#     # 生成椭球体的点云
#     u = np.linspace(0, 2 * np.pi, 50)
#     v = np.linspace(0, np.pi, 25)
#     u, v = np.meshgrid(u, v)
    
#     # 椭球体点云的参数化公式
#     x = ellipsoid_scale[0] * np.cos(u) * np.sin(v) + ellipsoid_position[0]
#     y = ellipsoid_scale[1] * np.sin(u) * np.sin(v) + ellipsoid_position[1]
#     z = ellipsoid_scale[2] * np.cos(v) + ellipsoid_position[2]

#     # 将点云展平成 (N, 3) 的形状
#     ellipsoid_points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1)
    
#     # 注册椭球体点云
#     ps.register_point_cloud("Ellipsoid", ellipsoid_points, radius=0.005, enabled=True, color=ellipsoid_color[:3], transparency=ellipsoid_color[3])

# # 添加 UI 控件
# def ellipsoid_ui_callback():
#     """在 Polyscope 界面中添加 UI 控件"""
#     global ui_x,ui_y,ui_z
    
#     Anychanged = False
#     changed, ui_x = psim.SliderFloat("Position-x", ui_x, v_min=-2, v_max=2)
#     Anychanged=Anychanged or changed
#     changed, ui_y = psim.SliderFloat("Position-y", ui_y, v_min=-2, v_max=2)
#     Anychanged=Anychanged or changed
#     changed, ui_z = psim.SliderFloat("Position-z", ui_z, v_min=-2, v_max=2)
#     Anychanged=Anychanged or changed

#     # changed |= psim.InputFloat3("Scale", ellipsoid_scale)
#     # changed |= psim.ColorEdit4("Color", ellipsoid_color)

#     # 如果参数发生变化，更新椭球体
#     if Anychanged:
#         update_ellipsoid((ui_x,ui_y,ui_z))

# # 注册 UI 回调
# ps.set_user_callback(ellipsoid_ui_callback)

# # 初次更新椭球体
# update_ellipsoid((ui_x,ui_y,ui_z))

# # 显示 Polyscope 窗口
# ps.show()
# ps.clear_user_callback()from tqdm import tqdm
import polyscope as ps
import polyscope.imgui as psim
import miniball


datapath = "/media/vrlab/rabbit/print3dingp/print_ngp_lyf/🐶NOMASK/ficus_d-1/volume/ngp_180/pred_rgbd"

# 获取目录中所有 .npy 文件
npy_files = sorted([os.path.join(datapath, f) for f in os.listdir(datapath) if f.endswith('.npy')])

# 检查是否有文件
if not npy_files:
    raise ValueError("No .npy files found in the directory!")

# 读取所有 .npy 文件并合并为一个张量
data_list = []
for file in tqdm(npy_files, desc="Loading .npy files", unit="file"):
    data = np.load(file)
    data_list.append(data)

# 将所有数组堆叠到一个大张量
big_tensor = np.stack(data_list, axis=0)  # 形状 (N, 273, 944, 4)，N 是文件数量

# 提取 r, g, b, alpha 通道
r = big_tensor[..., 0]  # 取第 1 通道
g = big_tensor[..., 1]  # 取第 2 通道
b = big_tensor[..., 2]  # 取第 3 通道
alpha = big_tensor[..., 3]  # 取第 4 通道

# 计算 alpha * (r + g + b)
pixel_values = alpha * (r + g + b)  # 保留原始形状

# 找到满足条件的索引
threshold = 0.4
indices = np.where(pixel_values > threshold)  # 返回满足条件的 (z, x, y) 坐标

# 组合三维坐标
z, x, y = indices
z = z.astype(np.float64) * 0.014
x = x.astype(np.float64) * 0.0846666
y = y.astype(np.float64) * 0.042333

coordinates = np.stack([z, x, y], axis=-1)  # 形状 (M, 3)，M 是满足条件的点的数量

# 输出结果
print(f"Number of coordinates: {coordinates.shape[0]}")


# 使用 miniball 计算最小外接球
mb = miniball.Miniball(coordinates)

# 获取球心和半径
center = mb.center()  # 球心
radius = np.sqrt(mb.squared_radius())  # 半径

print(f"Minimum Sphere Center: {center}")
print(f"Minimum Sphere Radius: {radius}")


# 可视化点云和最小外接球
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 50)
u, v = np.meshgrid(u, v)

x = radius * np.cos(u) * np.sin(v)
y = radius * np.sin(u) * np.sin(v)
z = radius * np.cos(v)

sphere_points = np.stack([x, y, z], axis=-1).reshape(-1, 3) + center

# 球心坐标
physical_center = np.array(center)  # 球心在物理空间中的位置

# 物理空间到图像空间的比例因子
z_factor = 1 / 0.014
x_factor = 1 / 0.0846666
y_factor = 1 / 0.042333

# 将球心从物理空间映射到图像空间
image_center = np.array([
    physical_center[0] * z_factor,
    physical_center[1] * x_factor,
    physical_center[2] * y_factor
])
# 四舍五入到最近的像素索引
image_center_index = np.round(image_center).astype(int)

# 确保索引在 `big_tensor` 的有效范围内
image_center_index[0] = np.clip(image_center_index[0], 0, big_tensor.shape[0] - 1)
image_center_index[1] = np.clip(image_center_index[1], 0, big_tensor.shape[1] - 1)
image_center_index[2] = np.clip(image_center_index[2], 0, big_tensor.shape[2] - 1)
print(f"球心对应的image stack index 为{image_center_index}")

# 获取最近点的值和索引
nearest_value = big_tensor[image_center_index[0], image_center_index[1], image_center_index[2]]
nearest_index = tuple(image_center_index)

# 输出结果
print(f"Physical Sphere Center: {physical_center}")
print(f"Image Center Index: {nearest_index}")
print(f"Value at Nearest Index: {nearest_value}")
# 初始化 Polyscope
# ps.init()
# # 注册点云数据
# ps.register_point_cloud("3D Points", coordinates, radius=0.005)


# ps.register_point_cloud("Bounding Sphere", sphere_points, radius=0.005)

# ps.show()

# # 椭球体参数
# ellipsoid_scale = [0.1, 0.1, 0.1]  # 椭球体的缩放参数
# ellipsoid_color = [1.0, 0.0, 0.0, 0.5]  # 椭球体颜色（RGBA）

# ui_x, ui_y ,ui_z= 0.5, 0.5, 0.5


# # 添加椭球体的 UI 控件和更新逻辑
# def update_ellipsoid(ellipsoid_position):
#     """更新和绘制椭球体的逻辑"""
#     # 生成椭球体的点云
#     u = np.linspace(0, 2 * np.pi, 50)
#     v = np.linspace(0, np.pi, 25)
#     u, v = np.meshgrid(u, v)
    
#     # 椭球体点云的参数化公式
#     x = ellipsoid_scale[0] * np.cos(u) * np.sin(v) + ellipsoid_position[0]
#     y = ellipsoid_scale[1] * np.sin(u) * np.sin(v) + ellipsoid_position[1]
#     z = ellipsoid_scale[2] * np.cos(v) + ellipsoid_position[2]

#     # 将点云展平成 (N, 3) 的形状
#     ellipsoid_points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1)
    
#     # 注册椭球体点云
#     ps.register_point_cloud("Ellipsoid", ellipsoid_points, radius=0.005, enabled=True, color=ellipsoid_color[:3], transparency=ellipsoid_color[3])

# # 添加 UI 控件
# def ellipsoid_ui_callback():
#     """在 Polyscope 界面中添加 UI 控件"""
#     global ui_x,ui_y,ui_z
    
#     Anychanged = False
#     changed, ui_x = psim.SliderFloat("Position-x", ui_x, v_min=-2, v_max=2)
#     Anychanged=Anychanged or changed
#     changed, ui_y = psim.SliderFloat("Position-y", ui_y, v_min=-2, v_max=2)
#     Anychanged=Anychanged or changed
#     changed, ui_z = psim.SliderFloat("Position-z", ui_z, v_min=-2, v_max=2)
#     Anychanged=Anychanged or changed

#     # changed |= psim.InputFloat3("Scale", ellipsoid_scale)
#     # changed |= psim.ColorEdit4("Color", ellipsoid_color)

#     # 如果参数发生变化，更新椭球体
#     if Anychanged:
#         update_ellipsoid((ui_x,ui_y,ui_z))

# # 注册 UI 回调
# ps.set_user_callback(ellipsoid_ui_callback)

# # 初次更新椭球体
# update_ellipsoid((ui_x,ui_y,ui_z))

# # 显示 Polyscope 窗口
# ps.show()
# ps.clear_user_callback()from tqdm import tqdm
import polyscope as ps
import polyscope.imgui as psim
import miniball


datapath = "/media/vrlab/rabbit/print3dingp/print_ngp_lyf/🐶NOMASK/ficus_d-1/volume/ngp_180/pred_rgbd"

# 获取目录中所有 .npy 文件
npy_files = sorted([os.path.join(datapath, f) for f in os.listdir(datapath) if f.endswith('.npy')])

# 检查是否有文件
if not npy_files:
    raise ValueError("No .npy files found in the directory!")

# 读取所有 .npy 文件并合并为一个张量
data_list = []
for file in tqdm(npy_files, desc="Loading .npy files", unit="file"):
    data = np.load(file)
    data_list.append(data)

# 将所有数组堆叠到一个大张量
big_tensor = np.stack(data_list, axis=0)  # 形状 (N, 273, 944, 4)，N 是文件数量

# 提取 r, g, b, alpha 通道
r = big_tensor[..., 0]  # 取第 1 通道
g = big_tensor[..., 1]  # 取第 2 通道
b = big_tensor[..., 2]  # 取第 3 通道
alpha = big_tensor[..., 3]  # 取第 4 通道

# 计算 alpha * (r + g + b)
pixel_values = alpha * (r + g + b)  # 保留原始形状

# 找到满足条件的索引
threshold = 0.4
indices = np.where(pixel_values > threshold)  # 返回满足条件的 (z, x, y) 坐标

# 组合三维坐标
z, x, y = indices
z = z.astype(np.float64) * 0.014
x = x.astype(np.float64) * 0.0846666
y = y.astype(np.float64) * 0.042333

coordinates = np.stack([z, x, y], axis=-1)  # 形状 (M, 3)，M 是满足条件的点的数量

# 输出结果
print(f"Number of coordinates: {coordinates.shape[0]}")


# 使用 miniball 计算最小外接球
mb = miniball.Miniball(coordinates)

# 获取球心和半径
center = mb.center()  # 球心
radius = np.sqrt(mb.squared_radius())  # 半径

print(f"Minimum Sphere Center: {center}")
print(f"Minimum Sphere Radius: {radius}")


# 可视化点云和最小外接球
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 50)
u, v = np.meshgrid(u, v)

x = radius * np.cos(u) * np.sin(v)
y = radius * np.sin(u) * np.sin(v)
z = radius * np.cos(v)

sphere_points = np.stack([x, y, z], axis=-1).reshape(-1, 3) + center

# 球心坐标
physical_center = np.array(center)  # 球心在物理空间中的位置

# 物理空间到图像空间的比例因子
z_factor = 1 / 0.014
x_factor = 1 / 0.0846666
y_factor = 1 / 0.042333

# 将球心从物理空间映射到图像空间
image_center = np.array([
    physical_center[0] * z_factor,
    physical_center[1] * x_factor,
    physical_center[2] * y_factor
])
# 四舍五入到最近的像素索引
image_center_index = np.round(image_center).astype(int)

# 确保索引在 `big_tensor` 的有效范围内
image_center_index[0] = np.clip(image_center_index[0], 0, big_tensor.shape[0] - 1)
image_center_index[1] = np.clip(image_center_index[1], 0, big_tensor.shape[1] - 1)
image_center_index[2] = np.clip(image_center_index[2], 0, big_tensor.shape[2] - 1)
print(f"球心对应的image stack index 为{image_center_index}")

# 获取最近点的值和索引
nearest_value = big_tensor[image_center_index[0], image_center_index[1], image_center_index[2]]
nearest_index = tuple(image_center_index)

# 输出结果
print(f"Physical Sphere Center: {physical_center}")
print(f"Image Center Index: {nearest_index}")
print(f"Value at Nearest Index: {nearest_value}")
# 初始化 Polyscope
# ps.init()
# # 注册点云数据
# ps.register_point_cloud("3D Points", coordinates, radius=0.005)


# ps.register_point_cloud("Bounding Sphere", sphere_points, radius=0.005)

# ps.show()

# # 椭球体参数
# ellipsoid_scale = [0.1, 0.1, 0.1]  # 椭球体的缩放参数
# ellipsoid_color = [1.0, 0.0, 0.0, 0.5]  # 椭球体颜色（RGBA）

# ui_x, ui_y ,ui_z= 0.5, 0.5, 0.5


# # 添加椭球体的 UI 控件和更新逻辑
# def update_ellipsoid(ellipsoid_position):
#     """更新和绘制椭球体的逻辑"""
#     # 生成椭球体的点云
#     u = np.linspace(0, 2 * np.pi, 50)
#     v = np.linspace(0, np.pi, 25)
#     u, v = np.meshgrid(u, v)
    
#     # 椭球体点云的参数化公式
#     x = ellipsoid_scale[0] * np.cos(u) * np.sin(v) + ellipsoid_position[0]
#     y = ellipsoid_scale[1] * np.sin(u) * np.sin(v) + ellipsoid_position[1]
#     z = ellipsoid_scale[2] * np.cos(v) + ellipsoid_position[2]

#     # 将点云展平成 (N, 3) 的形状
#     ellipsoid_points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1)
    
#     # 注册椭球体点云
#     ps.register_point_cloud("Ellipsoid", ellipsoid_points, radius=0.005, enabled=True, color=ellipsoid_color[:3], transparency=ellipsoid_color[3])

# # 添加 UI 控件
# def ellipsoid_ui_callback():
#     """在 Polyscope 界面中添加 UI 控件"""
#     global ui_x,ui_y,ui_z
    
#     Anychanged = False
#     changed, ui_x = psim.SliderFloat("Position-x", ui_x, v_min=-2, v_max=2)
#     Anychanged=Anychanged or changed
#     changed, ui_y = psim.SliderFloat("Position-y", ui_y, v_min=-2, v_max=2)
#     Anychanged=Anychanged or changed
#     changed, ui_z = psim.SliderFloat("Position-z", ui_z, v_min=-2, v_max=2)
#     Anychanged=Anychanged or changed

#     # changed |= psim.InputFloat3("Scale", ellipsoid_scale)
#     # changed |= psim.ColorEdit4("Color", ellipsoid_color)

#     # 如果参数发生变化，更新椭球体
#     if Anychanged:
#         update_ellipsoid((ui_x,ui_y,ui_z))

# # 注册 UI 回调
# ps.set_user_callback(ellipsoid_ui_callback)

# # 初次更新椭球体
# update_ellipsoid((ui_x,ui_y,ui_z))

# # 显示 Polyscope 窗口
# ps.show()
# ps.clear_user_callback()