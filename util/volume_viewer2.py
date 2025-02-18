import numpy as np
from PIL import Image
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='View 3D volume from stack of PNG images')
    parser.add_argument('--image_dir', type=str, help='Directory containing PNG image files')
    parser.add_argument('--x-scale', type=float, default=0.042333, help='Scale factor for width (default: 1.0)')
    parser.add_argument('--y-scale', type=float, default=0.0846666, help='Scale factor for height (default: 1.0)')
    parser.add_argument('--z-scale', type=float, default=0.014, help='Scale factor for layer spacing (default: 1.0)')
    parser.add_argument('--rgba', action='store_true', help='Use RGBA mode instead of grayscale')
    parser.add_argument('--transparent', action='store_true', help='Allow fully transparent pixels (alpha=0) to be invisible')
    parser.add_argument('--render', action='store_true', help='Render a high-resolution image with transparent background')
    
    parser.add_argument('--channelWise', "-c", type=str, help="只看某一个通道，取值[White, Red, Green, Blue]，只在预览print slice时有效")
    parser.add_argument('--presetScenen', "-p", type=str, help="单个场景的预设配置，有[fern]")
    
    args = parser.parse_args()
    

    DataType=None
    if args.image_dir.endswith('.npy'):
        volume = np.load(args.image_dir)
        DataType = "RGB"
        # volume是BGRA，需要转换为RGBA
        volume = volume[..., [2, 1, 0, 3]] 
    else:
        DataType = "half"
        image_files = sorted([f for f in os.listdir(args.image_dir) if f.endswith('.png')])
        if not image_files:
            raise ValueError("No PNG images found in the directory")
        # Load first image to get dimensions
        first_image = Image.open(os.path.join(args.image_dir, image_files[0]))
        width, height = first_image.size
        depth = len(image_files)
        # 构造一个3D的array
        volume = np.zeros((depth, height, width, 4), dtype=np.uint8)
        for i, img_file in enumerate(image_files):
            img = Image.open(os.path.join(args.image_dir, img_file)).convert('RGBA')
            volume[i] = np.array(img)
    
    if args.presetScenen=="fern":
        volume=volume[-800:-1,...]
        
    # 将volume的索引转换为点云坐标和颜色
    z_scale, y_scale, x_scale = 0.014, 0.0846666, 0.042333
    indices = np.argwhere(volume[..., 3] > 0)  # 找到所有非透明像素的索引
    point_cloud_coords = indices * np.array([z_scale, y_scale, x_scale])
    point_cloud_rgbd = volume[indices[:, 0], indices[:, 1], indices[:, 2], :] # 归一化颜色值
    if DataType == "RGB":
        point_cloud_transpraency= point_cloud_rgbd[:, 3]
    
    if DataType != "half" and args.channelWise:
        print("WARN: channelWise 只在预览print slice时有效，将忽略这个参数")
    else:
        if args.channelWise == "White":
            # 筛选出 rgbd 为 [255, 255, 255, 255] 的点
            white_points_mask = np.all(point_cloud_rgbd == [255 , 255, 255, 255], axis=1)
            point_cloud_coords = point_cloud_coords[white_points_mask]
            point_cloud_rgbd = point_cloud_rgbd[white_points_mask]

    point_cloud_colors=point_cloud_rgbd[:, :3]
    
    import polyscope as ps

    ps.set_ground_plane_mode("none")  # 将地面设置成None
    ps.set_background_color((0, 0, 0))  # 设置背景为黑色
    ps.set_up_dir("z_up")  # 设置up direction 为 z UP
    # Initialize polyscope
    ps.init()
    # `my_points` is a Nx3 numpy array
    ps_cloud = ps.register_point_cloud("my points", point_cloud_coords, point_render_mode='quad', radius=0.00033) # 14比较贴合实际
    if DataType == "RGB":
        ps_cloud.add_scalar_quantity("trans", point_cloud_transpraency)
        ps_cloud.set_transparency_quantity("trans")
    ps_cloud.add_color_quantity("colors", point_cloud_colors)
      
    # View the point cloud and mesh we just registered in the 3D UI
    ps.show()
    
    
if __name__ == "__main__":
    main() 