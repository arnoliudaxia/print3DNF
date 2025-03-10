import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import subprocess

folderPath = "🐶VDB/"
directories = [d for d in os.listdir(folderPath) if os.path.isdir(os.path.join(folderPath, d))]

for dir in tqdm(directories):
    if "✅"  in dir:
        continue
    # if "typical_building_building" != dir:
    #     continue
    # if Path(os.path.join(folderPath, dir , "preview-view1.mp4") ).exists():
    #     continue
    print(dir)
    datapath = os.path.join(folderPath, dir , "array/allData.npy") 
    big_tensor =np.load(datapath) # 形状 (N, 273, 944, 4)，N 是文件数量
    print(big_tensor.shape)



    # 提取 r, g, b, alpha 通道
    r = big_tensor[..., 0]  # 取第 1 通道
    g = big_tensor[..., 1]  # 取第 2 通道
    b = big_tensor[..., 2]  # 取第 3 通道
    alpha = big_tensor[..., 3]  # 取第 4 通道


    # 假设 alpha 是一个形状为 (depth, height, width) 的3D数组
    # 初始化3D bbox的范围，使用极大值和极小值确保首次比较能正确更新
    x_min, y_min, z_min = float('inf'), float('inf'), float('inf')
    x_max, y_max, z_max = -float('inf'), -float('inf'), -float('inf')

    for i in range(len(alpha)):
        coordinates = np.where(alpha[i] > .8)
        if coordinates[0].size > 0:
            # 计算当前层的bbox
            current_bbox = [
                coordinates[0].min(), # x_min
                coordinates[1].min(), # y_min
                i,                    # z_min for the current layer
                coordinates[0].max(), # x_max
                coordinates[1].max(), # y_max
                i                     # z_max for the current layer
            ]
            
            # 更新整体3D bbox
            x_min = min(x_min, current_bbox[0])
            y_min = min(y_min, current_bbox[1])
            z_min = min(z_min, current_bbox[2])
            x_max = max(x_max, current_bbox[3])
            y_max = max(y_max, current_bbox[4])
            z_max = max(z_max, current_bbox[5])

    # 输出最终的3D bbox
    if x_min != float('inf') and x_max != -float('inf'):
        print("3D Bounding Box: [x_min={}, y_min={}, z_min={}, x_max={}, y_max={}, z_max={}]".format(x_min, y_min, z_min, x_max, y_max, z_max))
    else:
        print("No elements satisfy the condition across all layers.")
        
    #按照3D bbox裁剪big_tensor
    big_tensor = big_tensor[z_min:z_max+1 , x_min:x_max+1 , y_min:y_max+1]
        
        
    expandRatio=.3
    xExpand=int((x_max-x_min)*expandRatio/2)
    yExpand = int((y_max - y_min) * expandRatio)
    zExpand = int((z_max - z_min) * expandRatio)

    # 计算扩展后的张量形状
    expanded_shape = (
        big_tensor.shape[0] + zExpand,  # z轴的扩展
        big_tensor.shape[1] + xExpand,  
        big_tensor.shape[2] + yExpand , 
        4
    )

    # 创建一个形状为expanded_shape的全零张量
    big_tensor_expanded = np.zeros(expanded_shape)

    # 计算原始big_tensor在扩展张量中的位置
    z_offset = zExpand // 2
    y_offset = yExpand // 2
    x_offset = xExpand // 2

    # 将原始big_tensor复制到big_tensor_expanded的中心
    # big_tensor_expanded[z_offset:z_offset + big_tensor.shape[0],
    #                      y_offset:y_offset + big_tensor.shape[1],
    #                      x_offset:x_offset + big_tensor.shape[2]] = big_tensor

    big_tensor_expanded[z_offset:z_offset + big_tensor.shape[0],
                        0: big_tensor.shape[1],
                        y_offset:y_offset+ big_tensor.shape[2]] = big_tensor

    # 打印结果形状确认
    print("Expanded tensor shape:", big_tensor_expanded.shape)

    # 取出datapath的目录
    datapath = os.path.dirname(datapath)
    datapath=os.path.join(datapath,"..", "cut")
    os.makedirs(datapath, exist_ok=True)
    np.save(os.path.join(datapath, "allData.npy"), big_tensor_expanded)

    # os.system(f"python /media/vrlab/rabbit/print3dingp/print_volume/preview_volume.py --input_folder {datapath} --onlyOneView ")

    subprocess.run(
        ["python", "/media/vrlab/rabbit/print3dingp/print_volume/preview_volume.py", "--input_folder", datapath, "--onlyOneView"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )