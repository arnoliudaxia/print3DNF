import numpy as np
import os
import sys

if len(sys.argv) < 2:
    print("Usage: python checkVolumeSize.py <directory>")
    sys.exit(1)

path = sys.argv[1]

if os.path.isdir(path):
    files = [f for f in os.listdir(path) if f.endswith('.npy') or f.endswith('.npz')]
elif os.path.isfile(path):
    files = [os.path.basename(path)]
    path = os.path.dirname(path)
    if not path:
        path = '.'
else:
    print("Invalid path")
    sys.exit(1)

for file in files:
    file_path = os.path.join(path, file)
    
    # 根据文件扩展名选择不同的加载方式
    if file.endswith('.npz'):
        data = np.load(file_path)['volume']
    else:
        data = np.load(file_path)


    x,y,z=[0.0846666, 0.042333, 0.014]
    

    print(f"File: {file} -> x: {data.shape[1] * x:.2f} mm, y: {data.shape[2] * y:.2f} mm, z: {data.shape[0] * z:.2f} mm")
