import numpy as np
import os

import sys

if len(sys.argv) < 2:
    print("Usage: python checkVolumeSize.py <directory>")
    sys.exit(1)

path = sys.argv[1]

if os.path.isdir(path):
    files = [f for f in os.listdir(path) if f.endswith('.npy')]
elif os.path.isfile(path):
    files = [path]
else:
    print("Invalid path")
    sys.exit(1)


for file in files:
    data = np.load(os.path.join(path, file))

    z = 0.014
    y = z * 2
    x = y * 3

    print(f"File: {file} -> x: {data.shape[1] * x:.2f} mm, y: {data.shape[2] * y:.2f} mm, z: {data.shape[0] * z:.2f} mm")
