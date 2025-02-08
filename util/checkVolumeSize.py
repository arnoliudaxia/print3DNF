import numpy as np
import os

import sys

if len(sys.argv) < 2:
    print("Usage: python checkVolumeSize.py <directory>")
    sys.exit(1)

directory = sys.argv[1]
files = [f for f in os.listdir(directory) if f.endswith('.npy')]

for file in files:
    data = np.load(os.path.join(directory, file))

    z = 0.014
    y = z * 2
    x = y * 3

    print(f"File: {file} -> x: {data.shape[1] * x:.2f} mm, y: {data.shape[2] * y:.2f} mm, z: {data.shape[0] * z:.2f} mm")
