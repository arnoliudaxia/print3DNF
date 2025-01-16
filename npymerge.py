#


import os
import numpy as np
from tqdm import tqdm

def stack_npy_files(input_directory, output_file):
    """
    Reads all .npy files in the specified directory, stacks them, 
    and saves the result as a single .npy file.

    Parameters:
        input_directory (str): Path to the directory containing .npy files.
        output_file (str): Path to the output .npy file.
    """
    npy_files = [f for f in os.listdir(input_directory) if f.endswith('.npy')]
    if not npy_files:
        print("No .npy files found in the directory.")
        return

    stacked_array = []

    for file in tqdm(npy_files):
        file_path = os.path.join(input_directory, file)
        try:
            data = np.load(file_path)
            stacked_array.append(data.astype(np.float16))
        except Exception as e:
            print(f"Error loading {file}: {e}")

    # Stack all loaded arrays along a new axis
    result = np.stack(stacked_array)

    # Save the stacked array to the output file
    np.save(output_file, result)
    print(f"Stacked array saved to {output_file}")

# Example usage
input_directory = "/media/vrlab/rabbit/print3dingp/print_ngp_lyf/🐶NOMASK/ficus_d-1/volume/ngp_180/array/"
output_file = f"{input_directory}/allData.npy"
stack_npy_files(input_directory, output_file)
