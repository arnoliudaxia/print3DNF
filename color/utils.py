import pandas as pd
import numpy as np
import colour
import cv2
from tqdm import tqdm
import os
from scipy.optimize import fsolve, least_squares
import json

K = {}
S = {}
x = 1
z = 0.014

rgb_to_concentration_map_file = 'color/data/color_map/10.csv'
rgb_to_concentration_map = pd.read_csv(rgb_to_concentration_map_file)
rgb_to_concentration_step = 1 / (len(rgb_to_concentration_map)**(1/3) - 1)
rgb_to_concentration_interpolate = round(len(rgb_to_concentration_map)**(1/3) - 1)

brightness_to_kw_concentration_map_file = 'color/data/color_map/brightness_kw_rates.csv'
brightness_to_kw_concentration_map = pd.read_csv(brightness_to_kw_concentration_map_file)
brightness_to_kw_step = 1 / (len(brightness_to_kw_concentration_map) - 1)
brightness_to_kw_interpolate = round(len(brightness_to_kw_concentration_map) - 1)

cmykwe_density_rgb = np.array(json.load(open('color/data/color_calib.json'))['density_rgb'])


for color in ['c', 'm', 'y', 'k', 'w']:
    data = pd.read_csv(f'./color/data/calib_data/{color}.csv')
    if color == 'c':
        wavelength = data['Wavelength'].values
    K[color], S[color] = data['K'].values, data['S'].values



def KMmodel(K, S, x):
    a = (S + K) / S
    b = np.sqrt(a**2 - 1)
    sinh_term = np.sinh(b * S * x)
    cosh_term = np.cosh(b * S * x)

    
    R = sinh_term / (a * sinh_term + b * cosh_term)
    T = b / (a * sinh_term + b * cosh_term)
    
    R[np.isnan(R)] = 0
    T[np.isnan(T)] = 0
    return R, T

# def KMmodel_vectorized(K, S, x):
#     a = (S + K) / S
#     b = np.sqrt(a**2 - 1)
#     sinh_term = np.sinh(b * S * x)
#     cosh_term = np.cosh(b * S * x)
    
#     R = sinh_term / (a * sinh_term + b * cosh_term)
#     T = b / (a * sinh_term + b * cosh_term)
    
#     # 向量化处理，确保在第0维度添加一个维度
#     R = np.where(np.isnan(R), 1, R)
#     T = np.where(np.isnan(T), 0, T)
    
#     return R, T

def calculate_rgb(wavelengths, spectrum):
    """
    Calculate RGB values from spectrum data using the colour package
    """
    # Create a SpectralDistribution object
    spd = colour.SpectralDistribution(
        dict(zip(wavelengths, spectrum)), 
        name="Sample SPD"
    )
    
    # Get CIE 1931 2-degree standard observer
    cmfs = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
    
    # Use D65 illuminant
    illuminant = colour.SDS_ILLUMINANTS['D65']
    
    # Convert to XYZ using explicit parameters and integration method
    XYZ = colour.sd_to_XYZ(
        spd,
        cmfs=cmfs,
        illuminant=illuminant,
        method='Integration',
        shape=colour.SpectralShape(300, 800, 1)  # Specify our wavelength range
    )
    
    # Convert XYZ to sRGB
    rgb = colour.XYZ_to_sRGB(XYZ / 100.0)  # Divide by 100 to convert from percentage
    
    # Ensure values are between 0 and 1
    rgb = np.clip(rgb, 0, 1)
    
    return rgb

# def calculate_rgb_vectorized(wavelengths, spectra):
#     """
#     Calculate RGB values from multiple spectrum data using the colour package (fully vectorized).

#     Parameters:
#     - wavelengths: 1D array-like, the wavelengths of the spectrum.
#     - spectra: 2D array-like, each row represents the spectrum values for a sample.

#     Returns:
#     - rgb_values: 2D array, each row contains the RGB values for a sample.
#     """
#     # Create a SpectralShape based on wavelengths
#     spectral_shape = colour.SpectralShape(min(wavelengths), max(wavelengths), wavelengths[1] - wavelengths[0])
    
#     # Get CIE 1931 2-degree standard observer
#     cmfs = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer'].copy()
#     cmfs = cmfs.interpolate(spectral_shape)
#     cmfs = cmfs.extrapolate(spectral_shape)
    
#     # Use D65 illuminant, extrapolated and interpolated to match the wavelengths
#     illuminant = colour.SDS_ILLUMINANTS['D65'].copy()
#     illuminant = illuminant.interpolate(spectral_shape)
#     illuminant = illuminant.extrapolate(spectral_shape)
    
#     # Normalize illuminant to 100
#     illuminant_values = illuminant.values
#     illuminant_values /= np.max(illuminant_values)  # 先让最大值 = 1
#     # illuminant_values *= 100                        # 再让最大值 = 100
    
#     # Precompute scaling factors for integration
#     delta_lambda = wavelengths[1] - wavelengths[0]
    
#     # Compute XYZ values for all spectra (matrix operations)
#     spectra = np.array(spectra)
#     cmfs_values = cmfs.values # Shape: (len(wavelengths), 3)
    
#     # Apply the illuminant to the spectra
#     spectra_weighted = spectra * illuminant_values[np.newaxis, :]
    
#     # Integrate to compute XYZ values
#     XYZ = delta_lambda * np.dot(spectra_weighted, cmfs_values)
    
#     # Convert XYZ to sRGB
#     RGB = colour.XYZ_to_sRGB(XYZ / 100.0)  # Divide by 100 to normalize to percentage
    
#     # Clip values to [0, 1] range
#     # RGB = np.clip(RGB, 0, 1)
    
#     return RGB


def concentration_to_rgbd(concentration):
    '''
    concentration: [c, m, y, k, w, clear]
    '''
    concentration = concentration / concentration.sum()
    K_blend = sum(concentration[j] * K[color] for j, color in enumerate(['c', 'm', 'y', 'k', 'w']))
    S_blend = sum(concentration[j] * S[color] for j, color in enumerate(['c', 'm', 'y', 'k', 'w']))
    R_blend, T_blend = KMmodel(K_blend, S_blend, x)
    C_blend = R_blend + T_blend
    C_rgb = calculate_rgb(wavelength, C_blend)

    _, T_alpha = KMmodel(K_blend, S_blend, z)
    T_alpha_rgb = calculate_rgb(wavelength, T_alpha)
    T_alpha = T_alpha_rgb.mean()
    density = -np.log(T_alpha) / z

    density = density * (1 - concentration[5])

    return np.concatenate([C_rgb, [density]])

# def concentration_to_rgbd_image(concentration):
#     '''
#     concentration: [c, m, y, k, w, clear]
#     '''
#     concentration_shape = concentration.shape
#     concentration = concentration / concentration.sum(axis=2)[:, :, np.newaxis]
#     concentration = concentration.reshape(-1, 6)
    
#     K_blend = sum(concentration[:, j].reshape(-1, 1) * K[color].reshape(1, -1) for j, color in enumerate(['c', 'm', 'y', 'k', 'w']))
#     S_blend = sum(concentration[:, j].reshape(-1, 1) * S[color].reshape(1, -1) for j, color in enumerate(['c', 'm', 'y', 'k', 'w']))
#     R_blend, T_blend = KMmodel_vectorized(K_blend, S_blend, x)
#     C_blend = R_blend + T_blend
#     print(C_blend.shape);exit()
#     C_rgb = calculate_rgb_vectorized(wavelength, C_blend)

#     _, T_alpha = KMmodel_vectorized(K_blend, S_blend, z)
#     T_alpha_rgb = calculate_rgb_vectorized(wavelength, T_alpha)
#     print(C_rgb.shape, T_alpha_rgb.shape);exit()
#     T_alpha = T_alpha_rgb.mean()
#     density = -np.log(T_alpha) / z

#     density = density * (1 - concentration[5])

#     return np.concatenate([C_rgb, [density]])

def rgb_to_concentration_least_square(target_rgb):
    def objective_function(concentration):
        # 计算当前浓度下的 RGB
        concentration = np.array([concentration[0], concentration[1], concentration[2], concentration[3], concentration[4], 0])
        concentration = concentration / concentration.sum()
        predicted_rgb = concentration_to_rgbd(concentration)[:3]  # 只取 RGB 部分
        # 返回目标 RGB 和预测 RGB 之间的差异
        return predicted_rgb - target_rgb

    # 初始猜测浓度值
    initial_concentration = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    # 使用最小二乘法进行优化
    result = least_squares(objective_function, initial_concentration, bounds=(0, 1))
    result.x = result.x / result.x.sum()
    # 返回优化后的浓度值
    concentration = np.array([result.x[0], result.x[1], result.x[2], result.x[3], result.x[4], 0])
    return concentration

def brightness_to_kw_concentration_least_square(target_brightness):
    def objective_function(concentration):

        concentration = np.array([0, 0, 0, concentration[0], concentration[1], 0])
        concentration = concentration / concentration.sum()
        predicted_rgb = concentration_to_rgbd(concentration)[:3]  # 只取 RGB 部分
        predicted_brightness = predicted_rgb.mean()
        # 返回目标 RGB 和预测 RGB 之间的差异
        return predicted_brightness - target_brightness
    
    initial_concentration = np.array([0.5, 0.5])
    result = least_squares(objective_function, initial_concentration, bounds=(0, 1))
    result.x = result.x / result.x.sum()
    # 返回优化后的浓度值
    concentration = np.array([result.x[0], result.x[1]])
    return concentration

def rgb_to_concentration(rgb_array):
    # 计算 floor 和 ceil
    r_floor = np.floor(rgb_array[:, 0] * 10) / 10
    r_ceil = np.ceil(rgb_array[:, 0] * 10) / 10
    g_floor = np.floor(rgb_array[:, 1] * 10) / 10
    g_ceil = np.ceil(rgb_array[:, 1] * 10) / 10
    b_floor = np.floor(rgb_array[:, 2] * 10) / 10
    b_ceil = np.ceil(rgb_array[:, 2] * 10) / 10

    # 生成所有可能的组合
    nearest_values = np.array([
        [r_floor, g_floor, b_floor],
        [r_floor, g_floor, b_ceil],
        [r_floor, g_ceil, b_floor],
        [r_floor, g_ceil, b_ceil],
        [r_ceil, g_floor, b_floor],
        [r_ceil, g_floor, b_ceil],
        [r_ceil, g_ceil, b_floor],
        [r_ceil, g_ceil, b_ceil]
    ]).transpose(2, 0, 1)

    def trilinear_interpolation(rgb, values):
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        r0, g0, b0 = values[:, 0, 0], values[:, 0, 1], values[:, 0, 2]

        def lerp(v0, v1, t):
            return v0 + t * (v1 - v0)

        def interpolate(c000, c001, c010, c011, c100, c101, c110, c111):
            c00 = lerp(c000, c001, (b - b0) / rgb_to_concentration_step)
            c01 = lerp(c010, c011, (b - b0) / rgb_to_concentration_step)
            c10 = lerp(c100, c101, (b - b0) / rgb_to_concentration_step)
            c11 = lerp(c110, c111, (b - b0) / rgb_to_concentration_step)

            c0 = lerp(c00, c01, (g - g0) / rgb_to_concentration_step)
            c1 = lerp(c10, c11, (g - g0) / rgb_to_concentration_step)
            return lerp(c0, c1, (r - r0) / rgb_to_concentration_step)

        def value_to_index(value):
            return (value[:, 0] // rgb_to_concentration_step + 1) * ((rgb_to_concentration_interpolate+1)**2) + \
                   (value[:, 1] // rgb_to_concentration_step + 1) * (rgb_to_concentration_interpolate+1) + \
                   (value[:, 2] // rgb_to_concentration_step + 1)

        indices = value_to_index(values.reshape(-1, 3)).astype(int)
        c_values = rgb_to_concentration_map.iloc[indices].values[:, 3:12].reshape(-1, 8, 9)


        c_c = interpolate(c_values[:, 0, 0], c_values[:, 1, 0], c_values[:, 2, 0], c_values[:, 3, 0],
                          c_values[:, 4, 0], c_values[:, 5, 0], c_values[:, 6, 0], c_values[:, 7, 0])
        c_m = interpolate(c_values[:, 0, 1], c_values[:, 1, 1], c_values[:, 2, 1], c_values[:, 3, 1],
                          c_values[:, 4, 1], c_values[:, 5, 1], c_values[:, 6, 1], c_values[:, 7, 1])
        c_y = interpolate(c_values[:, 0, 2], c_values[:, 1, 2], c_values[:, 2, 2], c_values[:, 3, 2],
                          c_values[:, 4, 2], c_values[:, 5, 2], c_values[:, 6, 2], c_values[:, 7, 2])
        c_k = interpolate(c_values[:, 0, 3], c_values[:, 1, 3], c_values[:, 2, 3], c_values[:, 3, 3],
                          c_values[:, 4, 3], c_values[:, 5, 3], c_values[:, 6, 3], c_values[:, 7, 3])
        c_w = interpolate(c_values[:, 0, 4], c_values[:, 1, 4], c_values[:, 2, 4], c_values[:, 3, 4],
                          c_values[:, 4, 4], c_values[:, 5, 4], c_values[:, 6, 4], c_values[:, 7, 4])
        p_r = interpolate(c_values[:, 0, 5], c_values[:, 1, 5], c_values[:, 2, 5], c_values[:, 3, 5],
                          c_values[:, 4, 5], c_values[:, 5, 5], c_values[:, 6, 5], c_values[:, 7, 5])
        p_g = interpolate(c_values[:, 0, 6], c_values[:, 1, 6], c_values[:, 2, 6], c_values[:, 3, 6],
                          c_values[:, 4, 6], c_values[:, 5, 6], c_values[:, 6, 6], c_values[:, 7, 6])
        p_b = interpolate(c_values[:, 0, 7], c_values[:, 1, 7], c_values[:, 2, 7], c_values[:, 3, 7],
                          c_values[:, 4, 7], c_values[:, 5, 7], c_values[:, 6, 7], c_values[:, 7, 7])
        p_d = interpolate(c_values[:, 0, 8], c_values[:, 1, 8], c_values[:, 2, 8], c_values[:, 3, 8],
                          c_values[:, 4, 8], c_values[:, 5, 8], c_values[:, 6, 8], c_values[:, 7, 8])

        return np.stack([c_c, c_m, c_y, c_k, c_w, np.zeros_like(c_c)], axis=1), np.stack([p_r, p_g, p_b, p_d], axis=1)

    concentration = trilinear_interpolation(rgb_array, nearest_values)
    return concentration


def adjust_density(concentration, target_rgbd):

    def calculate_kw_rate(brightness):
        brightness_indices = (brightness * brightness_to_kw_interpolate).astype(int)
        k_rate = brightness_to_kw_concentration_map.iloc[brightness_indices]['k_rate'].values
        w_rate = 1 - k_rate
        return k_rate, w_rate

    def concentration_to_density(concentration):
        density_rgb = concentration @ cmykwe_density_rgb
        alpha_rgb = 1 - np.exp(-density_rgb * z)
        alpha_max = np.max(alpha_rgb, axis=1)
        alpha_min = np.min(alpha_rgb, axis=1)
        alpha = alpha_rgb.mean(axis=1)
        density = -np.log(1 - alpha) / z
        return density

    density = concentration_to_density(concentration)



    target_density = target_rgbd[:, 3]
    brightness = target_rgbd[:, :3].mean(axis=1)

    k_rate, w_rate = calculate_kw_rate(brightness)

    kw_density = concentration_to_density(np.column_stack([np.zeros_like(k_rate), np.zeros_like(k_rate), np.zeros_like(k_rate), k_rate, w_rate, np.zeros_like(k_rate)]))

    add_rate = (target_density - density) / (kw_density - density)
    add_rate = np.clip(add_rate, 0, 0.3)  # 0.3 is the max rate, magicnum

    color_rate = target_density / density
    concentration = np.where(
        density[:, None] < target_density[:, None],
        concentration * (1 - add_rate[:, None]) + np.column_stack([np.zeros_like(k_rate), np.zeros_like(k_rate), np.zeros_like(k_rate), k_rate, w_rate, np.zeros_like(k_rate)]) * add_rate[:, None],
        np.column_stack([concentration[:, :5] * (color_rate)[:, None], 1 - (color_rate)])
    )
    return concentration, concentration_to_density(concentration)

def rgbd_to_concentration(rgbd):
    rgb = rgbd[:, :3]
    concentration, pred_rgbd = rgb_to_concentration(rgb)
    concentration, new_density = adjust_density(concentration, rgbd)
    pred_rgbd[:, 3] = new_density
    return concentration, pred_rgbd


# if __name__ == "__main__":


    # wavelengths = np.arange(300, 801, 1)  # 波长范围 400-700nm, 间隔 10nm
    # # spectra = np.random.rand(3, len(wavelengths))  # 随机生成 5 个样本的光谱数据
    # for color in ['c', 'm', 'y', 'k', 'w']:
    #     data = pd.read_csv(f'./color/data/calib_data/{color}.csv')
    #     K[color], S[color] = data['K'].values, data['S'].values
    #     spectra = data['R'].values.reshape(1, -1)
        

    #     # 计算 RGB
    #     # rgb_values = calculate_rgb_vectorized(wavelengths, spectra)

    #     # print(rgb_values)


    #     for i in range(len(spectra)):
    #         print(calculate_rgb(wavelengths, spectra[i]))


    # rgbds = np.array([
    #     [0.11, 0.22, 0.99, 0.1],
    #     [0.22, 0.33, 0.44, 1],
    #     [0.47556596,0.4282674,0.48980536, 10],
    #     [0.4,0.7,0.8, 100]
    # ])
    # import time
    # start_time = time.time()    
    # rgbds = np.random.rand(10000, 4)
    # concentrations, pred_rgbds = rgb_to_concentration(rgbds[:, :3])
    # end_time = time.time()

    # # print(concentrations)
    # # for concentration in concentrations:
    # #     print(concentration_to_rgbd(concentration))
    # # print('--------------------------------')
    # # for concentration in concentrations:
    # #     print(concentration_to_rgbd(concentration))
    # # exit()
    # concentrations, new_density = adjust_density(concentrations, rgbds)
    # # print(concentrations)
    # print(f"Time taken: {end_time - start_time} seconds")



    # print(rgbd_to_concentration(rgbds))

    # for concentration in concentrations:
    #     print(concentration_to_rgbd(concentration))




if __name__ == "__main__":
    results = []

    # 计算总的迭代次数
    total_iterations = 11 * 11 * 11

    # 使用 tqdm 包装最内层循环以显示进度条
    with tqdm(total=total_iterations, desc="Processing RGB values") as pbar:
        for r in range(11):
            for g in range(11):
                for b in range(11):
                    rgb = np.array([r, g, b]) / 10
                    concentration = rgb_to_concentration_least_square(rgb)
                    pred_rgbd = concentration_to_rgbd(concentration)
                    
                    # 将结果添加到列表中
                    results.append({
                        'r': rgb[0],
                        'g': rgb[1],
                        'b': rgb[2],
                        'c_c': concentration[0],
                        'c_m': concentration[1],
                        'c_y': concentration[2],
                        'c_k': concentration[3],
                        'c_w': concentration[4],
                        'pred_r': pred_rgbd[0],
                        'pred_g': pred_rgbd[1],
                        'pred_b': pred_rgbd[2],
                        'pred_d': pred_rgbd[3]
                    })

                    # 更新进度条
                    pbar.update(1)

    # 创建 DataFrame
    df = pd.DataFrame(results)

    # 将 DataFrame 写入 CSV 文件
    df.to_csv('color/data/color_map/10.csv', index=False)


    results = []

    for i in range(101):
        brightness = i / 100
        concentration = brightness_to_kw_concentration_least_square(brightness)
        k_rate = concentration[0]
        w_rate = concentration[1]
        results.append({
            'brightness': brightness,
            'k_rate': k_rate,
            'w_rate': w_rate
        })
        print(brightness, k_rate, w_rate)

    # 创建 DataFrame
    df = pd.DataFrame(results)

    # 将 DataFrame 写入 CSV 文件
    df.to_csv('color/data/color_map/brightness_kw_rates.csv', index=False)


