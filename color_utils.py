import numpy as np
import cv2

def SK_to_RT(S, K, x):
    # 计算 a 和 b
    a = (S + K) / S
    b = np.sqrt(a**2 - 1)
    
    # 计算 bSx
    bSx = b * S * x[:, np.newaxis]
    
    # 计算 R 和 T
    sinh_bSx = np.sinh(bSx)
    cosh_bSx = np.cosh(bSx)
    
    R = sinh_bSx / (a * sinh_bSx + b * cosh_bSx)
    T = b / (a * sinh_bSx + b * cosh_bSx)
    
    return R, T
    

def RT_to_RGBA(R, T):
    A = T.mean(axis=1)
    return np.concatenate([R, A.reshape(-1, 1)], axis=1)



if __name__ == "__main__":
    S = np.random.rand(100, 3)
    K = np.random.rand(100, 3)
    x = np.random.rand(100)
    R, T = SK_to_RT(S, K, x)
    print(R.shape, T.shape)
    print(RT_to_RGBA(R, T).shape)