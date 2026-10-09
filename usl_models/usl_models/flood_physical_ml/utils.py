import numpy as np
import scipy.special as sp
import tensorflow as tf

def get_gauss_legendre_IRK(q):
    x, w = sp.roots_legendre(q)
    c = 0.5 * (x + 1.0)
    b = 0.5 * w

    V = np.zeros((q, q))
    C = np.zeros((q, q))
    for i in range(q):
        for j in range(q):
            V[i, j] = c[i]**j
            C[i, j] = (c[i]**(j + 1)) / (j + 1)
            
    A = np.dot(C, np.linalg.inv(V))
    A = tf.convert_to_tensor(A, dtype=tf.float32)
    b = tf.convert_to_tensor(b, dtype=tf.float32)
    b = tf.reshape(b, (1, q))
    return A, b, c

def scale_back_dem(
    dem: np.array, 
    max_value: np.array, 
    min_value: np.array):
    return dem * (max_value-min_value) + min_value


if __name__ == "__main__":
    A, b, c = get_gauss_legendre_IRK(3)
    print(A)
    print(b)
    print(c)
    # test_dem = tf.random.normal([1000, 1000, 1])
    # rescaled_dem = scale_back_dem(test_dem, 50, 20)
    # print(test_dem[0, 0, 0])
    # print(rescaled_dem[0, 0, 0])