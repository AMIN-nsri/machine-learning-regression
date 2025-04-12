import pandas as pd
import math

import numpy as np


df = pd.read_excel('train.xlsx')

x = df['x'].values
y = df['y'].values
z = df['z'].values
f = df['F(x, y, z)'].values  # Target

X = np.column_stack([
    np.ones(len(x)),   # Bias term (b)
    x, x**2, x**3,     # x terms
    y, y**2, y**3,     # y terms
    z, z**2, z**3,     # z terms
    x*y, x**2*y, x*y**2,  # xy interaction
    x*z, x**2*z, x*z**2,  # xz interaction
    y*z, y**2*z, y*z**2,  # yz interaction
    x*y*z  # xyz interaction
])

# the Normal Equation
W = np.linalg.inv(X.T @ X) @ X.T @ f

print("b:", W[0])
print("W:")
for i, weight in enumerate(W[1:], start=1):
    print(f"w_{i} = {weight}")

def F(x, y, z):
    b =         round(-3.410605131648481e-13, 4)  # Intercept (b)
    # print(b)
    w = [
        round(0.5000000000003264, 4),  # w_1
        round(1.0000000000000022, 4),  # w_2
        round(0.24999999999999642, 4),  # w_3
        round(0.5999999999998531, 4),  # w_4
        round(0.20000000000000362, 4),  # w_5
        round(0.40000000000000024, 4),  # w_6
        round(0.2499999999997673, 4),  # w_7
        round(0.1100000000000037, 4),  # w_8
        round(0.23000000000000165, 4),  # w_9
        round(0.3100000000000005, 4),  # w_10
        round(0.7600000000000016, 4),  # w_11
        round(0.1400000000000004, 4),  # w_12
        round(0.1500000000000005, 4),  # w_13
        round(0.6200000000000011, 4),  # w_14
        round(0.8899999999999996, 4),  # w_15
        round(0.29999999999999993, 4),  # w_16
        round(0.09999999999999999, 4),  # w_17
        round(0.5999999999999993, 4),  # w_18
        0.21  # w_19
    ]
    # print(w)
    return (w[0]*x + w[1]*x**2 + w[2]*x**3 +
            w[3]*y + w[4]*y**2 + w[5]*y**3 +
            w[6]*z + w[7]*z**2 + w[8]*z**3 +
            w[9]*x*y + w[10]*x**2*y + w[11]*x*y**2 +
            w[12]*x*z + w[13]*x**2*z + w[14]*x*z**2 +
            w[15]*y*z + w[16]*y**2*z + w[17]*y*z**2 +
            w[18]*x*y*z + b)
def ceil_to_precision(value, precision):
    factor = 10 ** precision
    return math.ceil(value * factor) / factor


# def predict_f(x, y, z):
#     # Compute f(x, y, z) using precomputed polynomial regression weights
#     W = np.array([
#         -3.410605131648481e-13,  # Intercept (b)
#         0.5000000000003264,  # w_1
#         1.0000000000000022,  # w_2
#         0.24999999999999642,  # w_3
#         0.5999999999998531,  # w_4
#         0.20000000000000362,  # w_5
#         0.40000000000000024,  # w_6
#         0.2499999999997673,  # w_7
#         0.1100000000000037,  # w_8
#         0.23000000000000165,  # w_9
#         0.3100000000000005,  # w_10
#         0.7600000000000016,  # w_11
#         0.1400000000000004,  # w_12
#         0.1500000000000005,  # w_13
#         0.6200000000000011,  # w_14
#         0.8899999999999996,  # w_15
#         0.29999999999999993,  # w_16
#         0.09999999999999999,  # w_17
#         0.5999999999999993,  # w_18
#         0.21  # w_19
#     ])
#
#     # Construct the feature vector
#     X_new = np.array([
#         1, x, x ** 2, x ** 3,  # x
#         y, y ** 2, y ** 3,  # y
#         z, z ** 2, z ** 3,  # z
#               x * y, x ** 2 * y, x * y ** 2,  # xy
#               x * z, x ** 2 * z, x * z ** 2,  # xz
#               y * z, y ** 2 * z, y * z ** 2,  # yz
#               x * y * z  # xyz
#     ])
#
#     return round(np.dot(W, X_new), 2)


# # chatgpt's solution for a silly bug
# try:
#     del input  # Remove any reassignment of `input` if it exists
# except NameError:
#     pass  # `input` was not overwritten, so continue

# input = input().split()
# x_test, y_test, z_test = float(input[0]), float(input[1]), float(input[2])
x_test = float(input())
y_test = float(input())
z_test = float(input())
# predicted_f = predict_f(x_test, y_test, z_test)
# print(predicted_f)
print(ceil_to_precision(F(x_test, y_test, z_test),2))