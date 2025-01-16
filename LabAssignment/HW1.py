# import numpy as np
#
# N, M = map(int, input().split())
# train_data = []
# test_data = []
# for i in range(N):
#     train_data.append(list(map(float, input().split())))
# for i in range(M):
#     test_data.append(float(input()))
# x = np.array(train_data)
# x_poly = np.column_stack([x[:, 0]**3, x[:, 0]**2, x[:, 0], np.ones(x.shape[0])])
# w = np.linalg.inv(x_poly.T @ x_poly) @ x_poly.T @ x[:, 1]
# for i in test_data:
#     print(w[0] * i**3 + w[1] * i**2 + w[2] * i + w[3])

import numpy as np

N, M = map(int, input().split())
train_data = np.array([list(map(float, input().split())) for _ in range(N)])
test_data = np.array([float(input()) for _ in range(M)])

x_poly = np.column_stack([train_data[:, 0]**3, train_data[:, 0]**2, train_data[:, 0], np.ones(N)])
w = np.linalg.inv(x_poly.T @ x_poly) @ x_poly.T @ train_data[:, 1]

for i in test_data:
    print(w[0] * i**3 + w[1] * i**2 + w[2] * i + w[3])
