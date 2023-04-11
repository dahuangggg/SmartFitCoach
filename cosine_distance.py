import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d


connections = [
    (12, 24), (24, 26), (26, 28), (28, 32), (32, 30), (30, 28),
    (11, 23), (23, 25), (25, 27), (27, 29), (29, 31), (31, 27),
    (12, 11), (24, 23),
    (12, 14), (14, 16), (16, 18), (18, 20), (20, 16), (16, 22),
    (11, 13), (13, 15), (15, 17), (17, 19), (19, 15), (15, 21),
    (8, 6), (6, 5), (5, 4), (4, 0), (0, 1), (1, 2), (2, 3), (3, 7),
    (10, 9)
]

np.random.seed(42)


def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def add_noise(points, noise_factor=0.1):
    return points + noise_factor * np.random.randn(*points.shape)


def kabsch(P, Q):
    # 计算中心点
    P_center = np.mean(P, axis=0)
    Q_center = np.mean(Q, axis=0)

    # 移动到原点
    P_centered = P - P_center
    Q_centered = Q - Q_center

    # 计算协方差矩阵
    C = np.dot(np.transpose(P_centered), Q_centered)

    # 计算SVD
    U, _, Vt = np.linalg.svd(C)

    # 计算最优旋转矩阵
    d = np.sign(np.linalg.det(np.dot(Vt.T, U.T)))
    R = np.dot(Vt.T, np.dot(np.diag([1, 1, d]), U.T))

    return R


def best_rotation_using_svd(P, Q):
    H = np.dot(P.T, Q)
    U, _, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    return R


def landmarks_to_numpy(landmarks):
    return np.array([(landmark.x, landmark.y, landmark.z) for landmark in landmarks.landmark])


def landmarks_to_bone_arrays(landmarks):
    arr = np.zeros((len(connections), 3))
    for i, (start, end) in enumerate(connections):
        dx = landmarks.landmark[end].x - landmarks.landmark[start].x
        dy = landmarks.landmark[end].y - landmarks.landmark[start].y
        dz = landmarks.landmark[end].z - landmarks.landmark[start].z
        arr[i, :] = dx, dy, dz
        arr[i] /= np.sqrt(dx**2+dy**2+dz**2)
    return arr


def numpy_to_bone_arrays(arr):
    ret_arr = np.zeros((len(connections), 3))
    for i, (start, end) in enumerate(connections):
        dx = arr[end, 0] - arr[start, 0]
        dy = arr[end, 1] - arr[start, 1]
        dz = arr[end, 2] - arr[start, 2]
        ret_arr[i, :] = dx, dy, dz
        ret_arr[i] /= np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    return ret_arr


if __name__ == "__main__":
    # landmarks1 = ...  # 第一个姿态的关键点
    # landmarks2 = ...  # 第二个姿态的关键点
    #
    # P = landmarks_to_numpy(landmarks1)
    # Q = landmarks_to_numpy(landmarks2)
    for i in range(10):
        P_base = np.random.random((33, 3))
        angle = np.random.uniform(0, 2 * np.pi)
        rotation = R.from_rotvec([0, 0, angle])  # 创建一个绕Z轴旋转的旋转矩阵
        Q_base = rotation.apply(P_base)

        P = add_noise(P_base)
        Q = add_noise(Q_base)

        # 计算最优旋转矩阵
        R_optimal = best_rotation_using_svd(P, Q)

        # 将第一个姿势旋转到第二个姿势的坐标系中
        P_rotated = np.dot(P, R_optimal)

        # 计算旋转后的相似性
        similarity = cosine_similarity(P_rotated.flatten(), Q.flatten())
        # 计算旋转前的相似性
        similarity_ = cosine_similarity(P.flatten(), Q.flatten())
        print(f"旋转后的余弦相似度:{similarity}", f"旋转前的余弦相似度:{similarity_}", similarity > similarity_)


# 定义x和y值
x_values = [0, np.cos(np.radians(60)), np.cos(np.radians(45)), np.cos(np.radians(30)), 1]
y_values = [0, 0.3, 0.5, 0.9, 1]
# 使用线性插值创建插值函数
interpolation_function = interp1d(x_values, y_values, kind='linear', fill_value=(0, 1), bounds_error=False)
