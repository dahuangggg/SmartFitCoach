import numpy as np
from collections import namedtuple

connections = [
    (12, 24), (24, 26), (26, 28), (28, 32), (32, 30), (30, 28),
    (11, 23), (23, 25), (25, 27), (27, 29), (29, 31), (31, 27),
    (12, 11), (24, 23),
    (12, 14), (14, 16), (16, 18), (18, 20), (20, 16), (16, 22),
    (11, 13), (13, 15), (15, 17), (17, 19), (19, 15), (15, 21),
    (8, 6), (6, 5), (5, 4), (4, 0), (0, 1), (1, 2), (2, 3), (3, 7),
    (10, 9)
]

neighbours = []
for i in range(33):
    buffer = []
    for m, n in connections:
        if m == i:
            buffer.append(n)
        elif n == i:
            buffer.append(m)
    neighbours.append(buffer)

# add additional connections
for m, n in [(12, 11), (12, 24), (11, 12), (11, 23), (23, 11), (23, 24), (24, 23), (24, 12), (18, 20), (20, 18), (17, 19), (19, 17), (32, 30), (30, 32), (29, 31), (31, 29)]:
    neighbours[m].remove(n)



def landmarks_to_numpy(landmarks):
    return np.array([(landmark.x, landmark.y, landmark.z) for landmark in landmarks.landmark])


def landmarks_to_bone_arrays(landmarks):
    arr = np.zeros((len(connections), 3))
    lens = np.zeros(len(connections))
    for i, (start, end) in enumerate(connections):
        dx = landmarks.landmark[end].x - landmarks.landmark[start].x
        dy = landmarks.landmark[end].y - landmarks.landmark[start].y
        dz = landmarks.landmark[end].z - landmarks.landmark[start].z
        arr[i, :] = dx, dy, dz
        length = np.sqrt(dx**2+dy**2+dz**2)
        arr[i] /= length
        lens[i] = length
    return arr, lens


def normalize(arr):
    mean = np.mean(arr, axis=0)
    std = np.std(arr)
    return (arr-mean)/std, mean, std


def get_bottom_point(arr):
    i = np.argmin(arr[:, 1])
    return arr[i], i


def transform(arr, arr2, multiple=1):  # 你的关键点、标准关键点
    visited = np.zeros(33, dtype=bool)
    arr1 = arr.copy()

    def dfs(i):
        visited[i] = True
        for idx in neighbours[i]:  # connect i->idx
            dl2 = arr2[idx] - arr2[i]
            # direction = dl2 / (dl2 * dl2).sum()
            # dl = arr[idx] - arr[i]
            # length = (dl * dl).sum()
            arr1[idx] = arr1[i] + dl2*multiple
            if not visited[idx]:
                dfs(idx)
    for base in [12, 11, 24, 23]:
        dfs(base)
    return arr1

