import numpy as np
from PIL import Image, ImageDraw
from PyQt5.QtGui import QImage, QPixmap

from detect_utils import connections


def pil_image_to_qpixmap(pil_image):
    # 转换为 QImage 格式
    image_data = pil_image.convert("RGBA").tobytes("raw", "RGBA")
    q_image = QImage(image_data, pil_image.width, pil_image.height, QImage.Format_RGBA8888)
    # 转换为 QPixmap 格式
    q_pixmap = QPixmap.fromImage(q_image)
    return q_pixmap


def get_color(x):
    r, g, b = 0, 0, 0
    if x <= 0.5:
        r = 255
        g = int(255 * (x * 2))
    else:
        r = int(255 * (1 - (x - 0.5) * 2))
        g = 255
    return r, g, b, 255


# def draw_skeleton(keypoints, normal_vector=(0, 1, 1), point_radius=3, line_width=2, bone_color=None) -> Image.Image:
#     # 创建带透明度的背景透明图片
#     image = Image.new('RGBA', (512, 512), (0, 0, 0, 0))
#     draw = ImageDraw.Draw(image)
#
#     keypoints = keypoints * np.array([512, 512, 512])
#
#     # 计算投影矩阵
#     normal_vector = np.array(normal_vector) / np.linalg.norm(normal_vector)
#     projection_matrix = np.identity(3) - np.outer(normal_vector, normal_vector)
#
#     # 投影关键点
#     projected_keypoints = np.dot(keypoints, projection_matrix.T)
#
#     # 标准化坐标并映射到图片尺寸
#     min_xy = np.min(projected_keypoints[:, :2], axis=0)
#     max_xy = np.max(projected_keypoints[:, :2], axis=0)
#     normalized_keypoints = (projected_keypoints[:, :2] - min_xy) / (max_xy - min_xy)
#     image_keypoints = normalized_keypoints * np.array([512, 512])
#
#     # 绘制连接线
#     for i, connection in enumerate(connections):
#         start_point = image_keypoints[connection[0]]
#         end_point = image_keypoints[connection[1]]
#         draw.line([tuple(start_point), tuple(end_point)], fill=bone_color[i] if bone_color else (255, 255, 255, 255), width=line_width)
#
#     # 绘制关键点
#     for keypoint in image_keypoints:
#         x, y = keypoint
#         draw.ellipse([x - point_radius, y - point_radius, x + point_radius, y + point_radius], fill=(255, 0, 0, 255))
#
#     return image


def draw_skeleton(keypoints, normal_vector=(0, 1, 1), point_radius=3, line_width=2, bone_color=None) -> Image.Image:
    # 创建带透明度的背景透明图片
    image = Image.new('RGBA', (512, 512), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    keypoints = keypoints * np.array([512, 512, 512])
    axes_points = np.array([[0, 0, 0], [0, 0, 512], [0, 512, 0], [512, 0, 0]])

    # 计算投影矩阵
    normal_vector = np.array(normal_vector) / np.linalg.norm(normal_vector)
    projection_matrix = np.identity(3) - np.outer(normal_vector, normal_vector)


    # 投影关键点
    offset = 512 * np.ones(3) * 0.5
    projected_keypoints = np.dot(keypoints, projection_matrix.T) - offset
    projected_axes_points = np.dot(axes_points, projection_matrix.T)
    # 绘制关键点
    for keypoint in projected_keypoints:
        x, y = keypoint[:2] + offset[:2]
        draw.ellipse([x - point_radius, y - point_radius, x + point_radius, y + point_radius], fill=(255, 0, 0, 255))

    # 绘制连接线
    for i, connection in enumerate(connections):
        start_point = projected_keypoints[connection[0]][:2]
        end_point = projected_keypoints[connection[1]][:2]
        draw.line([tuple(start_point+offset[:2]), tuple(end_point+offset[:2])], fill=bone_color[i] if bone_color else (255, 255, 255, 255), width=line_width)

    start_point = projected_axes_points[0][:2]
    for i in range(1, 4):
        end_point = projected_axes_points[i][:2]
        draw.line([tuple(start_point+offset[:2]), tuple(end_point+offset[:2])], fill=[(0, 0, 255), (0, 255, 0), (255, 0, 0)][i-1], width=line_width)

    return image


# # 示例
# num_points = 33
# keypoints = np.random.random((num_points, 3))
# image = draw_skeleton(keypoints, connections)
# image.show()
