import sys
import numpy as np
from PIL import ImageQt
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QPoint

from plot_utils import draw_skeleton, pil_image_to_qpixmap


def quaternion_from_axis_angle(axis, angle):
    half_angle = angle / 2
    sin_half_angle = np.sin(half_angle)
    return np.array([np.cos(half_angle)] + [sin_half_angle * a for a in axis])


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2,
        w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2,
    ])


def quaternion_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quaternion_rotate_vector(q, v):
    qv = np.array([0] + v.tolist())
    q_conj = quaternion_conjugate(q)
    return quaternion_multiply(quaternion_multiply(q, qv), q_conj)[1:]


class SkeletonViewer(QWidget):
    def __init__(self, keypoints, parent=None):
        super(SkeletonViewer, self).__init__(parent)
        self.keypoints = keypoints

        self.rotation_quaternion = np.array([1, 0, 0, 0])  # Identity quaternion
        self.last_mouse_pos = None

        self.image_label = QLabel(self)
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        self.setLayout(layout)

        self.update_image()

    def update_image(self):
        normal_vector = quaternion_rotate_vector(self.rotation_quaternion, np.array([0, 0, 1]))
        image = draw_skeleton(self.keypoints, normal_vector)
        pixmap = pil_image_to_qpixmap(image)
        self.image_label.setPixmap(pixmap)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self.last_mouse_pos is not None:
            dx = event.x() - self.last_mouse_pos.x()
            dy = event.y() - self.last_mouse_pos.y()

            # 从用户视角调整视角
            horizontal_axis = np.array([0, 1, 0])
            vertical_axis = np.array([1, 0, 0])
            yaw_quaternion = quaternion_from_axis_angle(horizontal_axis, np.deg2rad(dx * 0.5))
            pitch_quaternion = quaternion_from_axis_angle(vertical_axis, np.deg2rad(dy * 0.5))
            # 更新旋转四元数
            self.rotation_quaternion = quaternion_multiply(self.rotation_quaternion, yaw_quaternion)
            self.rotation_quaternion = quaternion_multiply(self.rotation_quaternion, pitch_quaternion)

            self.update_image()
            self.last_mouse_pos = event.pos()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_mouse_pos = None


if __name__ == '__main__':
    app = QApplication(sys.argv)
    keypoints = np.random.random((33, 3)) # 生成随机关键点作为示例
    connections = []  # 这里没有连接线

    main_window = QMainWindow()
    viewer = SkeletonViewer(keypoints)
    main_window.setCentralWidget(viewer)
    main_window.show()

    sys.exit(app.exec_())
