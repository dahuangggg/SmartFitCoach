import random
import sys
from typing import Union

import cv2
import os
import numpy as np
from PyQt5.QtCore import QSize, pyqtSignal, QPropertyAnimation, QRect, Qt, QEasingCurve, QTimer, QPoint, QRectF, \
    QVariant, pyqtProperty, QThread
from PyQt5.QtGui import QIcon, QPainter, QColor, QPainterPath, QLinearGradient, QPalette, QPixmap, QCursor, QFont, \
    QBrush, QImage
from PyQt5.QtWidgets import QPushButton, QSlider, QStyleOptionSlider, QStyle, QLabel, QWidget, QApplication, \
    QMainWindow, QLayout, QVBoxLayout, QStackedLayout, QScrollArea, QHBoxLayout, QSpacerItem, QSizePolicy, QFrame
from PyQt5 import QtWidgets, QtGui, QtCore
from glob import glob
import re
import mediapipe as mp

import constants as c
from plot_utils import draw_skeleton, pil_image_to_qpixmap
from cosine_distance import *
from test import quaternion_rotate_vector, quaternion_from_axis_angle, quaternion_multiply

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


class MyPushButton(QPushButton):
    def __init__(self, content="", parent=None, slot=None, geometry: tuple = None,
                 style_sheet='QPushButton {background-color: transparent; border: none; color: white;}',
                 icon: Union[str, QIcon] = None, icon_size: tuple = (18, 18), pushed_effect=True, on_enter=None, tips=""):
        super().__init__(content, parent)
        self.slot = slot
        self.icon = icon
        self.icon_size = icon_size,
        self.pushed_effect = pushed_effect
        self.on_enter = on_enter
        if slot:
            self.released.connect(slot)
        if geometry:
            geometry = tuple(map(int, geometry))
            self.setGeometry(*geometry)
        if icon:
            if isinstance(icon, str):
                self.setIcon(QIcon("icons/"+self.icon+".png"))
            else:
                self.setIcon(icon)
        if icon_size:
            self.setIconSize(QSize(*icon_size))
        if tips:
            self.setToolTip(tips)
        self.setStyleSheet(style_sheet)
        # self.setMouseTracking(True)

    def mousePressEvent(self, e: QtGui.QMouseEvent) -> None:
        self.clicked.emit()
        e.ignore()

    def mouseReleaseEvent(self, event):
        if event.pos() in self.rect():
            self.released.emit()
        # event.ignore()

    def enterEvent(self, event):
        if self.on_enter is not None:
            self.on_enter()
        if self.pushed_effect and self.icon:
            self.setIcon(QIcon("icons/"+self.icon+"_pushed.png"))

    def leaveEvent(self, event):
        if self.pushed_effect and self.icon:
            self.setIcon(QIcon("icons/"+self.icon+".png"))


class ExitButton(MyPushButton):  # 套一层壳， 用于区分相对较特殊的退出按钮
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class CameraExitButton(MyPushButton):  # 套一层壳， 用于区分相对较特殊的退出按钮
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ChangeableButton(QPushButton):
    state_changed = pyqtSignal(int)

    def __init__(self, content="", parent=None, geometry: tuple = None,
                 style_sheet='QPushButton {background-color: transparent; border: none; color: white;}',
                 icons: tuple = (), icon_size: tuple = (18, 18), i=0, pushed_effect=True, on_enter=None, on_leave=None, tips=None):
        super().__init__(content, parent)
        self.num_states = len(icons)
        self.i = i
        self.icons = icons
        self.pushed_effect = pushed_effect
        self.on_enter = on_enter
        self.on_leave = on_leave
        self.tips = tips
        if geometry:
            geometry = tuple(map(int, geometry))
            self.setGeometry(*geometry)
        if icons:
            self.setIcon(QIcon("icons/"+icons[0]+".png"))
        if icon_size:
            self.setIconSize(QSize(*icon_size))
        if isinstance(tips, str):
            self.setToolTip(tips)
        self.setStyleSheet(style_sheet)
        self.released.connect(self.change_state)
        # self.setMouseTracking(True)

    def change_state(self):
        self.i += 1
        if self.i == self.num_states:
            self.i = 0
        if isinstance(self.tips, list):
            self.setToolTip(self.tips[self.i])
        if self.pushed_effect:
            self.setIcon(QIcon("icons/"+self.icons[self.i]+"_pushed.png"))
        else:
            self.setIcon(QIcon("icons/"+self.icons[self.i]+".png"))

        self.state_changed.emit(self.i)

    def set_state(self, i):
        self.i = i - 1
        self.change_state()

    def enterEvent(self, event):
        if self.on_enter is not None:
            self.on_enter()
        if self.pushed_effect and self.icons:
            self.setIcon(QIcon("icons/"+self.icons[self.i]+"_pushed.png"))

    def leaveEvent(self, event):
        if self.pushed_effect and self.icons:
            self.setIcon(QIcon("icons/"+self.icons[self.i]+".png"))
        if self.on_leave is not None:
            self.on_leave()

    def mousePressEvent(self, event) -> None:
        self.clicked.emit()
        event.ignore()

    def mouseReleaseEvent(self, event):
        if event.pos() in self.rect():
            self.released.emit()
        # event.ignore()


class MyVerticalSlider(QSlider):
    def __init__(self, parent=None, geometry: tuple = None, min=0, max=100, value=50, visible=True):
        super().__init__(Qt.Vertical, parent)
        self.setMinimum(min)
        self.setMaximum(max)
        self.setValue(value)
        self.setVisible(visible)
        self.alpha = 200
        if geometry:
            self.setGeometry(*geometry)

    def paintEvent(self, event):
        painter = QPainter(self)

        # 绘制背景
        bg_path = QPainterPath()
        bg_path.addRoundedRect(0, 0, self.width(), self.height(), 5, 5)
        bg_color = QColor(50, 50, 50, self.alpha)
        painter.fillPath(bg_path, bg_color)

        # 绘制音量条
        volume_path = QPainterPath()
        volume_path.addRoundedRect(2, self.height() - self.height() * (self.value() / self.maximum()) + 2, self.width() - 4, self.height() * (self.value() / self.maximum()) - 4, 5, 5)
        volume_gradient = QLinearGradient(0, 0, 0, self.height())
        volume_gradient.setColorAt(0, QColor(100, 100, 100, self.alpha))
        volume_gradient.setColorAt(1, QColor(255, 255, 255, self.alpha))
        # volume_gradient.setColorAt(1, QColor(0, 255, 0))
        painter.fillPath(volume_path, volume_gradient)

    def mousePressEvent(self, event):
        self.setValue(int(((self.height() - event.y()) / self.height()) * self.maximum()))

    def mouseMoveEvent(self, event):
        self.setValue(int(((self.height() - event.y()) / self.height()) * self.maximum()))

    def set_alpha(self, alpha):
        self.alpha = alpha
        self.update()


class VolumeControl(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.bar = MyVerticalSlider(self, geometry=(8, 0, 14, 75))
        self.bar.hide()
        self.button = ChangeableButton('', self, icons=("sound", "mute"), icon_size=(24, 18), on_enter=self.bar.show)
        self.button.setGeometry(0, self.bar.height(), 30, 30)
        self.resize(30, 105)
        self.button.state_changed.connect(lambda x: self.bar.set_alpha(200 - 150 * x))
        self.timer = QTimer()
        self.timer.timeout.connect(self.bar.hide)

    def leaveEvent(self, event):
        self.timer.start(600)

    def enterEvent(self, event):
        self.timer.stop()


class MyHorizontalSlider(QSlider):
    position_change = pyqtSignal(int)
    value_set = pyqtSignal(int)
    pressed = pyqtSignal()
    released = pyqtSignal()

    def __init__(self, parent=None, geometry: tuple = None, min=0, max=10000, value=0):
        super().__init__(Qt.Horizontal, parent)

        self.pressing = False
        self.slider_width = 10
        self.setMinimum(min)
        self.setMaximum(max)
        self.setValue(value)
        self.slider_position = None
        if geometry:
            geometry = tuple(map(int, geometry))
            self.setGeometry(*geometry)

    def paintEvent(self, event):
        painter = QPainter(self)

        # 绘制背景
        bg_path = QPainterPath()
        bg_path.addRoundedRect(0, 0, self.width(), self.height(), 5, 5)
        bg_color = QColor(50, 50, 50, 150)
        painter.fillPath(bg_path, bg_color)

        # 绘制j进度条
        progress_path = QPainterPath()
        progress_path.addRoundedRect(2, 2, self.width() * (self.value() / self.maximum()) - 4, self.height() - 4, 5, 5)
        progress_gradient = QLinearGradient(0, 0, self.width(), 0)
        progress_gradient.setColorAt(0, QColor(150, 0, 150, 150))
        progress_gradient.setColorAt(1, QColor(255, 0, 255, 150))
        painter.fillPath(progress_path, progress_gradient)

        # 绘制小滑块
        if self.slider_position is not None:
            option = QStyleOptionSlider()
            self.initStyleOption(option)
            option.subControls = QStyle.SC_SliderHandle
            option.sliderPosition = self.slider_position
            option.sliderValue = self.value()
            option.sliderMinimum = self.minimum()
            option.sliderMaximum = self.maximum()

            style = self.style()
            rect = style.subControlRect(QStyle.CC_Slider, option, QStyle.SC_SliderHandle, self)
            painter.setBrush(Qt.white)
            painter.setPen(Qt.white)
            painter.drawEllipse(rect)

    def mousePressEvent(self, event):
        self.pressing = True  # 设为True 应在 setValue触发valueChanged 之前， 这样单次点击才能触发实时预览
        self.setValue(int((event.x() / self.width()) * self.maximum()))
        self.slider_width = self.width() / (self.maximum() - self.minimum())
        x = event.x()
        self.slider_position = int((x - self.slider_width / 2) / self.slider_width)

        self.pressed.emit()

    def mouseMoveEvent(self, event):
        ratio = event.x() / self.width()
        value = np.clip(ratio * self.maximum(), self.minimum(), self.maximum())

        # self.valueChanged.emit(ratio)
        self.setValue(int(value))
        self.slider_width = self.width() / (self.maximum() - self.minimum())
        x = event.x()
        self.slider_position = np.clip(int((x - self.slider_width / 2) / self.slider_width), self.minimum(), self.maximum())

    def mouseReleaseEvent(self, event):
        self.slider_position = None
        self.value_set.emit(self.value())
        self.pressing = False
        self.released.emit()


class MyTextLabel(QLabel):
    def __init__(self, parent=None, text=None, geometry: tuple = (), font_size=8, bold=False, italic=False, color: tuple = (255, 255, 255), bg_color: tuple = (0, 0, 0, 0), index=None):
        super().__init__(parent=parent)
        self.index = index
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), QtGui.QColor(*bg_color))
        self.setPalette(p)
        self.setAutoFillBackground(True)
        self.setStyleSheet(f"background-color: rgba{bg_color}; color: rgb{color};")
        font = QtGui.QFont()
        font.setFamily("Verdana")
        # "Arial"
        # "Helvetica"
        # "Times"
        # "Courier"
        # "Verdana"
        # "Tahoma"
        # "Impact"
        # "Comic Sans MS"
        font.setPointSize(font_size)
        font.setBold(bold)
        font.setItalic(italic)
        self.setFont(font)
        if text:
            self.setText(text)
        if geometry:
            geometry = tuple(map(int, geometry))
            self.setGeometry(*geometry)
        # Add the label to a layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self)

    def update_text(self, text):
        self.setText(text)
        self.update()


class DurationLabel(MyTextLabel):
    value_changed = pyqtSignal(int)

    def __init__(self, parent=None, duration=0., **kwargs):
        self.duration = duration
        kwargs['text'] = f"{int(duration)//60:02d}:{int(duration)%60:02d}"
        super().__init__(parent=parent, **kwargs)

    def setDuration(self, duration: float):
        self.duration = duration
        self.value_changed.emit(duration)
        self.setText(f"{int(duration)//60:02d}:{int(duration)%60:02d}")


class SearchBox(QWidget):
    def __init__(self, parent=None, geometry: tuple = ()):
        super().__init__(parent)

        # Set the widget's background to transparent
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        # Create a layout for the widget
        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Create a label for the search icon
        self.icon_label = QtWidgets.QLabel()

        search_icon = QPixmap("icons/search-icon.png")
        search_icon = search_icon.scaled(QtCore.QSize(20, 20), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.icon_label.setPixmap(search_icon)
        self.icon_label.resize(30, 30)
        # self.icon_label.setScaledContents(True)
        self.layout.addWidget(self.icon_label)

        # Create a line edit for the search text
        self.text_edit = QtWidgets.QLineEdit()
        self.timer = QTimer()
        self.timer.timeout.connect(self.text_edit.hide)
        self.text_edit.setStyleSheet("background-color: rgba(255, 255, 255, 30);"
                                     "border-radius: 5px;"
                                     "padding: 5px;"
                                     "color: white;")
        self.text_edit.setPlaceholderText("Search")
        self.text_edit.resize(400, 30)
        self.text_edit.hide()
        self.layout.addWidget(self.text_edit)
        if geometry:
            self.setGeometry(*geometry)

    def enterEvent(self, event):
        self.text_edit.show()
        self.timer.stop()

    def leaveEvent(self, event):
        if self.text_edit.text() == "":
            self.timer.start(800)


class ExtensionIcon(QLabel):
    clicked = pyqtSignal()

    def __init__(self, parent=None, geometry: tuple = ()):
        super().__init__(parent)
        self.setScaledContents(True)
        self.timer = QTimer()
        self.timer.timeout.connect(lambda: self.set_visible(False))
        self.is_visible = False
        self.ableToTrigger = True
        if geometry:
            self.setGeometry(*geometry)

    def enterEvent(self, event):
        self.setPixmap(QPixmap("icons/bar_pushed.png"))
        self.set_visible(True)
        self.timer.stop()

    def leaveEvent(self, event):
        self.setPixmap(QPixmap("icons/bar.png"))
        self.timer.start(800)

    def mousePressEvent(self, event):
        event.ignore()  # 不处理该事件， 使其能传递到父窗口

    def mouseMoveEvent(self, event):
        self.ableToTrigger = False  # 拖动时禁用触发， 只有静态点击时触发
        event.ignore()

    def mouseReleaseEvent(self, event):
        if self.ableToTrigger:
            self.clicked.emit()
        self.ableToTrigger = True

    def set_visible(self, a0: bool):
        self.is_visible = a0
        if not a0:
            self.setPixmap(QPixmap())


class MyScrollArea(QWidget):
    double_clicked = pyqtSignal(int)

    def __init__(self, parent=None, widgets=None, geometry=()):
        super().__init__(parent)
        # self.setStyleSheet("background-color: transparent;")
        self.doubleClickWaiting = False
        self.doubleClickTimer = QTimer()
        self.doubleClickTimer.timeout.connect(lambda: self.setDoubleClickWaiting(False))
        if geometry:
            self.setGeometry(*geometry)
        self.setStyleSheet("""background-color: rgba(100, 100, 100, 50);
         border: none; 
         border-radius: 5px;""")
        vbox = QVBoxLayout()
        if widgets is None:
            return
        for i, w in enumerate(widgets):
            w.index = i  # 这样写可能不够标准， 应该尽可能避免在外部修改属性
            vbox.addWidget(w)
            line = QFrame()
            line.setFrameShape(QFrame.HLine)
            line.setStyleSheet('background-color: rgba(255, 255, 255, 40)')
            line.setFrameShadow(QFrame.Sunken)
            vbox.addWidget(line)
        inner_widget = QWidget()
        inner_widget.setLayout(vbox)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(inner_widget)
        vscrollbar = scroll_area.verticalScrollBar()
        vscrollbar.setStyleSheet("""
            QScrollBar:vertical {
            background: rgba(200, 200, 200, 20);
            width: 10px;
            margin: 2px 0 2px 0;
            border-radius: 3px;
            }

            QScrollBar::handle:vertical {
                background: rgba(255, 255, 255, 100);
                min-height: 30px;
                border-radius: 4px;
            }

            QScrollBar::add-line:vertical {
                border: none;
                background: none;
                height: 0px;
                subcontrol-position: bottom;
                subcontrol-origin: margin;
            }

            QScrollBar::sub-line:vertical {
                border: none;
                background: none;
                height: 0px;
                subcontrol-position: top;
                subcontrol-origin: margin;
            }

            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
            
            QScrollBar::sub-page:vertical {
            border-radius: 10px;
            background-color: rgba(0,0,0,0%);
            }
        """)
        hscrollbar = scroll_area.horizontalScrollBar()
        hscrollbar.setStyleSheet("""
                            QScrollBar:horizontal {
            background: rgba(200, 200, 200, 20);
            height: 10px;
            margin: 0 2px 0 2px;
            border-radius: 4px;
        }
        
        QScrollBar::handle:horizontal {
            background: rgba(255, 255, 255, 100);
            min-width: 20px;
            border-radius: 4px;
        }
        
        QScrollBar::add-line:horizontal {
            border: none;
            background: none;
        }
        
        QScrollBar::sub-line:horizontal {
            border: none;
            background: none;
        }
        
        QScrollBar::add-page:horizontal {
            border-radius: 10px;
            background-color: rgba(0,0,0,0%);
        }
        
        QScrollBar::sub-page:horizontal {
            border-radius: 10px;
            background-color: rgba(0,0,0,0%);
        }
                """)
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(scroll_area)
        self.setLayout(main_layout)

    def setDoubleClickWaiting(self, a0: bool):
        self.doubleClickWaiting = a0

    def mouseReleaseEvent(self, event):
        child = self.childAt(event.pos())
        name = type(child).__name__
        if child and name in {"MyContentLabel", "MyTextLabel"}:  #
            if name == "MyTextLabel":
                child = child.parent()
            self.doubleClickTimer.start(300)

            flag = True
            if self.doubleClickWaiting:
                flag = False
                self.double_clicked.emit(child.index)
            self.doubleClickWaiting = flag


class MyResultWidget(QWidget):
    def __init__(self, parent=None, result: dict = None, index=None):
        super().__init__(parent)
        self.index = index
        if result is None:
            return
        hbox = QHBoxLayout()
        widths = [700, 100, 170, 40]
        result['album_name'] = f"《{result['album_name']}》"
        for i, key in enumerate(['song_name', 'artist_tag', 'album_name', 'duration']):
            text_label = MyTextLabel(None, result[key])
            text_label.resize(widths[i], 30)

            hbox.addWidget(text_label)
        self.setLayout(hbox)


class MyContentLabel(QLabel):
    def __init__(self, parent=None, text="", index=None):
        super().__init__(parent)
        self.index = index
        self.setText(text)
        self.setStyleSheet("color: white; background-color: transparent;")


class MyProgressBar(QWidget):
    reach_end = pyqtSignal()
    time_set = pyqtSignal(float)

    def __init__(self, parent, value=678):
        super().__init__(parent)
        self.resize(QSize(600, 15))
        self.current_label = DurationLabel(self, duration=0, geometry=(0, 0, 45, 15), font_size=7)
        self.duration_label = DurationLabel(self, duration=value, geometry=(self.width() - 45, 0, 45, 15), font_size=7)
        self.slider = MyHorizontalSlider(self)
        self.slider.resize(QSize(self.width() - 2*45, 10))
        self.slider.move(45, 0)
        self.slider.valueChanged.connect(
            lambda v: self.current_label.setDuration(self.duration_label.duration * v / self.slider.maximum()))  #

        self.slider.value_set.connect(self._on_value_set)

    def _on_value_set(self, value):
        self.time_set.emit(value/self.slider.maximum()*self.duration_label.duration)

    def bar_time(self):
        return self.slider.value()/self.slider.maximum()*self.duration_label.duration

    def current(self) -> float:
        return self.current_label.duration

    def set_duration(self, duration: float):
        self.duration_label.setDuration(duration)

    def set_current(self, current: float):
        if not self.slider.pressing:  # 拖动时
            self.current_label.setDuration(current)
            self.slider.setValue(int(self.current_label.duration/self.duration_label.duration*self.slider.maximum()))


class Player(QLabel):
    sync_bar = pyqtSignal(int)
    reach_end = pyqtSignal()
    program_changed = pyqtSignal()

    def __init__(self, root, parent, fps=30.0, geometry=()):
        super().__init__(parent)
        self.mode = c.PLAY_MODE_SEQ
        self.rate = 1.0
        self.playing_list = []
        self.root = None
        self.setScaledContents(True)
        self.setStyleSheet("border: 2px solid white;")
        if geometry:
            self.setGeometry(*geometry)
        self.fps = None
        self.frames = None
        self.current_frame = 0
        self.current_file_idx = 0
        self.data = None
        self.playing = False
        self.name_label = MyTextLabel(self, font_size=20)
        self.name_label.resize(700, 45)
        self.bone_color = None
        self.rotation_quaternion = np.array([1, 0, 0, 0])  # Identity quaternion
        self.last_mouse_pos = None

        self.select_root(root)
        self.load(0, fps=fps)

        self.timer = QTimer()
        self.timer.timeout.connect(self.frame_forward)

        self.reset_button = MyPushButton("Reset", self, geometry=(10, self.height()-35, 45, 30), slot=self.reset_angle)

    def reset_angle(self):
        self.rotation_quaternion = np.array([1, 0, 0, 0])  # Identity quaternion
        self.update_frame()

    def set_rate(self, rate: float):
        if rate > 0:
            self.rate = rate
        if self.playing:
            self.timer.stop()
            self.timer.start(int(1000/(self.fps*self.rate)))

    def set_bone_color(self, bone_color):
        self.bone_color = bone_color

    def set_mode(self, mode):
        self.mode = mode

    def start(self):
        self.timer.start(int(1000/(self.fps*self.rate)))
        self.playing = True

    def pause(self):
        self.timer.stop()
        self.playing = False

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
            self.update_frame()
            self.last_mouse_pos = event.pos()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_mouse_pos = None

    def update_frame(self, i: int = None):
        """注： update方法并不能改变当前文件指针的位置"""
        if i is None:
            i = self.current_frame
        if i > self.frames:
            return False
        elif i == self.frames:
            i -= 1
        normal_vector = quaternion_rotate_vector(self.rotation_quaternion, np.array([0, 0, 1]))
        img = draw_skeleton(self.data[i], normal_vector, bone_color=self.bone_color)
        self.setPixmap(QPixmap(pil_image_to_qpixmap(img)))
        return True

    def frame_forward(self):
        if self.current_frame < self.frames:
            self.sync_bar.emit(self.current_frame)
            self.update_frame()
            self.current_frame += 1
        else:
            self.timer.stop()
            self.playing = False  # 播放停止
            self.reach_end.emit()

    def set_frame(self, i: int):
        if i >= self.frames:
            return False
        self.current_frame = i
        return True

    def get_frame(self, i: int = None):
        if i is None:
            i = self.current_frame
        if i >= self.frames:
            return None
        return self.data[i]

    def duration(self):
        return self.frames / self.fps

    def load(self, i=0, fps=30., rand=False):
        if i < 0 or i >= len(self.playing_list):
            return False
        name = os.path.basename(self.playing_list[i] if not rand else random.choice(self.playing_list))
        self.name_label.update_text(name[:-4])
        self.fps = fps
        self.current_file_idx = i
        path = os.path.join(self.root, name)
        if isinstance(name, str) and name.endswith('.npy') and os.path.exists(path):
            self.data = np.load(path)
            self.frames = self.data.shape[0]
        else:
            raise FileNotFoundError(f'file not found at path {path}')
        self.current_frame = 0
        self.update_frame()
        self.program_changed.emit()  # 一定在把所有自身属性更新之后发射信号（小心处理与其他线程的耦合）
        return True

    def select_root(self, root):
        self.root = root
        lst = glob(f'{root}/*.npy')
        self.playing_list = sorted(lst, key=lambda n: int(re.search(r'\\(\d+?)-', n).group(1)))

    def next(self):
        if self.mode == c. PLAY_MODE_SEQ:
            return self.load(self.current_file_idx+1)
        elif self.mode == c.PLAY_MODE_CYCLE:
            return self.load(i=self.current_file_idx)
        elif self.mode == c.PLAY_MODE_RANDOM:
            return self.load(rand=True)

    def previous(self):
        if self.mode == c.PLAY_MODE_SEQ:
            return self.load(self.current_file_idx - 1)
        elif self.mode == c.PLAY_MODE_CYCLE:
            return self.load(i=self.current_file_idx)
        elif self.mode == c.PLAY_MODE_RANDOM:
            return self.load(rand=True)


def text2icon(text: str):
    return QIcon(MyTextLabel(text=text).grab())


class CustomSwitch(QWidget):
    switched = pyqtSignal(bool)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._slider_position = 0
        self.animation = QPropertyAnimation(self, b'slider_position_property', self)
        self.animation.setDuration(100)
        self.is_on = False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)

        # Draw background
        background_rect = QRectF(0, 0, self.width(), self.height())
        radius = self.height() / 2
        painter.setBrush(QColor("gray") if 2*self._slider_position < self.width()-self.height() else QColor("#067d1d"))
        painter.drawRoundedRect(background_rect, radius, radius)

        # Draw slider
        slider_rect = QRectF(self.slider_position, 0, self.height(), self.height())
        painter.setBrush(QColor("white"))
        painter.drawEllipse(slider_rect)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_on = not self.is_on
            self.switched.emit(self.is_on)
            self.animation.setStartValue(QVariant(self.slider_position))
            self.animation.setEndValue(QVariant(self.width() - self.height() if self.is_on else 0))

            self.animation.start()
            event.ignore()

    @property
    def slider_position(self):
        return self._slider_position

    @slider_position.setter
    def slider_position(self, pos):
        self._slider_position = pos
        self.update()

    @pyqtProperty(QVariant)
    def slider_position_property(self):
        return QVariant(self.slider_position)

    @slider_position_property.setter
    def slider_position_property(self, value):
        self.slider_position = value  # .toInt()[0]


class LogoLabel(QLabel):
    def __init__(self, parent, geometry=()):
        super().__init__(parent)
        self.setScaledContents(True)
        self.setPixmap(QPixmap('icons/logo.png'))
        if geometry:
            self.setGeometry(*geometry)


class OpenCameraThread(QThread):
    camera_opened = pyqtSignal()

    def __init__(self, camera_window):
        super().__init__()
        self.camera_window = camera_window

    def run(self):
        self.camera_window.camera = cv2.VideoCapture(0)
        if self.camera_window.camera.isOpened():
            _, frame = self.camera_window.camera.read()
            height, width, _ = frame.shape
            self.camera_window.setFixedSize(width, height)
            self.camera_window.label.setGeometry(0, 0, width, height)
            self.camera_opened.emit()
        else:
            self.camera_window.camera_open_failure.emit()


def add_annotation(results, image):
    ret = False
    if results.pose_landmarks:
        color = (0, 255, 0)
        ret = True
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            # x = lm.x * image.shape[1]
            # y = lm.y * image.shape[0]
            # z = lm.z * image.shape[1]
            visibility = lm.visibility  # 获取可见性
            # 根据可见性设置不同的颜色
            if visibility < 0.5:
                color = (0, 0, 255)  # 红色，表示不可见
                ret = False
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=color, thickness=2))
    return ret


class CameraWindow(QWidget):
    info = pyqtSignal(str)
    detections_updated = pyqtSignal(dict)
    camera_open_failure = pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pose = None
        self.frame = None
        # Remove window frame
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)

        # Initialize camera
        self.camera = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.fps = 30

        # Initialize window content
        self.label = QLabel(self)
        self.label.setGeometry(0, 0, self.width(), self.height())
        self.label.setStyleSheet("background-color: gray;")
        self.close_button = CameraExitButton('', self, geometry=(self.width()-36, 0, 25, 25), icon='close', icon_size=(24, 24), slot=self.close)

        self.info_enabled = True
        self.info_timer = QTimer()
        self.info_timer.timeout.connect(self.on_info_timer_timeout)

        self.unrecognized_cnt = 0
        self.detected = False

    def on_info_timer_timeout(self):
        self.info_timer.stop()
        self.info_enabled = True

    def send_info(self, text: str):
        if self.info_enabled:
            self.info.emit(text)
            self.info_enabled = False
            self.info_timer.start(1000)

    def open_camera(self):
        self.open_camera_thread = OpenCameraThread(self)
        self.open_camera_thread.camera_opened.connect(self.start_timer)
        self.open_camera_thread.start()
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def close_camera(self):
        if self.camera is not None and self.camera.isOpened():
            self.camera.release()
            self.label.clear()
            self.label.setStyleSheet("background-color: gray;")
            self.pose.close()

    def start_timer(self):
        self.timer.start(1000 // self.fps)

    def update_frame(self):
        if self.camera is not None and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                frame = cv2.flip(frame, 1)
                results = self.pose.process(frame)
                retval = add_annotation(results, frame)
                if results.pose_landmarks:
                    self.detections_updated.emit({'bone_arrays': landmarks_to_bone_arrays(results.pose_landmarks), 'landmarks': landmarks_to_numpy(results.pose_landmarks)})
                if not retval:
                    self.unrecognized_cnt += 1
                else:
                    self.send_info("")  # info clear
                    self.unrecognized_cnt = 0
                    self.detected = True

                if self.unrecognized_cnt > 1*self.fps:
                    self.send_info("请全身站在摄像头范围内")
                    self.detected = False
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                self.frame = frame
                qimage = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimage)
                self.label.setPixmap(pixmap)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.old_pos = event.globalPos()
            self.old_top_left_pos = self.frameGeometry().topLeft()
            self.mouse_pressed = True

    def mouseMoveEvent(self, event):
        if self.mouse_pressed:
            delta = event.globalPos() - self.old_pos
            new_top_left_pos = self.old_top_left_pos + delta
            self.move(new_top_left_pos)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.mouse_pressed = False

    def enterEvent(self, event):
        self.close_button.show()

    def leaveEvent(self, event):
        self.close_button.hide()

    def closeEvent(self, event):
        self.close_camera()
        event.accept()

