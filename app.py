import json
import sys

import numpy as np
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel
from PyQt5.QtGui import QPainter, QColor, QIcon, QCursor
from PyQt5.QtCore import Qt, QPoint, QSize, QCoreApplication

from my_widgets import MyPushButton, ChangeableButton, MyVerticalSlider, MyHorizontalSlider, MyTextLabel, DurationLabel, \
    SearchBox, ExtensionIcon, ExitButton, VolumeControl, MyScrollArea, MyResultWidget, MyProgressBar, Player, \
    CustomSwitch, LogoLabel, CameraWindow, MyContentLabel

import constants as c
from cosine_distance import *
from plot_utils import get_color


# TODO: 菜单， 换肤中心， 分类， 开始前准备时间， 登录系统， 搜索课程， 导入与导出视频， 音乐健身, 收藏， 语音播报， 即时描述， 解析视频（拖拽）


class TransparentWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.camera_window = None
        self.camera_button = None
        self.progress_bar = None
        self.player = None
        self.showSearchResults = False
        self.extension_icon = None
        self.extension_mark = None
        self.search_box = None
        self.current_label = None
        self.duration_label = None
        self.volume_bar = None
        self.buttons = None

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setGeometry(100, 100, 1200, 1000)

        self.draggable = True
        self.state = c.state_brief
        self.offset = QPoint()
        self.setMouseTracking(True)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 220))
        painter.drawRect(self.rect())

    def initUI(self):
        exit_button = ExitButton('', self, geometry=(self.width() - 35, 5, 25, 25), slot=self.close, icon='close')
        play_button = ChangeableButton('', self, geometry=(self.width() / 2, self.height() - 95, 50, 50),
                                       icons=("play", "pause"), icon_size=(40, 40))
        state_button = ChangeableButton('', self, geometry=(self.width() / 2 - 130, self.height() - 88, 50, 38),
                                        icons=("list_cycle", "random_play", "one_cycle"), icon_size=(32, 24))
        state_button.state_changed.connect(lambda i: self.player.set_mode([c.PLAY_MODE_SEQ, c.PLAY_MODE_RANDOM, c.PLAY_MODE_CYCLE][i]))
        self.camera_button = ChangeableButton('', self, icons=('camera_off', 'camera'), geometry=(self.width() / 2 + 128, self.height() - 92, 50, 42), icon_size=(50, 40), tips=['打开摄像头', '关闭摄像头'])
        self.camera_window = CameraWindow()
        self.camera_button.state_changed.connect(lambda i: self.camera_window.open_camera() or self.camera_window.show() if i else self.camera_window.close_camera() or self.player.set_bone_color(None) or self.camera_window.hide())
        self.camera_window.camera_open_failure.connect(lambda: self.camera_button.set_state(0))  # 打开摄像头失败则图标自动变回未打开状态
        # self.favourite_button = ChangeableButton('', self, geometry=(self.width() / 2 + 130, 36, 30, 30), icons=("favourite_noclick", "favourite"), icon_size=(24, 24))
        # collect_button = ChangeableButton('', self, geometry=(self.width() / 2 + 130, self.height() - 84, 30, 30),
        #                                   icons=("collections_unclicked", "collections"), icon_size=(24, 24))
        # self.comments_button = MyPushButton('', self, geometry=(self.width() / 2 + 130, 32, 50, 38), icon="comments", icon_size=(32, 24))

        list_button = MyPushButton('', self, geometry=(self.width() - 50, self.height() - 36, 30, 30), icon="list",
                                   icon_size=(24, 18), tips="播放列表")
        # self.volume_bar = MyVerticalSlider(self, geometry=(self.width() - 72, self.height() - 106, 14, 75))
        self.player = Player('videos/session1', self, fps=30, geometry=(50, 60, 700, 700))
        self.player.show()

        self.info_label = MyTextLabel(self, font_size=20)
        self.info_label.setGeometry(400, 800, 700, 45)
        self.info_label.show()

        self.camera_window.info.connect(self.info_label.update_text)
        self.camera_window.detections_updated.connect(self.calculate_similarity)

        rate_button = ChangeableButton('', self, geometry=(self.width() / 2 + 400, self.height() - 42, 50, 36), icons=('1.0x', '1.25x', '1.5x', '2.0x', '0.5x', '0.75x'), icon_size=(50, 36))
        rate_button.state_changed.connect(lambda i: self.player.set_rate([1.0, 1.25, 1.5, 2.0, 0.5, 0.75][i]))

        previous_button = MyPushButton('', self, geometry=(self.width() / 2 - 60, self.height() - 95, 50, 50),
                                       icon="previous", icon_size=(30, 30), slot=self.player.previous)
        next_button = MyPushButton('', self, geometry=(self.width() / 2 + 60, self.height() - 95, 50, 50), icon="next",
                                   icon_size=(30, 30), slot=self.player.next)

        autoplay_switch = CustomSwitch(self)
        autoplay_switch.setFixedSize(40, 20)
        autoplay_switch.move(int(self.width() / 2 + 350), self.height() - 38)
        autoplay_switch.setToolTip("自动播放")
        logo = LogoLabel(self, geometry=(400, 12, 360, 28))

        volume_control = VolumeControl(self)
        volume_control.move(self.width() - 80, self.height() - 110)

        self.search_box = SearchBox(self, geometry=(15, 15, 280, 30))
        self.search_box.text_edit.returnPressed.connect(self.on_edit_finished)

        self.player.program_changed.connect(lambda: self.progress_bar.set_duration(self.player.duration()) or self.progress_bar.set_current(0))  # 换节目（节目当前帧归零）-》更新进度条（归零、更新总时长）
        self.progress_bar = MyProgressBar(self)
        self.progress_bar.set_duration(self.player.duration())  # 设置进度条初始时长
        self.progress_bar.move(int(self.width()/2-self.progress_bar.width()/2), self.height()-self.progress_bar.height()-20)

        self.progress_bar.time_set.connect(lambda t: self.player.set_frame(int(t*self.player.fps)))  # 拖动进度条最后-》设置播放器的当前帧
        self.progress_bar.slider.pressed.connect(self.player.timer.stop)  # 拖动时停止播放
        self.progress_bar.slider.released.connect(lambda: self.player.playing and self.player.start())  # and: 前真而后, or: 前假而后
        self.progress_bar.slider.valueChanged.connect(lambda v: self.progress_bar.slider.pressing and self.player.update_frame(int(self.progress_bar.bar_time()*self.player.fps)))  #

        self.player.reach_end.connect(lambda: play_button.set_state(0) if not autoplay_switch.is_on else self.player.next() and self.player.start())  # 进度条触底-》按键样式变为暂停样式 or 下一个视频

        self.player.sync_bar.connect(lambda f: self.progress_bar.set_current(f / self.player.fps))  # (在拖动时应断开此链接)

        play_button.state_changed.connect(lambda i: self.player.start() if i == 1 else self.player.pause())  # 按键状态-》播放状态

        widgets = [MyContentLabel(None, name.lstrip(self.player.root).strip('.npy')) for name in self.player.playing_list]
        program_scroller = MyScrollArea(self, widgets, geometry=(750, 70, 380, 710))
        program_scroller.double_clicked.connect(self.result_chosen)

        # widgets2 = [MyContentLabel(None, name.lstrip(self.player.root).strip('.npy')) for name in self.player.playing_list]
        # session_scroller = MyScrollArea(self, widgets2, geometry=(24, self.height()-240, 240, 200))

        self.buttons = [exit_button, play_button, previous_button, next_button, autoplay_switch, logo, list_button,
                        self.progress_bar, volume_control, rate_button, state_button, self.camera_button,
                        self.search_box, program_scroller]

    def result_chosen(self, index):
        print(f"chose {index}") if self.player.load(index) else print(f'failed to load {self.player.playing_list[index]}')

    def calculate_similarity(self, results):
        frame = self.player.get_frame()
        detected_bone_arrays = results['bone_arrays']
        if frame is not None:
            arr_standard = numpy_to_bone_arrays(frame)

            M = kabsch(detected_bone_arrays, arr_standard)
            detected_bone_arrays = np.dot(detected_bone_arrays, M)

            result = np.sum(arr_standard*detected_bone_arrays, axis=1)
            ys = interpolation_function(result)
            colors = [get_color(y) for y in ys]
            self.player.set_bone_color(colors)
            if not self.player.playing:
                self.player.update_frame()
            if self.camera_window.detected:
                score = np.mean(ys) * 100
                self.info_label.update_text(f"{score:.2f}%")

    def enterEvent(self, event):
        if not self.buttons:
            self.initUI()
        for button in self.buttons:
            button.show()

    def on_edit_finished(self):
        print(self.search_box.text_edit.text())
        self.search_box.text_edit.clear()
        self.showSearchResults = True

    def leaveEvent(self, event):
        if self.buttons:
            for button in self.buttons:
                button.hide()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.draggable:
            self.offset = event.pos()
        if QCursor().pos() - self.pos() not in self.search_box.rect():
            self.search_box.text_edit.hide()
        child = self.childAt(event.pos())
        if child and child.inherits("QAbstractButton"):
            self.draggable = False
            # print("draggable set False")

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self.draggable:
            self.move(self.pos() + event.pos() - self.offset)

    def mouseReleaseEvent(self, event):
        self.draggable = True

    def closeEvent(self, event):
        QApplication.quit()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TransparentWindow()
    window.show()
    sys.exit(app.exec_())
