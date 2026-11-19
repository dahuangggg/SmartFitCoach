# SmartFitCoach

[中文](#中文) | [English](#english)

![GuideLine](https://user-images.githubusercontent.com/97234929/231226227-1e9d7fa6-a367-4f37-99b7-f75ce7ad1cdd.png)

---

## 中文

基于计算机视觉的健身动作指导桌面应用：播放标准动作的 3D 骨架，并与摄像头中的实时姿态进行比对打分。

### 状态

早期项目，不再维护。

### 功能

- 播放 `videos/session1` 中预处理好的 `.npy` 骨架序列（3D 关键点），可鼠标拖拽旋转视角，`Reset` 复位
- 播放控制：播放/暂停、上一个/下一个、进度条拖动、0.5x–2.0x 倍速、自动播放开关、播放列表（双击切换）
- 播放模式：列表顺序、随机、单曲循环
- 打开摄像头后用 MediaPipe Pose 检测实时姿态，经 Kabsch 对齐后按骨骼方向余弦相似度为标准骨架着色，并显示总体相似度百分比；未检测到全身时提示“请全身站在摄像头范围内”
- 无边框、半透明、置顶窗口，鼠标移入时显示控件

### 技术栈

Python、PyQt5、MediaPipe Pose、OpenCV、NumPy、SciPy、Pillow

### 本地运行

```bash
git clone https://github.com/dahuangggg/SmartFitCoach.git
cd SmartFitCoach
pip install -r requirements.txt
pip install scipy   # 代码中使用但未列入 requirements.txt
python app.py
```

注意：

- `requirements.txt` 中的 `mediapipe==0.9.2.1` 已无法从 PyPI 安装，需改用仍提供 `mp.solutions` 的 0.10.x 版本（如 `0.10.14`）。
- 骨架文件需命名为 `序号-名称.npy`（如 `1-side step+clap.npy`）；播放列表排序的正则按 Windows 路径分隔符 `\` 编写。
- 新视频需先用 `process_video.py` 提取关键点生成 `.npy`；该脚本的输入目录写死为 `videos/session3`，且导入了仓库中不存在的 `window_capture` 模块，使用前需自行修改。

### 目录结构

```
app.py              主窗口与组件连接
my_widgets.py       播放器、摄像头窗口及自定义控件
cosine_distance.py  姿态对齐（Kabsch）与相似度计算
detect_utils.py     关键点与骨骼向量工具
plot_utils.py       骨架绘制与图像转换
process_video.py    从视频提取关键点并保存为 .npy
constants.py        状态与播放模式常量
test.py             骨架查看器示例 / 四元数工具
icons/              界面图标
videos/session1/    示例动作视频与骨架数据
```

---

## English

A computer-vision fitness coaching desktop app: it plays a reference exercise as a 3D skeleton and scores your live webcam pose against it.

### Status

Early project, no longer actively maintained.

### Features

- Plays pre-processed `.npy` skeleton sequences (3D landmarks) from `videos/session1`; drag to rotate the view, `Reset` to restore it
- Playback controls: play/pause, previous/next, seekable progress bar, 0.5x–2.0x speed, autoplay switch, playlist (double-click to switch)
- Play modes: sequential, random, single-item loop
- With the camera on, MediaPipe Pose detects your pose; after Kabsch alignment, the reference skeleton's bones are colored by per-bone cosine similarity and an overall similarity percentage is shown. A hint appears when your full body isn't in frame
- Frameless, translucent, always-on-top window; controls show on mouse hover

### Tech Stack

Python, PyQt5, MediaPipe Pose, OpenCV, NumPy, SciPy, Pillow

### Getting Started

```bash
git clone https://github.com/dahuangggg/SmartFitCoach.git
cd SmartFitCoach
pip install -r requirements.txt
pip install scipy   # imported by the code but not listed in requirements.txt
python app.py
```

Notes:

- `mediapipe==0.9.2.1` in `requirements.txt` is no longer available on PyPI; use a 0.10.x release that still ships `mp.solutions` (e.g. `0.10.14`).
- Skeleton files must be named `<index>-<name>.npy` (e.g. `1-side step+clap.npy`); the playlist sort regex is written for Windows `\` path separators.
- To add a video, run `process_video.py` to extract landmarks into `.npy`. The script's input folder is hard-coded to `videos/session3` and it imports a `window_capture` module that is not in the repo, so edit it before use.

### Directory Structure

```
app.py              main window and component wiring
my_widgets.py       player, camera window and custom widgets
cosine_distance.py  pose alignment (Kabsch) and similarity
detect_utils.py     landmark and bone-vector helpers
plot_utils.py       skeleton drawing and image conversion
process_video.py    extract landmarks from videos into .npy
constants.py        state and play-mode constants
test.py             skeleton viewer demo / quaternion helpers
icons/              UI icons
videos/session1/    sample exercise video and skeleton data
```
