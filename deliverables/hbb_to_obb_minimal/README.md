# HBB 转 OBB 数据集处理小交付包

这个文件夹是独立的小交付包，只负责完成 UTE 数据集中“水平框 HBB 标注补充角度信息，生成旋转框 OBB 标注”的数据处理部分。它不依赖主项目路径，拿到这个文件夹后，只要按下面步骤下载数据并放到指定目录，就可以独立运行。

## 1. 功能说明

输入 UTE 数据集中的 `pixel.csv` 水平框标注，按同一车辆跨帧中心点的运动方向估计角度 `theta`，输出带旋转框信息的新 CSV：

- `theta`：弧度制车辆方向角；
- `theta_deg`：角度制方向角；
- `theta_conf`：角度置信度，轨迹直接估计为 1.0，前后填补为 0.55；
- `obb_x1` ~ `obb_y4`：旋转框四个角点；
- `obb_w` / `obb_h` / `obb_area`：旋转框几何尺寸和面积；
- overlay 图片：抽取 3 帧，把 OBB 画回视频帧，检查标注是否与车辆像素位置对应。

## 2. 文件夹结构

拿到交付包后，建议保持下面结构：

```text
hbb_to_obb_minimal/
├── prepare_hbb_to_obb.py
├── config_ute_sample.json
├── requirements.txt
├── README.md
├── data/
│   └── UTE/
│       └── datasets/
│           ├── XAM-N/
│           │   ├── XAM-N-5/
│           │   └── XAM-N-6/
│           └── PKDD/
│               └── PKDD-8/
└── outputs/              # 运行后自动生成
```

`data/` 和 `outputs/` 不需要提前创建完整内容，按第 3 节下载数据后会有 `data/UTE/datasets/...`，运行脚本后会自动生成 `outputs/`。

## 3. 下载 UTE 数据集

数据来自 Ubiquitous Traffic Eye 官方仓库：

```text
https://github.com/Ruyi-Feng/Ubiquitous-Traffic-Eye
```

### 方式 A：推荐，直接下载指定分支

在 `hbb_to_obb_minimal/` 目录下执行：

```bash
mkdir -p data
git clone -b UTE --depth 1 https://github.com/Ruyi-Feng/Ubiquitous-Traffic-Eye.git data/UTE
```

下载完成后，至少应存在这些文件：

```text
data/UTE/datasets/XAM-N/XAM-N-5/pixel.csv
data/UTE/datasets/XAM-N/XAM-N-5/frenet.csv
data/UTE/datasets/XAM-N/XAM-N-5/sample video.mp4

data/UTE/datasets/XAM-N/XAM-N-6/pixel.csv
data/UTE/datasets/XAM-N/XAM-N-6/frenet.csv
data/UTE/datasets/XAM-N/XAM-N-6/sample video.mp4

data/UTE/datasets/PKDD/PKDD-8/pixel.csv
data/UTE/datasets/PKDD/PKDD-8/frenet.csv
data/UTE/datasets/PKDD/PKDD-8/sample video.mp4
```

可以用下面命令检查：

```bash
ls "data/UTE/datasets/XAM-N/XAM-N-5/pixel.csv"
ls "data/UTE/datasets/XAM-N/XAM-N-5/sample video.mp4"
```

### 方式 B：网页手动下载

如果不能使用 `git clone`，也可以打开：

```text
https://github.com/Ruyi-Feng/Ubiquitous-Traffic-Eye/tree/UTE/datasets
```

手动下载并放成下面的相对路径：

```text
hbb_to_obb_minimal/data/UTE/datasets/XAM-N/XAM-N-5/
hbb_to_obb_minimal/data/UTE/datasets/XAM-N/XAM-N-6/
hbb_to_obb_minimal/data/UTE/datasets/PKDD/PKDD-8/
```

路径必须和 `config_ute_sample.json` 中的 `root` 字段一致。如果数据放在别的位置，可以直接修改 `config_ute_sample.json` 里的 `root`。

## 4. 安装环境

在 `hbb_to_obb_minimal/` 目录下执行：

```bash
pip install -r requirements.txt
```

依赖很少，只需要：

- `numpy`
- `opencv-python`

如果已经在主项目环境或 Anaconda 环境里安装过这两个包，可以跳过安装。

## 5. 运行命令

### 5.1 先小样本试跑

建议先用 XAM-N-5 跑 5000 行，确认环境、数据路径和视频读取都正常：

```bash
python prepare_hbb_to_obb.py --config config_ute_sample.json --datasets xamn5 --max-rows 5000
```

看到类似输出即表示成功：

```text
[OBB] loading XAM-N-5 pixel table: ...
[OBB] writing OBB csv: outputs/processed/xamn5_pixel_obb.csv
[OBB] wrote overlay: outputs/figures/xamn5_obb_overlay_f205.jpg
[OBB] XAM-N-5 direct theta rate: 1.000
[OBB] summary saved to outputs/reports/obb_summary.json
```

### 5.2 完整处理 XAM-N-5

```bash
python prepare_hbb_to_obb.py --config config_ute_sample.json --datasets xamn5
```

### 5.3 完整处理 XAM-N-5、XAM-N-6、PKDD-8

```bash
python prepare_hbb_to_obb.py --config config_ute_sample.json --datasets xamn5 xamn6 pkdd8
```

## 6. 输出结果

运行后会生成：

```text
outputs/
├── processed/
│   ├── xamn5_pixel_obb.csv
│   ├── xamn6_pixel_obb.csv
│   └── pkdd8_pixel_obb.csv
├── figures/
│   └── *_obb_overlay_*.jpg
└── reports/
    └── obb_summary.json
```

其中：

- `outputs/processed/*_pixel_obb.csv` 是最终 OBB 标注表；
- `outputs/figures/*_obb_overlay_*.jpg` 是抽帧可视化；
- `outputs/reports/obb_summary.json` 记录处理行数、车辆数、直接估角比例和输出文件路径。

## 7. 如何确认结果正确

打开 `outputs/processed/xamn5_pixel_obb.csv`，表头应包含：

```text
frame,time_s,vehicle_id,x,y,w,h,class,cx,cy,obb_w,obb_h,obb_area,obb_aspect_ratio,theta,theta_deg,theta_conf,obb_x1,obb_y1,...,obb_x4,obb_y4
```

再打开 `outputs/figures/` 下的 overlay 图片，能看到黄色旋转框画在车辆位置上，红点是车辆中心点。由于视频可能是裁剪或降采样版本，overlay 仅用于人工检查显示效果；真正用于后续实验的是 CSV 中逐行对应的 OBB 坐标和角度。

## 8. 方法解释

1. 读取 `pixel.csv` 中每一行车辆水平框 `(x, y, w, h)`；
2. 计算车辆中心点 `(cx, cy)`；
3. 对同一辆车按帧号排序，在前后若干帧内找位移最大的稳定片段；
4. 用 `atan2(dy, dx)` 得到车辆运动方向角；
5. 对静止或短时缺失角度的车辆，用同一车辆前后有效角度填补；
6. 用车辆中心点、长边、短边和方向角计算 OBB 四个角点；
7. 保持输出 CSV 与原始 `pixel.csv` 一行一车辆对应，方便后续与 `frenet.csv` 里的速度、加速度、车道等运动学字段合并。

## 9. 汇报时可以这样说

数据集原始标注是水平框 HBB。我先完成了 HBB 到 OBB 的数据处理模块：不是重新训练检测器，而是利用 `pixel.csv` 中同一车辆跨帧中心点轨迹估计车辆方向角，再把每一行水平框转换成带 `theta` 和四角点坐标的旋转框 OBB。这样生成的 OBB 仍然和原始 pixel 表逐行对应，车辆 ID、时间、车道、速度、加速度等信息可以继续对齐，后续可以直接用于交通状态识别和预测实验。

