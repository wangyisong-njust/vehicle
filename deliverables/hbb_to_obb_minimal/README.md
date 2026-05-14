# HBB 转 OBB 数据集处理小交付包

这个文件夹只保留“水平框标注补充角度信息并生成旋转框”的最小代码，便于单独给老师或合作者演示数据集处理部分。

## 1. 这部分已经做好的内容

输入 UTE 数据集中的 `pixel.csv` 水平框标注，按车辆轨迹中心点的运动方向估计角度 `theta`，并输出带旋转框信息的新 CSV：

- `theta`：弧度制车辆方向角；
- `theta_deg`：角度制方向角；
- `theta_conf`：角度置信度，轨迹直接估计为 1.0，前后填补为 0.55；
- `obb_x1` ~ `obb_y4`：旋转框四个角点；
- `obb_w` / `obb_h` / `obb_area`：旋转框几何尺寸和面积；
- overlay 图片：抽取 3 帧，把 OBB 画回视频帧检查是否与车辆像素位置对应。

## 2. 环境安装

```bash
cd deliverables/hbb_to_obb_minimal
pip install -r requirements.txt
```

如果已经使用项目主环境 `vehicle_ute`，可以直接运行，不需要重复安装。

## 3. 直接运行示例

在项目根目录已有 `data_check/` 数据时，直接执行：

```bash
cd deliverables/hbb_to_obb_minimal
python prepare_hbb_to_obb.py --config config_ute_sample.json --datasets xamn5
```

也可以一次跑三个数据集：

```bash
python prepare_hbb_to_obb.py --config config_ute_sample.json --datasets xamn5 xamn6 pkdd8
```

如果只是想先确认环境和流程能跑通，可以加 `--max-rows 5000` 做小样本试跑：

```bash
python prepare_hbb_to_obb.py --config config_ute_sample.json --datasets xamn5 --max-rows 5000
```

## 4. 输出位置

默认输出在当前小交付包内部：

```text
deliverables/hbb_to_obb_minimal/outputs/
├── processed/
│   └── xamn5_pixel_obb.csv
├── figures/
│   ├── xamn5_obb_overlay_f2000.jpg
│   ├── xamn5_obb_overlay_f4000.jpg
│   └── xamn5_obb_overlay_f6000.jpg
└── reports/
    └── obb_summary.json
```

## 5. 方法解释

1. 读取 `pixel.csv` 中每一行车辆水平框 `(x, y, w, h)`；
2. 计算车辆中心点 `(cx, cy)`；
3. 对同一辆车按帧号排序，在前后若干帧内找位移最大的稳定片段；
4. 用 `atan2(dy, dx)` 得到车辆运动方向角；
5. 对静止或短时缺失角度的车辆，用同一车辆前后有效角度填补；
6. 用车辆中心点、长边、短边和方向角计算 OBB 四个角点；
7. 保持输出 CSV 与原始 `pixel.csv` 一行一车辆对应，方便后续与 frenet / 运动学字段合并。

## 6. 汇报时可以这样说

数据集原始标注是水平框 HBB，我没有重新训练检测器，而是利用 `pixel.csv` 中同一车辆跨帧中心点轨迹估计车辆方向角，再把每一行水平框转换成带 `theta` 和四角点坐标的旋转框 OBB。这样做的好处是：第一，旋转框仍然和原始 pixel 表逐行对应；第二，车辆 ID、时间、速度、加速度、车道等运动学信息不会丢；第三，可以直接抽帧画 overlay 检查 OBB 是否贴合车辆像素位置。
