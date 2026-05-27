# 第二部分交付：特征提取与交通状态识别独立包

> **本包是项目交付的第二部分**。承接第一部分 `hbb_to_obb_minimal/` 输出的 OBB 标注表，完成
> "滑窗特征提取 → 4 通道空间张量产出 → 自动状态标签生成 → XGBoost 当前状态识别"四条完整链路。
> 本包可独立运行，所有路径相对包根目录，复现成本极低。

## 1. 本包做什么

1. **从 OBB 标注表与 frenet 轨迹表提取滑窗交通特征**
   - 宏观量：速度、密度、流量、车头时距
   - 微观博弈量：加速度干扰、MGTI
   - 空间量：HBB / HF-GO 网格占有率（高保真多边形裁剪解析积分）
   - OBB 几何量：方向波动指数、theta 置信度

2. **同时产出每窗一个 4 通道 4×12 网格张量**（`.npz`）
   - 通道 0：OBB 高保真占有率
   - 通道 1：HBB 轴对齐占有率
   - 通道 2：单元面积加权 sin(θ)
   - 通道 3：单元面积加权 cos(θ)

3. **自动生成四类交通状态标签**（畅通 / 缓行 / 拥挤 / 堵塞）
   - K-Means 聚类（在速度比 / 密度 / 变道率 / 方向波动指数上）
   - 物理顺序兜底：若聚类不满足"速度递减 + 密度递增 + 占有率递增"则用风险分位数兜底
   - 时间维滑窗多数投票平滑

4. **训练 XGBoost 当前状态分类器并输出**：
   - 分层 70/30 划分
   - 类别加权样本
   - 完整指标（Accuracy、Macro-F1、Weighted-F1、混淆矩阵、特征重要性）
   - 配套可视化（混淆矩阵图、特征重要性条形图、状态分布图）

## 2. 文件夹结构

```
delivery2/
├── README.md                                ← 本文档
├── requirements.txt                         ← Python 依赖
├── configs/
│   └── datasets.json                        ← 数据集与超参数配置
├── src/
│   ├── features.py                          ← 特征提取与网格张量构造
│   ├── state_labels.py                      ← 4 类状态自动标签
│   ├── classifier.py                        ← XGBoost 训练 / 评估
│   └── plotting.py                          ← matplotlib 可视化辅助
├── scripts/
│   ├── 01_extract_features.py               ← 步骤 1：抽特征 + 网格张量
│   ├── 02_build_state_labels.py             ← 步骤 2：生成状态标签
│   ├── 03_run_classification.py             ← 步骤 3：XGBoost 训练评估
│   └── run_all.sh                           ← 一键复现脚本
├── data/UTE/datasets/                       ← UTE 数据放置位置（自行下载）
├── inputs/                                  ← 上一部分输出的 OBB 表放置位置
└── outputs/                                 ← 运行后自动产生
    ├── features/
    │   ├── {ds}_windows.csv
    │   ├── {ds}_grid_tensors.npz
    │   └── all_windows.csv
    ├── labels/state_labels.csv
    ├── reports/
    │   ├── feature_summary.json
    │   ├── labels_summary.json
    │   └── classification_results.json
    └── figures/
        ├── state_distribution.png
        ├── confusion_matrix.png
        └── feature_importance.png
```

## 3. 准备阶段

### 3.1 准备 UTE 原始数据（来自第一部分交付包的同源）

```bash
mkdir -p data
git clone -b UTE --depth 1 https://github.com/Ruyi-Feng/Ubiquitous-Traffic-Eye.git data/UTE
```

下载完成后应至少存在：

```text
data/UTE/datasets/XAM-N/XAM-N-6/frenet.csv
data/UTE/datasets/XAM-N/XAM-N-6/sample video.mp4
data/UTE/datasets/XAM-N/XAM-N-5/frenet.csv
data/UTE/datasets/XAM-N/XAM-N-5/sample video.mp4
data/UTE/datasets/PKDD/PKDD-8/frenet.csv
data/UTE/datasets/PKDD/PKDD-8/sample video.mp4
```

### 3.2 把第一部分交付包产出的 OBB 表复制进 `inputs/`

第一部分交付包（`hbb_to_obb_minimal/`）运行后会产出 3 个 OBB pixel 表，把它们复制到本包 `inputs/`：

```bash
mkdir -p inputs
cp /path/to/hbb_to_obb_minimal/outputs/processed/xamn6_pixel_obb.csv inputs/
cp /path/to/hbb_to_obb_minimal/outputs/processed/xamn5_pixel_obb.csv inputs/
cp /path/to/hbb_to_obb_minimal/outputs/processed/pkdd8_pixel_obb.csv inputs/
```

检查：

```bash
ls -lh inputs/
# 应看到 xamn5_pixel_obb.csv  xamn6_pixel_obb.csv  pkdd8_pixel_obb.csv
```

### 3.3 安装 Python 环境

```bash
pip install -r requirements.txt
```

依赖很轻：`numpy`、`opencv-python`、`scikit-learn`、`xgboost`、`matplotlib`。Python 3.10+ 推荐。

## 4. 一键复现

进入包根目录后：

```bash
bash scripts/run_all.sh
```

或分步运行：

```bash
python scripts/01_extract_features.py
python scripts/02_build_state_labels.py
python scripts/03_run_classification.py
```

预期输出（截选关键日志）：

```text
[FEATURE] loading XAM-N-6
[FEATURE] wrote 324 windows -> outputs/features/xamn6_windows.csv
[FEATURE] wrote tensors (324, 4, 4, 12) -> outputs/features/xamn6_grid_tensors.npz
...
[FEATURE] summary -> outputs/reports/feature_summary.json

[LABEL] wrote 1644 labels -> outputs/labels/state_labels.csv
[LABEL] summary -> outputs/reports/labels_summary.json
[LABEL] figure -> outputs/figures/state_distribution.png

[CLS] xamn6: 324 windows, label distribution = [82, 81, 80, 81]
[CLS] metrics -> outputs/reports/classification_results.json
[CLS] Macro-F1=0.9590, Accuracy=0.9588
[CLS] figures -> outputs/figures/confusion_matrix.png, outputs/figures/feature_importance.png
```

## 5. 关键复现指标参考

如果数据 / 环境正确，主要数字应接近：

| 项目 | 参考值 |
|---|---:|
| XAM-N-6 滑窗数 | 324 |
| XAM-N-5 滑窗数 | 261 |
| PKDD-8 滑窗数 | 1059 |
| 4 通道网格张量形状（XAM-N-6） | (324, 4, 4, 12) |
| XAM-N-6 状态分布 | 畅通 82 / 缓行 81 / 拥挤 80 / 堵塞 81 |
| XGBoost-OBB Macro-F1（分层 70/30） | ~0.96 |
| XGBoost-OBB Accuracy | ~0.96 |
| 状态单调性核验 | 速度递减、密度/占有率递增（通过）|

## 6. 输入输出对照

### 6.1 步骤 1：特征提取 `01_extract_features.py`

| 输入文件 | 内容 |
|---|---|
| `data/UTE/datasets/{ds}/frenet.csv` | 车辆轨迹与运动学（车速、加速度、车道）|
| `data/UTE/datasets/{ds}/sample video.mp4` | 视频帧尺寸（用于网格离散化）|
| `inputs/{ds}_pixel_obb.csv` | 第一部分交付的 OBB 标注表 |

| 输出文件 | 内容 |
|---|---|
| `outputs/features/{ds}_windows.csv` | 每窗一行的滑窗特征表（含 30+ 维特征）|
| `outputs/features/{ds}_grid_tensors.npz` | 每窗 (4, 4, 12) 网格张量 |
| `outputs/features/all_windows.csv` | 全部数据集合并的窗口特征表 |
| `outputs/reports/feature_summary.json` | 每数据集窗口数与文件路径汇总 |

### 6.2 步骤 2：状态标签 `02_build_state_labels.py`

| 输入 | 内容 |
|---|---|
| `outputs/features/all_windows.csv` | 步骤 1 产出 |

| 输出 | 内容 |
|---|---|
| `outputs/labels/state_labels.csv` | 每窗 label（int）+ label_name（中文）+ score |
| `outputs/reports/labels_summary.json` | 每类计数、阈值、每数据集分布 |
| `outputs/figures/state_distribution.png` | XAM-N-6 上 4 类状态分布柱状图 |

### 6.3 步骤 3：XGBoost 分类 `03_run_classification.py`

| 输入 | 内容 |
|---|---|
| `outputs/features/all_windows.csv` + `outputs/labels/state_labels.csv` | 步骤 1、2 产出 |

| 输出 | 内容 |
|---|---|
| `outputs/reports/classification_results.json` | 完整指标 + 混淆矩阵 + 特征重要性 |
| `outputs/figures/confusion_matrix.png` | 4×4 混淆矩阵 |
| `outputs/figures/feature_importance.png` | Top 12 特征 |

## 7. 配置说明（`configs/datasets.json`）

| 字段 | 含义 |
|---|---|
| `datasets.{key}.root` | 数据集相对包根目录的路径 |
| `datasets.{key}.fps` | 视频帧率（用于时间-帧号换算）|
| `datasets.{key}.frame_offset` | OBB pixel 表起始帧偏移 |
| `datasets.{key}.segment_length_m` | 路段物理长度（米）|
| `datasets.{key}.speed_limit_kmh` | 限速（用于速度比计算）|
| `feature.window_s` | 滑窗长度（秒），默认 5 |
| `feature.step_s` | 滑窗步长（秒），默认 1 |
| `feature.grid_cols` / `grid_rows` | 网格离散化尺寸，默认 12×4 |
| `feature.n_states` | 状态类别数，默认 4 |
| `feature.label_smoothing_window` | 标签多数投票平滑窗口，默认 5 |
| `experiment.main_dataset` | 主任务数据集，默认 xamn6 |
| `experiment.test_ratio` | 训练 / 测试划分比例，默认 0.30 |
| `experiment.xgboost` | XGBoost 完整超参 |

## 8. 常见问题

**Q：第一步报错 `Missing OBB pixel table ...`**
检查 `inputs/` 下是否已有 `{ds}_pixel_obb.csv`。这些文件来自第一部分交付包 `hbb_to_obb_minimal/outputs/processed/` 目录，需要手动复制过来。

**Q：第一步特别慢**
首次跑特征抽取在 XAM-N-6（约 100 万行 OBB）上耗时约 1–2 分钟（CPU）。再次跑会一样耗时——特征抽取不缓存。

**Q：状态标签数量与预期不符**
检查 `outputs/reports/labels_summary.json` 的 `label_counts_main`。如四类不均衡严重，确认 K-Means 聚类是否落入物理顺序兜底——会自动切换到风险分位数模式，类别数会接近均衡。

**Q：XGBoost Macro-F1 偏低（< 0.90）**
通常是状态标签生成异常导致。可先打开 `outputs/labels/state_labels.csv` 用 Excel 排查每个数据集的标签是否合理（畅通速度高于堵塞、堵塞密度最大等）。

**Q：网格张量怎么读？**

```python
import numpy as np
data = np.load("outputs/features/xamn6_grid_tensors.npz", allow_pickle=False)
tensors = data["tensors"]            # shape (324, 4, 4, 12)
start_s = data["start_s"]            # shape (324,)
window_id = data["window_id"]        # shape (324,)
channel_names = data["channel_names"]
# tensors[i] 是第 i 个窗口的 4 通道 4×12 张量
```

## 9. 复现成功的判断

完整链路跑通后应满足：

1. ✅ `outputs/features/` 下应有 6 个文件（3 个 `_windows.csv` + 3 个 `_grid_tensors.npz`）+ `all_windows.csv`
2. ✅ `outputs/labels/state_labels.csv` 行数 = 全部窗口数（约 1644 行）
3. ✅ `outputs/reports/classification_results.json` 中 `metrics.f1_macro` ≥ 0.93
4. ✅ `outputs/figures/` 下 3 张 PNG 图都可正常打开

如全部满足，则本包功能完整复现。

---

*本包不包含未来状态预测模型、长时预测扩展、消融实验等高阶模块，这些将在后续交付包中提供。*
