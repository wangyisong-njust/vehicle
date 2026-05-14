# 复现说明

本说明用于在本地从原始 UTE 数据重新生成 OBB 标注、滑窗特征、模型实验结果、图表和最终实验报告。

最终只需要重点查看两份文档：

- `docs/reproduction_guide.md`：环境配置、数据准备、运行命令、输出位置和排错；
- `docs/experiment_report.md`：方法公式、创新点、实验结果、消融分析、参数敏感性分析和结论。

## 1. 项目结构

```text
.
├── configs/
│   └── datasets.json
├── data_check/
│   ├── XAM-N/
│   ├── XAM-S/
│   └── PKDD/
├── docs/
│   ├── experiment_report.md
│   ├── reproduction_guide.md
│   └── references/
├── outputs/
│   ├── features/
│   ├── figures/
│   ├── processed/
│   └── reports/
├── scripts/
└── src/
    └── ute_pipeline/
```

核心入口是 `scripts/run_all.sh`。核心配置是 `configs/datasets.json`。算法实现集中在 `src/ute_pipeline/`。

## 2. 推荐环境

已验证环境：

| 项目 | 版本 |
|---|---|
| Python | 3.10 |
| NumPy | 1.26.4 |
| SciPy | 1.11.4 |
| scikit-learn | 1.4.2 |
| XGBoost | 2.1.4 |
| PyTorch | 1.12.1 |
| OpenCV | 4.8+ |
| SUMO | 1.22.0，可用但当前实验不依赖 |

硬件建议：

| 项目 | 建议 |
|---|---|
| 内存 | 16 GB 以上 |
| 磁盘 | 5 GB 以上 |
| GPU | 可选，当前 LSTM 规模较小，CPU 也能运行 |

随机性说明：

- 默认 seed 为 42；
- LSTM 在不同硬件上可能有轻微波动；
- 不承诺逐位一致，核心指标小范围浮动属于正常现象。

## 3. 环境配置

### 方式 A：conda

目的：创建与当前项目一致的运行环境。

```bash
conda env create -f environment.yml
conda activate vehicle_ute
```

预期结果：

- conda 环境创建成功；
- 激活后命令行环境名显示 `vehicle_ute`。

注意：老版本 CUDA 驱动不适合直接升级到 PyTorch 2.x，建议先使用 `environment.yml` 中的版本组合复现实验。

### 方式 B：venv + pip

目的：不使用 conda 时安装依赖。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

如果 `torch==1.12.1` 在当前平台没有可用轮子，可以安装本机可用的 PyTorch 版本，但需要重新运行实验并以新结果为准。

## 4. 环境自检

目的：检查核心依赖是否可以导入。

```bash
python scripts/check_runtime.py
```

预期结果：

- 每个核心依赖显示 `[OK]`；
- 最后一行显示 `环境自检通过。`

如果出现 `[MISS]`，按提示补装对应依赖。

## 5. 数据准备

数据来自 UTE 公开仓库：

- <https://github.com/Ruyi-Feng/Ubiquitous-Traffic-Eye>

当前项目使用的数据结构如下：

```text
data_check/
├── XAM-N/
│   ├── XAM-N-5/
│   └── XAM-N-6/
├── XAM-S/
│   └── XAM-S-9/
└── PKDD/
    └── PKDD-8/
```

主流程需要以下文件：

```text
data_check/XAM-N/XAM-N-6/pixel.csv
data_check/XAM-N/XAM-N-6/frenet.csv
data_check/XAM-N/XAM-N-6/sample video.mp4
data_check/XAM-N/XAM-N-5/pixel.csv
data_check/XAM-N/XAM-N-5/frenet.csv
data_check/PKDD/PKDD-8/pixel.csv
data_check/PKDD/PKDD-8/frenet.csv
data_check/PKDD/PKDD-8/sample video.mp4
```

数据分工：

| 数据集 | 用途 |
|---|---|
| `XAM-N-6` | 主实验，覆盖晚高峰状态变化 |
| `XAM-N-5` | OBB 效果补充验证，公开视频为降采样版本，不作为完整逐帧主实验 |
| `PKDD-8` | 自由流补充和跨场景泛化 |
| `XAM-S-9` | 轨迹级补充，公开样例无 `pixel.csv`，不进入 OBB 主实验 |

数据划分：

| 实验任务 | 数据来源 | 划分方式 |
|---|---|---|
| 当前状态识别 | XAM-N-6 | 70% 训练、30% 测试，按四类状态分层随机划分 |
| 消融实验 | XAM-N-6 | 5 折分层交叉验证 |
| 参数敏感性 | XAM-N-6 | 5 折交叉验证 |
| 未来状态预测 | XAM-N-6 | 按时间顺序前 70% 训练、后 30% 测试 |
| LSTM/Fusion | XAM-N-6 | 训练段后 20% 作为验证段，用于融合权重选择 |
| 恶化预测 | XAM-N-6 | 连续时间分组 GroupKFold out-of-fold 评估 |
| OBB 补充验证 | XAM-N-5、PKDD-8 | 不参与主训练，只做标注效果和场景合理性检查 |

## 6. 一键复现

目的：从原始 `pixel.csv` 和 `frenet.csv` 重新生成所有结果。

```bash
bash scripts/run_all.sh
```

脚本执行顺序：

1. `scripts/01_prepare_obb.py`：HBB 转 OBB，生成 `*_pixel_obb.csv` 和抽帧可视化图；
2. `scripts/02_extract_features.py`：提取 OBB/HBB 占有率、平均车头时距、加速度干扰、MGTI 等滑窗特征；
3. `scripts/03_run_experiments.py`：运行当前状态识别、未来状态预测、消融实验、参数敏感性分析和恶化预测；
4. `scripts/05_auto_verify.py`：输出自动核验 JSON；
5. `scripts/06_validate_obb_effect.py`：输出 OBB 效果补充验证 JSON；
6. `scripts/04_make_report.py`：汇总生成 `docs/experiment_report.md`。

预期输出：

```text
outputs/processed/xamn6_pixel_obb.csv
outputs/processed/xamn5_pixel_obb.csv
outputs/processed/pkdd8_pixel_obb.csv
outputs/features/xamn6_windows.csv
outputs/features/xamn5_windows.csv
outputs/features/pkdd8_windows.csv
outputs/features/all_windows.csv
outputs/reports/experiment_results.json
outputs/reports/auto_verification.json
outputs/reports/obb_effect_validation.json
docs/experiment_report.md
```

## 7. 分步运行

目的：只重跑某个阶段，适合调试或替换数据后局部复现。

```bash
python scripts/01_prepare_obb.py --datasets xamn6 xamn5 pkdd8
python scripts/02_extract_features.py --datasets xamn6 xamn5 pkdd8
python scripts/03_run_experiments.py
python scripts/05_auto_verify.py
python scripts/06_validate_obb_effect.py
python scripts/04_make_report.py
```

如果只想重跑主实验和 PKDD，不重跑 XAM-N-5，可以把前两条命令中的 `xamn5` 去掉，但最终报告中的 OBB 补充验证会缺少 XAM-N-5 对应结果。

## 8. 输出位置

| 路径 | 内容 |
|---|---|
| `outputs/processed/` | HBB 转 OBB 后的标注表，包含角度、角度置信度和四点坐标 |
| `outputs/features/` | 滑窗特征表 |
| `outputs/reports/` | JSON 指标、核验结果和中间摘要 |
| `outputs/figures/` | 混淆矩阵、预测曲线、消融图、参数敏感性图、稳健性图、TreeSHAP 图、SHAP 反事实曲线、R/F 散点图、PKDD 概率图、状态时空图、HF-GO 局部对比和 OBB 抽帧可视化 |
| `docs/experiment_report.md` | 最终实验报告 |

## 9. 复现成功的判断

关键结果应接近下面数值：

| 指标 | 参考值 |
|---|---:|
| XAM-N-6 OBB 行数 | 1,043,909 |
| XAM-N-5 OBB 行数 | 334,721 |
| PKDD-8 OBB 行数 | 586,534 |
| XAM-N-6 特征窗口 | 324 |
| XAM-N-5 特征窗口 | 261 |
| PKDD-8 特征窗口 | 1,059 |
| 状态类别数 | 4（畅通/缓行/拥挤/堵塞） |
| XGBoost-OBB Macro-F1（分层划分，主结果） | ~0.94 |
| SVM-OBB Macro-F1（分层划分，基线） | ~0.90 |
| LR-OBB Macro-F1（分层划分，基线） | ~0.93 |
| XGBoost-OBB Macro-F1（时间序列划分，补充） | ~0.43 |
| XGBoost-future Macro-F1（3s 预测） | ~0.64 |
| XGBoost-OBB 多随机种子 Macro-F1 均值 | ~0.94 |
| XGBoost-future 多随机种子 Macro-F1 均值 | ~0.65 |
| 文献对齐消融 M1 V+D Macro-F1 | 以重跑结果为准 |
| 文献对齐消融 M3 V+D+R+F Macro-F1 | 以重跑结果为准 |
| MGTI 单调性 | 通过（畅通 < 缓行 < 拥挤 < 堵塞） |
| 恶化预测最佳 AUC | 以重跑结果为准 |
| 恶化预测正样本率（3s/5s/8s） | ~65% |
| 5 折时间序列 CV Macro-F1 均值 | ~0.50 |

如果结果不一致，优先检查：

1. 是否使用同一份数据；
2. `configs/datasets.json` 是否被改动（特别是 `n_states`、`quantiles`、MGTI 权重、XGBoost 参数）；
3. 依赖版本是否变化；
4. 是否只重跑了部分脚本；
5. `docs/experiment_report.md` 是否在 `05_auto_verify.py` 和 `06_validate_obb_effect.py` 之后重新生成。

## 10. 常见问题

| 现象 | 处理 |
|---|---|
| `cv2` 导入失败 | 安装 `opencv-python` 或 `opencv-python-headless` |
| `xgboost` 导入失败 | 检查 `xgboost==2.1.4` 是否安装 |
| `torch` 版本装不上 | 使用本机可安装版本后重新跑实验，不要直接沿用旧指标 |
| CSV 文件较大 | `outputs/processed/*.csv` 可删除后通过 `run_all.sh` 重建 |
| 想替换数据 | 在 `configs/datasets.json` 新增数据条目，再传给脚本的 `--datasets` 参数 |
| 报告结果没有更新 | 重新运行 `python scripts/04_make_report.py` |

## 11. 当前不纳入的内容

- 当前版本不做 SUMO 仿真；
- 当前 OBB 由 `pixel.csv` 和轨迹方向生成，不训练 YOLOv8-OBB 检测器。原因是公开数据没有人工旋转框真值，直接用伪 OBB 标签再训练检测器不会突破伪标签质量上限，还可能破坏 pixel 表与车辆运动学字段的一一对应关系；
- 当前状态标签是无监督聚类构造的参考标签（4 类，基于速度比、密度、变道干扰率和方向波动指数），不是人工逐窗口真值；
- 当前空间特征是窗口级和采样网格级实现，不是完整逐像素多边形裁剪；
- 恶化预测的标签基于连续分数差异（第 65 百分位数阈值），非离散状态跳跃。
