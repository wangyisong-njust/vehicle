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
| 近五年文献方法对比 | XAM-N-6 | 与当前状态识别共用同一分层划分，补充 SVM/RF/KNN/GBDT/XGBoost 等基线 |
| 消融实验 | XAM-N-6 | 5 折分层交叉验证 |
| 参数敏感性 | XAM-N-6 | 5 折交叉验证；正文展示 1/3/5/8 秒，30 秒以上仅保留为 JSON 失败模式记录 |
| 未来状态预测 | XAM-N-6 | 按时间顺序前 70% 训练、后 30% 测试，默认预测 3 秒后状态 |
| LSTM/Fusion | XAM-N-6 | 训练段后 20% 作为验证段，用于融合权重选择 |
| 恶化预测 | XAM-N-6 | 连续时间分组 GroupKFold out-of-fold 评估 |
| OBB 补充验证 | XAM-N-5、PKDD-8 | 不参与主训练，只做标注效果和场景合理性检查 |

划分方式说明：

- 当前状态识别用分层随机划分，目的是检验四类状态在特征空间中的可分性；时间序列划分另作为补充，专门观察未见时段泛化。
- 近五年文献方法对比使用 SVM、RF、KNN、GBDT、XGBoost、LSTM、GRU 等常见基线；状态识别基线共用同一分层划分，时序模型在未来预测任务中比较。
- 未来状态预测必须按时间顺序训练和测试，避免把未来窗口信息泄露到训练段。主实验预测 3 秒后状态；正文只展示 1/3/5/8 秒短时状态预测。30 秒以上状态预测因类别支持不足，保留为失败模式/覆盖范围分析，不作为论文主结果。
- 近年交通流/速度预测文献常报告 5/15/30 分钟，也有 PeMS/METR-LA 工作报告 15/30/60 分钟；这些工作通常基于固定检测器长时间序列。本项目主数据 XAM-N-6 约 5.5 分钟，不能把 15/30 分钟写成 UTE 主实验结论。
- 为和长时预测论文同口径补充，当前另建 PeMS08 扩展实验：预测对象改为 flow 与 speed，步长设为 5/15/30 分钟；PeMS08 原始粒度为 5 分钟，不能构造真实 3 分钟标签，因此不重复列 3 分钟结果。这类数据没有 pixel 表和车辆框，不能验证本项目的 HBB→OBB、HF-GO 和车辆微观扰动特征。
- 扩展实验当前脚本实现 Persistence、Seasonal Persistence、Historical Average、Ridge-Lag 和验证集调权的 Ours-TSFusion；如果后续要冲更强长时预测论文口径，可继续加入 ARIMA、SVR、LSTM、GRU、DCRNN、STGCN、GraphWaveNet、TYRE 等。这属于“长时交通流/速度预测”补充，不应替代 UTE 主实验。
- 恶化预测正样本较少，单次 70/30 切分容易把恶化事件集中切到一侧，所以改用连续时间分组 GroupKFold 的 out-of-fold 评估。
- LSTM 使用低维 V+D+F 时序通道，减少 324 个窗口小样本下的过拟合；XGBoost 使用完整 OBB/HF-GO/MGTI 特征，承担高维非线性判别。

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

长时交通流/速度预测扩展实验不放进 `run_all.sh`，因为它会额外下载 PeMS08 数据。需要复现 5/15/30 分钟结果时单独运行：

```bash
python scripts/07_run_long_horizon_forecasting.py --dataset PEMS08 --auto-download
python scripts/04_make_report.py
```

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
outputs/reports/long_horizon_forecasting.json
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
python scripts/07_run_long_horizon_forecasting.py --dataset PEMS08 --auto-download
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
| `outputs/reports/` | JSON 指标、核验结果和中间摘要，含 PeMS08 长时预测扩展结果 |
| `outputs/figures/` | 混淆矩阵、预测曲线、消融图、参数敏感性图、PeMS08 长时预测图、稳健性图、TreeSHAP 图、SHAP 反事实曲线、消融 t 检验矩阵、R/F 散点图、V-D-R-F 特征空间、PKDD 概率图、状态时空图、HF-GO 局部对比和 OBB 抽帧可视化 |
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
| XGBoost-OBB Macro-F1（分层划分，主结果） | ~0.96 |
| RF-OBB Macro-F1（近五年文献基线） | ~0.93 |
| GBDT-OBB Macro-F1（近五年文献基线） | ~0.94 |
| KNN-OBB Macro-F1（近五年文献基线） | ~0.86 |
| SVM-OBB Macro-F1（分层划分，基线） | ~0.90 |
| LR-OBB Macro-F1（分层划分，基线） | ~0.93 |
| XGBoost-OBB Macro-F1（时间序列划分，补充） | ~0.48 |
| XGBoost-future Macro-F1（3s 预测） | ~0.46 |
| GRU-future Macro-F1（3s 预测，近五年文献基线） | ~0.47 |
| Fusion-future Macro-F1（3s 预测） | ~0.47 |
| XGBoost-OBB 多随机种子 Macro-F1 均值 | ~0.94 |
| XGBoost-future 多随机种子 Macro-F1 均值 | ~0.49 |
| 文献对齐消融 M1 V+D Macro-F1 | ~0.94 |
| 文献对齐消融 M3 V+D+R+F Macro-F1 | ~0.95 |
| 本文 M4 消融 Macro-F1 | ~0.96 |
| MGTI 单调性 | 通过（畅通 < 缓行 < 拥挤 < 堵塞） |
| 恶化预测最佳 PR-AUC / ROC-AUC | ~0.30 / ~0.63 |
| 恶化预测正样本率（3s/5s/8s） | ~12-13%（mean + 1.0σ 显著事件阈值） |
| 5 折时间序列 CV Macro-F1 均值 | ~0.47 |

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
- 当前空间特征是窗口级与网格级实现，HF-GO 已使用多边形裁剪计算 OBB 与网格单元交叠面积；项目不做逐像素语义分割或额外检测器训练；
- 恶化预测的标签基于连续分数差异（mean + 1.0σ 显著恶化事件阈值），将恶化限定为稀疏预警事件，正样本率约 12%。
