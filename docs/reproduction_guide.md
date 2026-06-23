# 复现说明

本说明用于在本地从原始 UTE 数据重新生成 OBB 标注、滑窗特征、GTSEP-DL 短时预测实验、PeMS08 长时预测实验、图表和实验结果报告。

最终只需要重点查看以下文档：

- `docs/reproduction_guide.md`：环境配置、数据准备、运行命令、输出位置和排错；
- `docs/扰动门控LSTM_实验报告.md`：新版 GTSEP-DL、扰动门控 LSTM、短时预测、消融实验和 PeMS08 长时预测结果；
- `docs/experiment_report.md`：旧版完整实验报告和历史对照内容，仅作项目背景参考。

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

下表为生成报告数值的精确版本，复现时请严格对齐（尤其是 PyTorch）。短时深度模型在 CPU 上训练，结果对 torch/BLAS 版本敏感，换版本可能使 94 样本测试集上的 Macro-F1 漂移数个百分点。

| 项目 | 版本 |
|---|---|
| Python | 3.11 |
| NumPy | 1.26.4 |
| SciPy | 1.15.3 |
| scikit-learn | 1.7.1 |
| XGBoost | 2.1.4 |
| PyTorch | 2.9.0 |
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
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

如果 `torch==2.9.0` 在当前平台没有可用轮子，可以安装本机可用的 PyTorch 版本，但短时深度模型的数值会随之变化（在 94 样本测试集上 Macro-F1 可能漂移数个百分点、消融次序可能改变），此时应重新运行实验并以新结果为准。

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
| LSTM / GTSEP-DL | XAM-N-6 | 按时间顺序 70/30 划分；LSTM / GRU 基线 3 种子概率集成；GTSEP-DL 主任务预测 3s，另补充 5s、8s 多步长扫描 |
| 恶化预测 | XAM-N-6 | 连续时间分组 GroupKFold out-of-fold 评估 |
| OBB 补充验证 | XAM-N-5、PKDD-8 | 不参与主训练，只做标注效果和场景合理性检查 |

划分方式说明：

- 当前状态识别用分层随机划分，目的是检验四类状态在特征空间中的可分性；时间序列划分另作为补充，专门观察未见时段泛化。
- 近五年文献方法对比使用 SVM、RF、KNN、GBDT、XGBoost、LSTM、GRU 等常见基线；状态识别基线共用同一分层划分，时序模型在未来预测任务中比较。
- 未来状态预测必须按时间顺序训练和测试，避免把未来窗口信息泄露到训练段。主实验预测 3 秒后状态；正文只展示 1/3/5/8 秒短时状态预测。30 秒以上状态预测因类别支持不足，保留为失败模式/覆盖范围分析，不作为论文主结果。
- 近年交通流/速度预测文献常报告 5/15/30 分钟，也有 PeMS/METR-LA 工作报告 15/30/60 分钟；这些工作通常基于固定检测器长时间序列。本项目主数据 XAM-N-6 约 5.5 分钟，不能把 15/30 分钟写成 UTE 主实验结论。
- 为与长时预测论文同口径补充，另建 PeMS08 扩展实验：主任务为 flow/speed 连续值回归，预测步长为 5/15/30 分钟，评价指标为 MAE / RMSE。该任务对比 Persistence、RidgeLag、LSTM-deep、GRU-deep 和 GTSEP-DL。PeMS08 没有 pixel 表和车辆框，不能验证本项目的 HBB→OBB、HF-GO 和车辆微观扰动特征，只用于验证预测骨架在固定检测器宏观序列上的长时适配能力。
- PeMS08 另补充四类状态分类验证：将 speed 按阈值映射为畅通/缓行/拥挤/堵塞，并采用“状态保持先验 + 高置信转移修正”的混合分类策略。该补充实验用于说明长时回归结果对交通状态趋势判断的支撑作用，不替代 flow/speed 回归主任务。
- 恶化预测正样本较少，单次 70/30 切分容易把恶化事件集中切到一侧，所以改用连续时间分组 GroupKFold 的 out-of-fold 评估。
- LSTM-future 使用低维 V+D+F 时序通道，减少 324 窗口小样本下的过拟合；XGBoost-future 使用完整 OBB/HF-GO/MGTI 特征，承担高维非线性判别；GTSEP-DL 把每窗口 4×4×12 旋转框张量与 8 维宏微观标量描述符按帧拼接，并将 MGTI 作为独立扰动描述符输入扰动门控 LSTM，作为单模型端到端预测前端。

## 6. 一键复现

目的：从原始 `pixel.csv` 和 `frenet.csv` 重新生成所有结果。

```bash
bash scripts/run_all.sh
```

脚本执行顺序：

1. `scripts/01_prepare_obb.py`：HBB 转 OBB，生成 `*_pixel_obb.csv` 和抽帧可视化图；
2. `scripts/02_extract_features.py`：提取 OBB/HBB 占有率、平均车头时距、加速度干扰、MGTI 等滑窗特征，并保存每窗口 4×4×12 网格张量到 `outputs/features/{ds}_grid_tensors.npz`；
3. `scripts/03_run_experiments.py`：运行当前状态识别、消融实验、参数敏感性分析、未来状态预测（XGBoost / LSTM / GRU / **GTSEP-DL 扰动门控 LSTM**）和恶化预测；
4. `scripts/05_auto_verify.py`：输出自动核验 JSON；
5. `scripts/06_validate_obb_effect.py`：输出 OBB 效果补充验证 JSON；
6. `scripts/04_make_report.py`：汇总生成 `docs/experiment_report.md`。

PeMS08 长时预测扩展实验不放进 `run_all.sh`，因为它会额外下载 PeMS08 数据。需要复现 5/15/30 分钟 flow/speed 回归结果和状态分类补充结果时单独运行：

```bash
python scripts/07_run_long_horizon_forecasting.py --dataset PEMS08 --auto-download
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
outputs/features/xamn6_grid_tensors.npz
outputs/features/xamn5_grid_tensors.npz
outputs/features/pkdd8_grid_tensors.npz
outputs/reports/experiment_results.json
outputs/reports/long_horizon_forecasting.json
outputs/reports/auto_verification.json
outputs/reports/obb_effect_validation.json
docs/扰动门控LSTM_实验报告.md
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
```

如果只想重跑主实验和 PKDD，不重跑 XAM-N-5，可以把前两条命令中的 `xamn5` 去掉，但最终报告中的 OBB 补充验证会缺少 XAM-N-5 对应结果。

## 8. 输出位置

| 路径 | 内容 |
|---|---|
| `outputs/processed/` | HBB 转 OBB 后的标注表，包含角度、角度置信度和四点坐标 |
| `outputs/features/` | 滑窗特征表 |
| `outputs/reports/` | JSON 指标、核验结果和中间摘要，含 PeMS08 长时预测扩展结果 |
| `outputs/figures/` | 混淆矩阵、预测曲线、消融图、参数敏感性图、PeMS08 长时预测图、稳健性图、TreeSHAP 图、SHAP 反事实曲线、消融 t 检验矩阵、R/F 散点图、V-D-O-M 特征敏感性分布图、PKDD 概率图、状态时空图、HF-GO 局部对比和 OBB 抽帧可视化 |
| `docs/扰动门控LSTM_实验报告.md` | 新版 GTSEP-DL 实验结果报告 |
| `docs/experiment_report.md` | 旧版完整实验报告和历史对照材料 |
