# UTE Traffic State Experiment

基于 UTE 真实无人机交通数据，完成"水平框转旋转框 → 交通状态特征 → 状态识别与预测"的一条可复现实验链路，用于支撑论文实验部分。

## 项目主要内容

1. 从 `pixel.csv` 生成带角度的 OBB（旋转框）标注；
2. 基于 HBB / OBB 占有率与 `frenet.csv` 提取平均车头时距、加速度干扰、MGTI 等特征；
3. 完成状态识别、状态预测、HBB/OBB 消融实验、参数敏感性分析、PKDD 跨场景泛化；
4. 输出论文实验所需的指标表、图表与报告。

## 数据安排

- `XAM-N-6`：主实验（晚高峰，覆盖畅通到拥堵）
- `PKDD-8`：自由流补充与跨场景泛化（补"畅通"样本量）
- `XAM-N-5`：OBB 效果补充验证（公开 sample video 为降采样版本，不作为完整逐帧主实验）
- `XAM-S-9`：轨迹级补充（无 pixel.csv，不进入 OBB 主实验）

## 目录速览

| 路径 | 说明 |
|---|---|
| `src/ute_pipeline/` | 算法核心实现 |
| `scripts/` | 复现入口脚本（编号执行） |
| `configs/datasets.json` | 数据集与实验超参的统一配置 |
| `docs/` | 复现说明、实验报告、参考资料 |
| `data_check/` | UTE 公开样例数据 |
| `outputs/` | 最近一次跑出的 CSV / 图 / JSON 结果 |

## 快速开始

```bash
# 1. 创建环境
conda env create -f environment.yml
conda activate vehicle_ute

# 2. 环境自检
python scripts/check_runtime.py

# 3. 一键复跑全部实验
bash scripts/run_all.sh
```

不使用 conda 时可改用 `python -m venv .venv && pip install -r requirements.txt`。

## 主要输出

- `outputs/processed/*_pixel_obb.csv`：HBB → OBB 标注表
- `outputs/features/*_windows.csv`：滑窗交通状态特征
- `docs/experiment_report.md`：完整实验报告
- `outputs/figures/`：论文图表与 OBB 抽帧可视化

## 进一步阅读

- 复现步骤、数据准备、常见问题：[`docs/reproduction_guide.md`](docs/reproduction_guide.md)
- 方法、创新点、实验结果与分析：[`docs/experiment_report.md`](docs/experiment_report.md)
