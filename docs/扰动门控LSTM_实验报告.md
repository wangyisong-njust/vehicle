# GTSEP-DL 道路交通状态预测实验结果报告

## 1. 实验任务与数据

本实验围绕 GTSEP-DL 道路交通状态预测方法开展验证，实验包括短时无人机交通状态预测和 PeMS08 长时交通参数预测两部分。

短时预测实验采用 UTE XAM-N-6 无人机航拍数据，按时间顺序划分训练集与测试集，训练集占 70%，测试集占 30%。预测目标为未来 3 s、5 s 和 8 s 的四类交通状态。为全面评估模型性能，短时状态分类指标采用 Accuracy、Precision、Recall、Macro-F1 和 Weighted-F1，其中 Macro-F1 用于衡量模型对不同交通状态类别的整体识别能力。

长时预测实验采用 PeMS08 固定检测器数据，数据时间粒度为 5 min，包含 flow、occupancy 和 speed 等交通变量。长时任务预测未来 5 min、15 min 和 30 min 的 flow 与 speed 连续值，并进一步补充基于 speed 阈值的四类交通状态分类验证。回归任务评价指标采用 MAE 和 RMSE，分类补充实验采用 Accuracy 和 Macro-F1。

## 2. GTSEP-DL 模型设置

GTSEP-DL 以 4 通道空间张量、8 维宏微观标量特征和 MGTI 扰动描述符作为输入。4 通道空间张量包括 OBB 占有率、HBB 参考占有率、车辆朝向 sinθ 和 cosθ。模型首先通过 2D-CNN 提取空间结构特征，再将空间编码、标量特征和 MGTI 扰动描述符输入扰动门控 LSTM 进行时序建模，最后输出未来交通状态类别。

短时预测主实验参数如下。

| 参数 | 数值 |
|---|---:|
| sequence length | 8 |
| hidden size | 64 |
| learning rate | 0.0007 |
| epochs | 40 |
| batch size | 32 |
| seed | 161 |

## 3. 短时交通状态预测结果

### 3.1 3 s 主预测结果

| 模型 | Accuracy | Precision | Recall | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|---:|
| XGBoost-future | 0.6064 | 0.5741 | 0.4957 | 0.4620 | 0.6531 |
| XGBoost-temporal-future | 0.4043 | 0.4866 | 0.3314 | 0.3298 | 0.5019 |
| PSO-XGBoost-future | 0.6596 | 0.5694 | 0.5562 | 0.4978 | 0.6997 |
| LSTM-future | 0.6915 | 0.4892 | 0.4820 | 0.4700 | 0.7278 |
| GRU-future | 0.7021 | 0.4944 | 0.4888 | 0.4759 | 0.7378 |
| 1D-CNN-LSTM-future | 0.7766 | 0.4287 | 0.4390 | 0.4324 | 0.7601 |
| LSTSC-future | 0.7021 | 0.4837 | 0.4728 | 0.4650 | 0.7360 |
| GTSEP-DL | **0.8191** | 0.5537 | 0.5313 | **0.5417** | **0.8346** |

在 3 s 短时预测任务中，GTSEP-DL 在 Accuracy、Macro-F1 和 Weighted-F1 三项核心指标上均取得最优结果。与最强外部基线 PSO-XGBoost 相比，GTSEP-DL 的 Macro-F1 提升 4.39 个百分点；与 GRU-future 相比，Macro-F1 提升 6.58 个百分点。

### 3.2 对比文献方法结果

PSO-XGBoost、1D-CNN-LSTM 和 LSTSC 分别对应对比1、对比2和对比3的代表性方法。为保证结果可比，三类对比方法均在本文 XAM-N-6 数据、同一时间顺序训练/测试划分和同一 3 s 未来状态预测任务上重新实现并评估。由于三篇对比文献的数据来源、任务定义和原始评价指标并不完全一致，本文不直接引用原论文数值，而是保留其核心模型结构，在本文四类短时状态分类任务上重新训练和测试。

三类对比方法的实现口径如下。

| 对比方法 | 对应文献 | 原文方法要点 | 本文复现口径 |
|---|---|---|---|
| PSO-XGBoost | 对比1：基于 BO-FCM 和 PSO-XGBoost 的城市快速路交通状态识别 | 原文先用 BO-FCM 对流量、速度、占有率进行四类交通状态划分，再用 PSO 优化 XGBoost 超参数进行状态识别 | 本文已统一使用 XAM-N-6 四类状态标签，因此保留 PSO 优化 XGBoost 的监督分类部分，在相同 3 s 未来状态预测划分上训练 |
| 1D-CNN-LSTM | 对比2：Traffic State Prediction Using One-Dimensional Convolution Neural Networks and Long Short-Term Memory | 原文使用 1D-CNN 提取一维交通序列特征，再用 LSTM 建模历史依赖，原任务为 flow/speed 回归预测 | 本文将 1D-CNN 特征编码器和 LSTM 时序主干迁移为四类状态分类器，输入同一标量时序特征，输出未来 3 s 状态 |
| LSTSC | 对比3：A multi-modal attention neural network for traffic flow prediction by capturing long-short term sequence correlation | 原文通过长短期时序相关模块、注意力机制和 CNN 捕获 PeMS 交通流序列相关性，原任务为 PeMS08/PeMSD7(M) 长时回归预测 | 本文构建轻量 LSTSC-style 分类基线，用短时分支、长时分支和注意力融合模拟长短期时序相关建模，并在 XAM-N-6 四类状态任务上评估 |

| 模型 | Accuracy | Precision | Recall | Macro-F1 | Weighted-F1 |
|------|----------|-----------|--------|----------|-------------|
| PSO-XGBoost | 0.6596 | **0.5694** | **0.5562** | 0.4978 | 0.6997 |
| 1D-CNN-LSTM | 0.7766 | 0.4287 | 0.4390 | 0.4324 | 0.7601 |
| LSTSC | 0.7021 | 0.4837 | 0.4728 | 0.4650 | 0.7360 |
| GTSEP-DL（本文） | **0.8191** | 0.5537 | 0.5313 | **0.5417** | **0.8346** |

与三类对比文献方法相比，GTSEP-DL 在 Accuracy、Macro-F1 和 Weighted-F1 上均取得最优结果。PSO-XGBoost 的 Precision 和 Recall 略高，但其 Accuracy、Macro-F1 和 Weighted-F1 均低于本文方法，说明 GTSEP-DL 在整体判别稳定性和类别综合性能上更优。

### 3.3 3 s、5 s、8 s 多步长预测结果

| Horizon | GTSEP-DL Accuracy | GTSEP-DL Macro-F1 | 最强外部基线 Macro-F1 | 最优模型 |
|---:|---:|---:|---:|---|
| 3 s | **0.8191** | **0.5417** | 0.4788 | GTSEP-DL |
| 5 s | **0.8043** | **0.5355** | 0.4224 | GTSEP-DL |
| 8 s | **0.5955** | **0.4066** | 0.3620 | GTSEP-DL |

GTSEP-DL 在 3 s、5 s 和 8 s 三个短时预测步长上均取得最优结果，说明该方法在不同短时预测尺度下具有稳定优势。

## 4. 消融实验结果

| 消融变体 | 实验含义 | Accuracy | Macro-F1 |
|---|---|---:|---:|
| GTSEP-DL | 完整方法 | **0.8191** | **0.5417** |
| w/o OBB orientation | 去掉车辆朝向 sinθ/cosθ 通道 | 0.6702 | 0.4524 |
| w/o HBB channel | 去掉 HBB 参考占有率通道 | 0.7766 | 0.4806 |
| w/o 2D-CNN | flatten+MLP 替代 2D-CNN | 0.4894 | 0.3055 |
| w/o LSTM | CNN 后时间平均池化 | 0.5106 | 0.2792 |
| w/o spatial tensor | 空间张量置零 | 0.6702 | 0.4505 |
| w/o MGTI | MGTI 扰动描述符置零 | 0.8085 | 0.5195 |
| w/o disturbance gate | 标准 LSTM 替代扰动门控 LSTM | 0.7553 | 0.4933 |

消融结果表明，完整 GTSEP-DL 的预测性能最高。去掉 2D-CNN、LSTM 或空间张量后，Macro-F1 明显下降，说明空间结构编码和时序建模是模型性能提升的关键。去掉扰动门控结构后，Macro-F1 从 0.5417 降至 0.4933，说明扰动门控 LSTM 能有效增强模型对交通状态演化过程的建模能力。去掉 MGTI 后，Macro-F1 降至 0.5195，说明 MGTI 扰动描述符对短时状态预测具有正向贡献。

## 5. PeMS08 长时回归预测结果

### 5.1 Flow 预测结果

| Horizon | 指标 | Persistence | RidgeLag | LSTM-deep | GRU-deep | Ours-ST-LSTM |
|---:|---|---:|---:|---:|---:|---:|
| 5 min | MAE | 15.880 | 17.130 | 20.546 | 19.588 | **15.129** |
| 5 min | RMSE | 24.563 | 25.760 | 31.721 | 30.112 | **23.468** |
| 15 min | MAE | 19.512 | 19.365 | 21.939 | 21.307 | **16.946** |
| 15 min | RMSE | 29.838 | 29.152 | 33.708 | 32.459 | **26.326** |
| 30 min | MAE | 24.191 | 21.455 | 23.133 | 22.538 | **18.290** |
| 30 min | RMSE | 36.679 | 32.519 | 35.645 | 34.385 | **28.613** |

在 flow 长时预测任务中，Ours-ST-LSTM 在 5 min、15 min 和 30 min 三个预测步长下的 MAE 与 RMSE 均为最低。相较各步长下的最强基线，Ours-ST-LSTM 的 MAE 分别降低 4.73%、12.49% 和 14.75%。

### 5.2 Speed 预测结果

| Horizon | 指标 | Persistence | RidgeLag | LSTM-deep | GRU-deep | Ours-ST-LSTM |
|---:|---|---:|---:|---:|---:|---:|
| 5 min | MAE | 0.793 | 1.331 | 1.805 | 1.717 | **0.782** |
| 5 min | RMSE | 1.528 | 2.449 | 4.095 | 3.882 | **1.503** |
| 15 min | MAE | 1.258 | 1.883 | 1.957 | 1.865 | **1.252** |
| 15 min | RMSE | 2.693 | 3.534 | 4.371 | 4.165 | **2.604** |
| 30 min | MAE | 1.635 | 2.390 | 2.061 | 2.034 | **1.592** |
| 30 min | RMSE | 3.718 | 4.536 | 4.586 | 4.480 | **3.480** |

在 speed 长时预测任务中，Ours-ST-LSTM 同样在 5 min、15 min 和 30 min 三个预测步长下取得最低 MAE 与 RMSE。相较最强基线 Persistence，Ours-ST-LSTM 的 MAE 分别降低 1.39%、0.48% 和 2.63%。

## 6. PeMS08 长时状态映射补充结果

为进一步验证长时 speed 回归结果对交通状态判断的支撑作用，将预测 speed 按阈值映射为四类交通状态，并采用“状态保持先验 + 高置信转移修正”的后处理策略进行补充验证。该部分不是 Ours-ST-LSTM 内部新增的分类器，也不作为长时任务的主实验结论；它只用于说明连续值回归预测结果经过交通工程阈值映射后，能够支撑未来状态趋势判断。

| Horizon | Persistence Macro-F1 | 直接阈值映射 Macro-F1 | 状态保持修正 Macro-F1 | 状态保持修正 Accuracy |
|---:|---:|---:|---:|---:|
| 5 min | 0.8585 | 0.8404 | **0.8624** | **0.9446** |
| 15 min | 0.7629 | 0.7379 | **0.7712** | **0.9139** |
| 30 min | 0.6976 | 0.6702 | **0.7103** | 0.8928 |

结果表明，在长时状态映射补充实验中，状态保持修正策略在 5 min、15 min 和 30 min 三个预测步长上的 Macro-F1 均超过 Persistence 和直接阈值映射。该结果说明，长时连续交通参数预测能够有效支撑未来交通状态趋势判断，但论文表述中应将其定位为基于 speed 回归输出的状态映射验证，而不是单独提出一个长时分类模型。

## 7. 实验结论

综合短时与长时实验结果，GTSEP-DL 在 UTE 无人机短时交通状态预测任务中取得最优性能，并在 3 s、5 s 和 8 s 多步长预测中均优于外部基线。消融实验验证了空间张量、2D-CNN、MGTI 扰动描述符和扰动门控 LSTM 的有效性。

在 PeMS08 长时预测任务中，Ours-ST-LSTM 在 flow 和 speed 的 5 min、15 min 和 30 min 共 6 个回归预测组合上均取得最低 MAE 与 RMSE，优于 Persistence、RidgeLag、LSTM-deep 和 GRU-deep 等基线。同时，基于 speed 回归输出的状态映射补充实验也表明，状态保持修正能够在不同预测步长上提升未来交通状态判断效果。

上述结果表明，本文方法在无人机短时交通状态预测和固定检测器长时交通参数预测两个任务中均具有较好的预测精度和泛化适配能力。
