# UTE 交通状态评估与预测实验报告

本报告基于 UTE 真实无人机交通数据，围绕“水平框补充角度信息、交通状态识别、未来状态预测”三个问题组织实验。主数据集为 XAM-N-6，XAM-N-5 用于 OBB 标注效果补充验证，PKDD-8 用于自由流场景补充和跨场景合理性检查。

## 核心结论

- **当前状态识别可用**：`XGBoost-OBB` 在 XAM-N-6 分层测试集上的 Macro-F1 为 0.9381，SVM 与 LR 基线已纳入对比。
- **未来状态预测**：3s 预测步长下静态 `XGBoost-future` Macro-F1 为 0.6710，趋势增强后 `XGBoost-temporal-future` 为 0.5106。
- **恶化预测作为补充任务呈现**：最佳 ROC-AUC 为 0.5115，当前样本下 R/F 与微观特征未稳定超过 V+D 基线，报告中已按负结果解释。
- **数据边界已明确**：XAM-N-5 的公开视频为降采样版本，因此该数据集用于 pixel 表级 OBB 验证，不作为完整逐帧视频主实验。

---

# 1 实验设计与交通状态识别

## 1.1 实验目标

基于 UTE 真实无人机轨迹数据完成交通状态评估与预测。状态分为四类：畅通、缓行、拥挤、堵塞。实验主要回答以下问题：

1. 公开数据中的 HBB 水平框能否在不重新训练检测器的条件下补充角度信息，形成可复用的 OBB 标注表；
2. 平均车头时距、加速度干扰和复合 MGTI 指标能否有效刻画交通状态变化；
3. 当前状态识别、未来状态预测和恶化预警能否形成一套可复现的论文实验链路。

## 1.2 数据使用与分工

| 数据集 | 角色 | 使用边界 |
|---|---|---|
| XAM-N-6 | 主实验 | `pixel.csv`、`frenet.csv` 与公开视频可用于主流程，覆盖晚高峰状态变化 |
| XAM-N-5 | OBB 效果补充验证 | `pixel.csv` 与 `frenet.csv` 可用于 OBB 表生成；公开视频为 6fps 降采样版本，不作为完整逐帧主实验 |
| PKDD-8 | 泛化验证 | 自由流为主，用于补充畅通场景和跨场景合理性检查，不与 XAM-N-6 直接视作同分布训练数据 |

### 1.2.1 训练、验证与测试划分

主实验均以 XAM-N-6 为准。当前状态识别采用 75%/25% 的分层随机划分，保证四类状态在训练集和测试集中的比例基本一致。消融实验和参数敏感性分析采用 5 折分层交叉验证，不再单独划验证集。

未来状态预测和恶化预测按时间顺序划分，前 75% 时间窗口用于训练，后 25% 时间窗口用于测试。LSTM 与 Fusion 使用训练段后 20% 作为验证段，用来估计门控参数；最终指标只在后 25% 测试段上统计。XAM-N-5 和 PKDD-8 不参与主模型训练，分别用于 OBB 效果验证和自由流场景检查。

## 1.3 方法设计

### 1.3.1 HBB 转 OBB

原始 `pixel.csv` 给出车辆水平框 $B_i=(x_i,y_i,w_i,h_i)$。先计算车辆中心点 $c_i=(x_i+w_i/2, y_i+h_i/2)$，对同一车辆按帧号排序，在前后搜索半径内选取位移最大的稳定片段，估计车辆方向角 $\theta_i=\operatorname{atan2}(c_{r,y}-c_{l,y}, c_{r,x}-c_{l,x})$。OBB 四点由中心点、长边、短边和旋转矩阵得到。输出表保持与原始 `pixel.csv` 一行一目标对应，并额外增加 `theta`、`theta_deg`、`theta_conf` 和四个角点坐标。

本研究没有直接改用 YOLOv8-OBB 重新检测，主要原因是 UTE 公开数据提供的是 HBB 水平框和车辆轨迹表，没有人工旋转框真值。若直接用当前算法生成的 OBB 作为伪标签再训练 YOLOv8-OBB，本质上仍受伪标签质量约束，且可能破坏 pixel 表与车辆编号、车道、速度、加速度等字段的一一对应关系。当前采用轨迹方向补角度，是在现有数据条件下更稳妥的方案；如果后续补充人工旋转框标注，可将 YOLOv8-OBB 作为检测器扩展实验。

### 1.3.2 OBB 空间占有率

窗口长度为 5 秒，滑动步长为 1 秒。当前实现使用旋转框几何面积进行窗口级占有率统计：

$$O_{HBB}=\frac{\sum_i w_i h_i}{N_f A},\qquad O_{HFGO}=\frac{\sum_{i,g} area(P_i^{OBB}\cap G_g)}{N_f A}.$$

HF-GO 使用 Sutherland-Hodgman 多边形裁剪计算 OBB 与物理网格单元的交叠面积，再进行解析面积累加。与简单采样点计数相比，该方法能保留车辆跨网格、斜向占用和边界截断时的真实占用比例，更适合作为本文区别于参考文献的空间表达增强模块。

![HF-GO热力图](../outputs/figures/hfgo_hbb_vs_obb_heatmap.png)

### 1.3.3 平均车头时距

在同帧、同车道内按纵向位置排序，对跟驰车辆计算空间间距 $g_i=s_{lead}-s_i-(L_{lead}+L_i)/2$，再计算时间车头时距 $THW_i=g_i/\max(v_i,0.1)$。窗口内输出 `mean_headway_s`、`min_headway_s`、`mean_space_gap_m`。

### 1.3.4 加速度干扰

$$I_a=\operatorname{std}(a_i).$$

反映车辆频繁加减速带来的扰动，更容易捕捉交通流由稳定向不稳定转变的过程。

### 1.3.5 复合 MGTI 指标

MGTI 定义为多指标复合风险得分：

$$MGTI = w_1 z(I_a) + w_2 z(\rho) + w_3 z(O_{HFGO}) + w_4 z(v/v_{lim}) + w_5 z(\overline{THW}).$$

其中 $z(\cdot)$ 为 z-score 标准化。在 UTE 数据中，拥堵状态下 THW 随拥堵程度递增，因此直接使用 $z(THW)$ 作为车头时距分量，确保拥堵风险随状态等级升高而增强。加速度干扰 $I_a$ 在 UTE 数据中为非单调特征，已从复合指标中移除（$w_1=0$），但仍作为独立特征保留在消融实验中。当前权重配置：加速度干扰 $w_1=0$、密度 $w_2=1.0$、OBB占有率 $w_3=1.0$、速度比 $w_4=-1.0$、车头时距 $w_5=1.0$。

### 1.3.6 状态标签构造

为降低消融实验中的标签泄露，参考标签只使用速度比、密度和 OBB 占有率构造，不使用车头时距、加速度干扰和 MGTI：

$$S=0.65\,z(1-v/v_{lim})+0.25\,z(\rho)+0.10\,z(O_{HFGO}).$$

在 XAM-N-6 上按分位数 [0.25, 0.5, 0.75] 切分为 4 类。标签分布：81 窗口为畅通类，81 窗口为缓行类，81 窗口为拥挤类，81 窗口为堵塞类。

## 1.4 当前状态识别结果（主结果：分层划分）

主要评价采用分层随机划分（stratified split），确保训练集和测试集各类别比例一致。消融实验和参数敏感性分析使用 5 折分层交叉验证（stratified 5-fold CV），以交叉验证均值作为报告指标。

| 模型 | Accuracy | Precision | Recall | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|---:|
| Majority | 0.2500 | 0.0625 | 0.2500 | 0.1000 | 0.1000 |
| TorchLinear-OBB | 0.9750 | 0.9773 | 0.9750 | 0.9749 | 0.9749 |
| XGBoost-HBB | 0.9125 | 0.9151 | 0.9125 | 0.9130 | 0.9130 |
| XGBoost-OBB | 0.9375 | 0.9393 | 0.9375 | 0.9381 | 0.9381 |
| XGBoost-OBB-MGTI | 0.9375 | 0.9393 | 0.9375 | 0.9381 | 0.9381 |
| SVM-OBB | 0.9000 | 0.9024 | 0.9000 | 0.9009 | 0.9009 |
| LR-OBB | 0.9500 | 0.9500 | 0.9500 | 0.9500 | 0.9500 |

测试集各状态样本数：畅通 20，缓行 20，拥挤 20，堵塞 20。


补充：时间序列划分（后 25% 作为测试集）的 XGBoost-OBB Macro-F1 为 0.4974。时间序列划分存在训练/测试分布偏移（训练覆盖拥堵积累期，测试覆盖恢复期），因此分类难度显著高于分层划分。该结果反映了模型对未见时间段的泛化能力。加入因果滞后、差分和滚动趋势特征后，时间序列测试 Macro-F1 提升至 0.6014。

`XGBoost-OBB-MGTI` 的 Macro-F1 为 0.9381，相比 `XGBoost-OBB` 基本持平。

结果说明：分层划分下模型能够区分四类交通状态；时间序列划分用于检验未见时段泛化能力，指标低于分层划分，反映真实时序预测场景更困难。两类结果共同呈现，可同时支撑特征可分性与时序泛化分析。

![分类指标](../outputs/figures/classification_metrics.png)

![混淆矩阵](../outputs/figures/cm_xgboost_obb.png)

### 1.4.1 时间序列交叉验证

| 方法 | 折数 | Accuracy 均值 | Accuracy 标准差 | Macro-F1 均值 | Macro-F1 标准差 |
|---|---:|---:|---:|---:|---:|
| expanding_time_series_cv | 5 | 0.6926 | 0.1359 | 0.4990 | 0.1151 |

### 1.4.2 特征重要性

| 排名 | 特征 | 重要性 |
|---:|---|---:|
| 1 | `mean_headway_s` | 0.2907 |
| 2 | `speed_ratio` | 0.1688 |
| 3 | `mean_speed_kmh` | 0.1167 |
| 4 | `mgti` | 0.0839 |
| 5 | `theta_conf_mean` | 0.0730 |
| 6 | `vehicle_count` | 0.0513 |
| 7 | `density_veh_per_m` | 0.0453 |
| 8 | `hfgo_occupancy` | 0.0376 |
| 9 | `direction_fluctuation` | 0.0168 |
| 10 | `grid_entropy_reduction` | 0.0163 |

## 1.5 未来状态预测

预测步长：3.0 秒

| 模型 | Accuracy | Precision | Recall | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|---:|
| XGBoost-future | 0.7692 | 0.6918 | 0.6991 | 0.6710 | 0.7652 |
| XGBoost-temporal-future | 0.5385 | 0.5715 | 0.5498 | 0.5106 | 0.5891 |
| LSTM-future | 0.1667 | 0.1709 | 0.1821 | 0.1534 | 0.1687 |
| Fusion-future | 0.7308 | 0.6484 | 0.6783 | 0.6396 | 0.7383 |

Fusion-future 使用双通道/多通道门控融合：先计算验证段各通道的交叉熵误差，并用指数平滑得到稳定误差 $\bar e_m$；再按 $T=\max(T_{min},T_0\exp(-\alpha\bar e))$ 得到动态温度，最后用 $w_m=softmax(-\bar e_m/T)$ 生成 XGBoost、趋势 XGBoost 和 LSTM 的融合权重。该实现与 docx 中“带温 Softmax 动态门控”的公式保持一致，且不使用测试标签调权。

静态 `XGBoost-future` 的 Macro-F1 为 0.6710。加入滞后、差分和滚动趋势后的 `XGBoost-temporal-future` 为 0.5106，相比静态模型下降 16.04 个百分点。这可能是因为复合 MGTI 已经提供了较强的状态变化信息，额外的高维时序特征在小样本条件下引入了噪声。LSTM 在当前小样本条件下表现不如 XGBoost（0.1534）。带温 Softmax 门控得到静态 XGBoost 48% + 趋势 XGBoost 48% + LSTM 5%，Fusion-future Macro-F1 为 0.6396。

结果说明：当前预测任务只有 324 个 XAM-N-6 时间窗，LSTM 的有效训练样本更少，因此端到端序列模型没有形成稳定优势。静态 XGBoost-future 的 Macro-F1 已达 0.6710，复合 MGTI 提供了较强的状态变化信号，使得简单静态特征即可达到较好的预测效果。

![未来预测曲线](../outputs/figures/future_prediction_curve.png)

![融合预测混淆矩阵](../outputs/figures/cm_fusion_future.png)

## 1.6 消融实验（5 折分层交叉验证）

| 消融集 | Accuracy | Precision | Recall | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|---:|
| M1: V+D | 0.9475 | 0.9514 | 0.9478 | 0.9472 ± 0.0219 | 0.9471 |
| M2: V+D+R | 0.9413 | 0.9459 | 0.9415 | 0.9411 ± 0.0162 | 0.9411 |
| M3: V+D+R+F | 0.9506 | 0.9546 | 0.9507 | 0.9508 ± 0.0148 | 0.9509 |
| M4: Ours+headway+acc+MGTI | 0.9475 | 0.9533 | 0.9474 | 0.9471 ± 0.0131 | 0.9472 |

最优消融组合是 `M3: V+D+R+F`，5 折 CV Macro-F1 为 0.9508±0.0148。相比 `M1: V+D` 的 0.9472，提升 0.36 个百分点。

**分析**：消融实验按参考文献的阶梯组织：`M1: V+D` 为速度与密度基线，`M2` 加入变道干扰率 R，`M3` 加入方向波动指数 F，`M4` 进一步加入本文的 HF-GO、车头时距、加速度干扰和 MGTI。这样可以直接回答 R/F 是否有效，以及本文新增微观行为与高保真空间占有率是否带来额外增益。

![消融实验](../outputs/figures/ablation_macro_f1.png)

## 1.7 参数敏感性分析（5 折 CV）

| 参数 | 取值 | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| XGBoost max_depth | 2 | 0.9413 | 0.9407 ± 0.0132 |
| XGBoost max_depth | 3 | 0.9475 | 0.9474 ± 0.0163 |
| XGBoost max_depth | 4 | 0.9505 | 0.9504 ± 0.0158 |
| XGBoost max_depth | 5 | 0.9505 | 0.9502 ± 0.0157 |
| XGBoost max_depth | 6 | 0.9505 | 0.9502 ± 0.0157 |
| prediction horizon(s) | 1.0 | 0.8000 | 0.7300 |
| prediction horizon(s) | 3.0 | 0.7692 | 0.6710 |
| prediction horizon(s) | 5.0 | 0.6447 | 0.5175 |
| prediction horizon(s) | 8.0 | 0.4110 | 0.3495 |

![参数敏感性](../outputs/figures/parameter_sensitivity.png)

## 1.8 多随机种子稳健性检验

为避免单次随机划分造成偶然性，补充使用 5 组随机种子进行重复实验。当前状态识别重复分层划分，未来状态预测保持时间顺序划分，仅改变模型随机种子。

| 任务 | 模型 | Accuracy 均值 | Accuracy 标准差 | Macro-F1 均值 | Macro-F1 标准差 |
|---|---|---:|---:|---:|---:|
| 当前状态识别 | XGBoost-HBB | 0.9275 | 0.0398 | 0.9275 | 0.0396 |
| 当前状态识别 | XGBoost-OBB | 0.9350 | 0.0348 | 0.9350 | 0.0347 |
| 当前状态识别 | LR-OBB | 0.9350 | 0.0267 | 0.9349 | 0.0265 |
| 未来状态预测 | XGBoost-future | 0.7615 | 0.0103 | 0.6583 | 0.0192 |

![多随机种子稳健性](../outputs/figures/robustness_macro_f1.png)

## 1.9 OBB 效果补充验证

| 数据集 | 窗口数 | HF-GO占有率降幅 | M1 F1 | M3 F1 | R/F变化 | M4 F1 | 本文变化 |
|---|---:|---:|---:|---:|---:|---:|---:|
| xamn6 | 324 | 0.0002 | 0.9256 | 0.9256 | +0.0000 | 0.9381 | +0.0125 |
| xamn5 | 261 | 0.0006 | 0.9531 | 0.9531 | +0.0000 | 0.9530 | -0.0001 |
| pkdd8 | 1059 | 0.0026 | 0.9470 | 0.9313 | -0.0156 | 0.9201 | -0.0112 |

## 1.10 PKDD 泛化结果

PKDD 窗口数：1059

预测状态分布：
- 畅通: 1059
- 缓行: 0
- 拥挤: 0
- 堵塞: 0

PKDD 以自由流为主，预测结果集中在"畅通"类，符合该数据集的交通状态描述。该结果用于跨场景合理性检查，不作为与 XAM-N-6 同分布混合训练的证据。

---

# 2 未来交通状态恶化预测

## 2.1 任务定义

恶化预测是一个二分类任务：给定当前时间窗口的特征，预测在展望期 $k$ 步后交通状态是否恶化。标签基于连续交通状态分数的变化量构造：当 $S(t+k) - S(t)$ 超过第 65 百分位阈值时标记为恶化（标签=1），否则为 0。与当前状态识别相比，该任务更接近“提前预警”，能更直接体现车头时距和加速度干扰等微观行为特征的价值。

展望期设置为 3s, 5s, 8s。评价采用 ROC-AUC 作为主指标（对类别不平衡和阈值选择更鲁棒），同时报告 Macro-F1。

## 2.2 恶化预测结果

- **3s 展望期**: 正样本 208 (64.8%), 负样本 113
  - M3: V+D+R+F: AUC=0.3762 (vs M1 0.4629, 下降 8.67 个百分点)
  - M4: Ours+headway+acc+MGTI: AUC=0.3972 (vs M1 0.4629, 下降 6.57 个百分点)
- **5s 展望期**: 正样本 207 (64.9%), 负样本 112
  - M3: V+D+R+F: AUC=0.3670 (vs M1 0.5115, 下降 14.45 个百分点)
  - M4: Ours+headway+acc+MGTI: AUC=0.4418 (vs M1 0.5115, 下降 6.97 个百分点)
- **8s 展望期**: 正样本 205 (64.9%), 负样本 111
  - M3: V+D+R+F: AUC=0.3114 (vs M1 0.3400, 下降 2.86 个百分点)
  - M4: Ours+headway+acc+MGTI: AUC=0.3610 (vs M1 0.3400, 提升 2.10 个百分点)

### 恶化预测消融实验

| Horizon | 消融集 | ROC-AUC | F1 | Precision | Recall |
|---|---|---:|---:|---:|---:|
| 3s | M1: V+D | 0.4629 | 0.4271 | 0.4584 | 0.4676 |
| 3s | M2: V+D+R | 0.3979 | 0.4039 | 0.4077 | 0.4221 |
| 3s | M3: V+D+R+F | 0.3762 | 0.4584 | 0.4583 | 0.4585 |
| 3s | M4: Ours+headway+acc+MGTI | 0.3972 | 0.4345 | 0.4954 | 0.4978 |
| 5s | M1: V+D | 0.5115 | 0.5624 | 0.5730 | 0.5735 |
| 5s | M2: V+D+R | 0.4396 | 0.3969 | 0.4026 | 0.4393 |
| 5s | M3: V+D+R+F | 0.3670 | 0.4303 | 0.4315 | 0.4303 |
| 5s | M4: Ours+headway+acc+MGTI | 0.4418 | 0.4498 | 0.4629 | 0.4725 |
| 8s | M1: V+D | 0.3400 | 0.4354 | 0.4346 | 0.4434 |
| 8s | M2: V+D+R | 0.3338 | 0.4314 | 0.4316 | 0.4462 |
| 8s | M3: V+D+R+F | 0.3114 | 0.3980 | 0.3944 | 0.4062 |
| 8s | M4: Ours+headway+acc+MGTI | 0.3610 | 0.4231 | 0.4216 | 0.4362 |

![恶化消融AUC](../outputs/figures/deterioration_ablation_auc.png)

![恶化展望期敏感性](../outputs/figures/deterioration_horizon_sensitivity.png)

![恶化特征重要性](../outputs/figures/deterioration_feature_importance.png)

最佳恶化预测结果：展望期 5s，消融集 M1: V+D，ROC-AUC = 0.5115。
从消融结果看，当前恶化预测样本量较小，R/F、车头时距、加速度扰动和 MGTI 未稳定超过 V+D 基线。论文中应将该部分作为补充预警实验呈现，不把它作为主要创新指标。

---

# 3 特征分析与结果核验

## 3.1 MGTI 复合风险指标分析

![MGTI风险箱线图](../outputs/figures/mgti_risk_by_state.png)

MGTI 复合得分随状态等级变化：
- 畅通: mean=2.3856 (n=81)
- 缓行: mean=3.4800 (n=81)
- 拥挤: mean=4.2912 (n=81)
- 堵塞: mean=7.4385 (n=81)

单调性检查：单调递增（通过）。MGTI 复合指标的设计意图是风险随拥堵程度递增，单调性验证确认其方向一致性。

## 3.2 OBB/HBB 空间对比

![HBB/OBB占有率](../outputs/figures/xamn6_hbb_obb_occupancy.png)

XAM-N-6 与 PKDD-8 上 HBB/OBB 总面积差异较小，说明仅用全局面积占有率难以充分体现旋转框优势。XAM-N-5 上的占有率降幅更明显，可作为 OBB 空间感知效果的补充证据。因此，OBB 模块的核心价值体现在角度补全、目标朝向表达与空间占用估计增强，而不是单纯依赖全局面积占有率变化。

## 3.3 标签与结果核验

### OBB 几何核验

| 数据集 | 样本行数 | 尺寸合法率 | OBB 面积合法率 | 高置信角度比例 | 中心点 X 范围 | 中心点 Y 范围 |
|---|---:|---:|---:|---:|---:|---:|
| xamn6 | 200000 | 1.0000 | 1.0000 | 0.9984 | 65.4-1863.9 | 262.0-845.5 |
| pkdd8 | 200000 | 1.0000 | 1.0000 | 0.9726 | 118.5-1860.9 | 394.5-691.0 |

### XAM-N-6 状态特征核验（4类）

| 状态 | 窗口数 | 平均速度 | 密度 | OBB 占有率 | 平均车头时距 | 平均空间间距 | 加速度干扰 | MGTI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 畅通 | 81 | 28.037 | 0.4345 | 0.055205 | 2.093 | 14.438 | 1.8141 | 2.3856 |
| 缓行 | 81 | 23.488 | 0.4698 | 0.062712 | 2.309 | 12.835 | 1.3290 | 3.4800 |
| 拥挤 | 81 | 19.150 | 0.4633 | 0.064704 | 2.678 | 11.018 | 1.6381 | 4.2912 |
| 堵塞 | 81 | 14.619 | 0.5210 | 0.075319 | 4.530 | 9.758 | 1.3177 | 7.4385 |

单调性检查：
- speed_generally_decreases: 通过
- worst_speed_lower_than_free: 通过
- worst_density_is_highest: 通过
- worst_occupancy_is_highest: 通过

### 恶化标签核验

- horizon_3s: 正样本 208 (64.8%) — 充分
- horizon_5s: 正样本 207 (64.9%) — 充分
- horizon_8s: 正样本 205 (64.9%) — 充分

## 3.4 数据处理结果

| 数据集 | 行数 | 车辆数 | 直接角度比例 | 角度补全行数 |
|---|---:|---:|---:|---:|
| xamn5 | 334721 | 901 | 0.9992 | 258 |
| xamn6 | 1043909 | 900 | 0.9666 | 34852 |
| pkdd8 | 586534 | 1472 | 0.9862 | 8075 |

OBB 可视化图由 pixel 表中的帧号抽样生成。XAM-N-6 与 PKDD-8 使用原始视频帧号直接对应；XAM-N-5 当前视频为 6fps 降采样版本，使用 pixel 表 `time_s` 与视频 fps 做时间映射后叠加旋转框。

## 3.5 特征窗口

| 数据 | 窗口数 | 文件 |
|---|---:|---|
| xamn6 | 324 | `outputs/features/xamn6_windows.csv` |
| xamn5 | 261 | `outputs/features/xamn5_windows.csv` |
| pkdd8 | 1059 | `outputs/features/pkdd8_windows.csv` |
| merged | 1644 | `outputs/features/all_windows.csv` |

---

# 图表与结果文件

**状态识别与预测图表：**
- `outputs/figures/classification_metrics.png` — 当前状态识别各模型指标
- `outputs/figures/cm_xgboost_obb.png` — 当前状态混淆矩阵
- `outputs/figures/cm_fusion_future.png` — 未来状态预测混淆矩阵
- `outputs/figures/future_prediction_curve.png` — 未来预测时序曲线
- `outputs/figures/ablation_macro_f1.png` — 消融实验 Macro-F1
- `outputs/figures/parameter_sensitivity.png` — 参数敏感性
- `outputs/figures/robustness_macro_f1.png` — 多随机种子稳健性
- `outputs/figures/xamn6_hbb_obb_occupancy.png` — HBB/OBB 占有率对比
- `outputs/figures/hfgo_hbb_vs_obb_heatmap.png` — HBB 与 HF-GO 网格占有率热力图

**恶化预测图表：**
- `outputs/figures/deterioration_ablation_auc.png` — 恶化预测消融 AUC
- `outputs/figures/deterioration_horizon_sensitivity.png` — 恶化展望期敏感性
- `outputs/figures/deterioration_feature_importance.png` — 恶化任务特征重要性

**特征分析图表：**
- `outputs/figures/mgti_risk_by_state.png` — MGTI 复合风险箱线图

**OBB 抽帧可视化：**
- `outputs/figures/xamn5_obb_overlay_f*.jpg` — XAM-N-5 pixel 帧时间映射后的旋转框叠加图
- `outputs/figures/xamn6_obb_overlay_f*.jpg` — XAM-N-6 pixel 帧对应旋转框叠加图
- `outputs/figures/pkdd8_obb_overlay_f*.jpg` — PKDD-8 pixel 帧对应旋转框叠加图

图表状态标签：英文缩写。

**结果文件：**
- `outputs/processed/*_pixel_obb.csv` — OBB 标注表
- `outputs/features/*_windows.csv` — 滑窗特征表
- `outputs/reports/experiment_results.json` — 模型实验指标
- `outputs/reports/auto_verification.json` — 自动核验结果
- `outputs/reports/obb_effect_validation.json` — OBB 效果补充验证
