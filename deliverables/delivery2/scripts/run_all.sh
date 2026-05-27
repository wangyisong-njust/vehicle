#!/usr/bin/env bash
# 一键复现脚本：特征提取 → 状态标签生成 → XGBoost 分类
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
cd "${ROOT}"

python scripts/01_extract_features.py
python scripts/02_build_state_labels.py
python scripts/03_run_classification.py

echo ""
echo "[DONE] 全部步骤完成。结果文件："
echo "  - outputs/features/*_windows.csv"
echo "  - outputs/features/*_grid_tensors.npz"
echo "  - outputs/labels/state_labels.csv"
echo "  - outputs/reports/{feature_summary,labels_summary,classification_results}.json"
echo "  - outputs/figures/{state_distribution,confusion_matrix,feature_importance}.png"
