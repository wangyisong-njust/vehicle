#!/usr/bin/env bash
set -euo pipefail

# 不依赖调用时的 cwd，自动定位到项目根
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

python scripts/01_prepare_obb.py --datasets xamn6 xamn5 pkdd8
python scripts/02_extract_features.py --datasets xamn6 xamn5 pkdd8
python scripts/03_run_experiments.py
python scripts/05_auto_verify.py
python scripts/06_validate_obb_effect.py
python scripts/04_make_report.py
