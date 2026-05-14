#!/usr/bin/env python
"""环境自检：检查关键依赖是否安装，以及 GPU/CUDA 是否可用。"""
from __future__ import annotations

import argparse
import importlib
import sys

# Import matplotlib first so PIL/Lerc resolves conda's libstdc++ before
# xgboost/torch can pull in an older system copy through CUDA paths.
REQUIRED = ["matplotlib", "numpy", "pandas", "scipy", "sklearn", "xgboost", "torch", "cv2", "tqdm"]

MODULE_VERSION = {
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether core runtime dependencies can be imported.")
    parser.parse_args()

    missing: list[str] = []
    for mod in REQUIRED:
        try:
            m = importlib.import_module(mod)
            version = getattr(m, "__version__", "unknown")
            print(f"[OK]   {MODULE_VERSION.get(mod, mod):<16} {version}")
        except Exception as exc:
            print(f"[MISS] {MODULE_VERSION.get(mod, mod):<16} {exc}")
            missing.append(mod)

    try:
        import torch
        print(f"[INFO] torch.cuda.is_available() = {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[INFO] device count = {torch.cuda.device_count()}, name = {torch.cuda.get_device_name(0)}")
    except Exception:
        pass

    if missing:
        print(f"\n缺少依赖：{missing}")
        print("请先按 README 配置环境，再运行 scripts/run_all.sh。")
        return 1
    print("\n环境自检通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
