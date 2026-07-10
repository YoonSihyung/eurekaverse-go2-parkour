#!/usr/bin/env python
"""GPT가 생성한 지형들을 heightfield 이미지(PNG)로 렌더링 (GPU 불필요).

사용법:
  python render_terrains.py outputs/run_eurekaverse/<RUN_ID>/terrain_iter-0_run-0.py
  python render_terrains.py <RUN_ID의 terrain_*.py 아무거나>  # 파일당 PNG 1장 (지형 10종 격자)
출력: 같은 폴더에 <파일명>.png
"""
import sys, importlib.util
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_module(path):
    spec = importlib.util.spec_from_file_location("terrain_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def main(terrain_file):
    terrain_file = Path(terrain_file)
    mod = load_module(terrain_file)
    # set_terrain_N 함수들 수집
    fns = [(n, f) for n, f in vars(mod).items()
           if n.startswith("set_terrain_") and callable(f)]
    fns.sort(key=lambda x: int(x[0].split("_")[-1]) if x[0].split("_")[-1].isdigit() else 999)
    n = len(fns)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 2.2*rows))
    axes = np.atleast_2d(axes)
    for i, (name, fn) in enumerate(fns):
        ax = axes[i // cols][i % cols]
        try:
            hf, goals = fn(18.0, 4.0, 0.05, 1.0)  # 셀 18x4m, 해상도 0.05, 난이도 1.0 (최고)
            im = ax.imshow(hf.T, origin="lower", cmap="terrain", aspect="auto", vmin=-1.0, vmax=1.0)
            g = np.asarray(goals, dtype=float)
            ax.plot(g[:, 0], g[:, 1], "r.-", markersize=6, linewidth=0.8)
            doc = (fn.__doc__ or "").strip().split("\n")[0][:60]
            ax.set_title(f"{name}\n{doc}", fontsize=7)
        except Exception as e:
            ax.set_title(f"{name}: {type(e).__name__}", fontsize=7, color="red")
        ax.set_xticks([]); ax.set_yticks([])
    for j in range(n, rows*cols):
        axes[j // cols][j % cols].axis("off")
    out = terrain_file.with_suffix(".png")
    plt.tight_layout()
    plt.savefig(out, dpi=110)
    print(f"저장됨: {out}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    for f in sys.argv[1:]:
        main(f)
