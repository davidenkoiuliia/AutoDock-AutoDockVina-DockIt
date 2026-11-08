import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(".")
OUT_DIR = Path("output")
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
    "axes.linewidth": 1.0,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.8,
})

def read_rmsf(path):
    x, y = [], []
    with open(path, "r") as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            nums = []
            for tok in line.split():
                try:
                    nums.append(float(tok))
                except ValueError:
                    pass
            if len(nums) >= 2:
                x.append(int(round(nums[0])))
                y.append(nums[1])
    return np.array(x), np.array(y)

def plot_rmsf(path, out_png, title, xlabel, color):
    x, y = read_rmsf(path)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, y, lw=2, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("RMSF, Å")
    ax.grid(True)
    fig.savefig(OUT_DIR / out_png)
    plt.close(fig)

plot_rmsf(
    DATA_DIR / "rmsf_protein.dat",
    "rmsf_protein.png",
    "RMSF белка по остаткам",
    "Номер остатка",
    "#2ca02c"
)

plot_rmsf(
    DATA_DIR / "rmsf_ligand.dat",
    "rmsf_ligand.png",
    "RMSF лиганда по атомам",
    "Номер атома",
    "#ff7f0e"
)

print("RMSF plots saved in output/")