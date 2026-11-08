import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Папки и параметры симуляции
DATA_DIR = Path(".")      # где лежат файлы .dat
OUT_DIR = Path("output")
OUT_DIR.mkdir(exist_ok=True)

N_FRAMES = 1000         # число фреймов в траектории
TOTAL_NS = 500.0        # 500 ns
NS_PER_FRAME = TOTAL_NS / N_FRAMES   # 0.5 ns на фрейм

FILENAMES = {
    "rmsd_protein": "rmsd_protein.dat",
    "rmsd_ligand":  "rmsd_ligand.dat",
    "rmsf_protein": "rmsf_protein.dat",
    "rmsf_ligand":  "rmsf_ligand.dat",
    "rg_protein":   "rg_protein.dat",
}

# -------------------- общая функция чтения 2 колонок --------------------
def read_2col(path):
    x_vals, y_vals = [], []
    with open(path, "r") as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            cols = line.split()
            nums = []
            for c in cols:
                try:
                    nums.append(float(c))
                except ValueError:
                    pass
            if len(nums) >= 2:
                x_vals.append(nums[0])
                y_vals.append(nums[1])
    return np.array(x_vals), np.array(y_vals)

def frame_to_ns(frame):
    return (frame - 1) * NS_PER_FRAME

# -------------------- загрузка данных --------------------
x_pro, y_pro   = read_2col(DATA_DIR / FILENAMES["rmsd_protein"])
x_lig, y_lig   = read_2col(DATA_DIR / FILENAMES["rmsd_ligand"])
x_rmsf_pro, y_rmsf_pro = read_2col(DATA_DIR / FILENAMES["rmsf_protein"])
x_rmsf_lig, y_rmsf_lig = read_2col(DATA_DIR / FILENAMES["rmsf_ligand"])
x_rg, y_rg     = read_2col(DATA_DIR / FILENAMES["rg_protein"])

time_ns_pro = frame_to_ns(x_pro)
time_ns_lig = frame_to_ns(x_lig)
time_ns_rg  = frame_to_ns(x_rg)

# -------------------- настройки стиля (без привязки к Arial) --------------------
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "axes.linewidth": 1.0,
    "axes.edgecolor": "black",
    "axes.facecolor": "white",
    "grid.alpha": 0.3,
    "grid.linewidth": 0.8,
    "figure.figsize": (7, 4),
})

# Цвета
colors = {
    "protein": "#1f77b4",
    "ligand":  "#d62728",
    "rmsf":    "#2ca02c",
    "rg":      "#9467bd",
}

# ---------------------------- 1. RMSD белка ----------------------------
fig, ax = plt.subplots()
ax.plot(time_ns_pro, y_pro, color=colors["protein"], lw=2, label="RMSD")
ax.set_xlabel("Время, нс")
ax.set_ylabel("RMSD, Å")
ax.set_title("RMSD белка во время симуляции")
ax.grid(True, color="0.85")
ax.legend(loc="best", frameon=False, prop={"size": 10})
fig.tight_layout()
fig.savefig(OUT_DIR / "rmsd_protein.png")
plt.close(fig)

# ---------------------------- 2. RMSD лиганда ----------------------------
fig, ax = plt.subplots()
ax.plot(time_ns_lig, y_lig, color=colors["ligand"], lw=2, label="RMSD")
ax.set_xlabel("Время, нс")
ax.set_ylabel("RMSD, Å")
ax.set_title("RMSD лиганда во время симуляции")
ax.grid(True, color="0.85")
ax.legend(loc="best", frameon=False, prop={"size": 10})
fig.tight_layout()
fig.savefig(OUT_DIR / "rmsd_ligand.png")
plt.close(fig)

# ---------------------------- 3. Rg белка ----------------------------
fig, ax = plt.subplots()
ax.plot(time_ns_rg, y_rg, color=colors["rg"], lw=2, label="Rg")
ax.set_xlabel("Время, нс")
ax.set_ylabel("Rg, Å")
ax.set_title("Радиус инерции белка во время симуляции")
ax.grid(True, color="0.85")
ax.legend(loc="best", frameon=False, prop={"size": 10})
fig.tight_layout()
fig.savefig(OUT_DIR / "rg_protein.png")
plt.close(fig)

print("Done. Plots saved in output/")