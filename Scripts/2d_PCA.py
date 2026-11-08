import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import scipy.ndimage as ndimage

# Read data from pca.dat file
data = np.loadtxt('pca.dat', skiprows=1)
pca_modes = data[:, 1:]  # Skip frame numbers

# Use first two principal components for free energy landscape
pc1 = pca_modes[:, 0]
pc2 = pca_modes[:, 1]

def calculate_free_energy_2d(x, y, temp=300, nbins=200, sigma=2.0):
    """
    Calculate 2D free energy landscape from PCA data in kcal/mol
    """
    hist, xedges, yedges = np.histogram2d(x, y, bins=nbins, density=True)
    hist = np.where(hist > 0, hist, 1e-10)
    R = 1.987  # cal/mol·K (for kcal/mol)
    free_energy = -R * temp * np.log(hist) / 1000  # kcal/mol
    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2
    X, Y = np.meshgrid(x_centers, y_centers)
    free_energy_smooth = ndimage.gaussian_filter(free_energy.T, sigma=sigma)
    free_energy_smooth = free_energy_smooth - np.min(free_energy_smooth)  # Shift to start at 0
    return X, Y, free_energy_smooth

# Calculate free energy landscape
X, Y, F = calculate_free_energy_2d(pc1, pc2, temp=300, nbins=200, sigma=2.0)

# Find the top 10 minimal free energy values
F_flat = F.flatten()
F_flat = F_flat[np.isfinite(F_flat)]
top_10_indices = np.argsort(F_flat)[:10]
top_10_values = F_flat[top_10_indices]

# Print the top 10 minimal free energy values
print("Top 10 Minimal Free Energy Values (kcal/mol):")
print("-" * 40)
for i, value in enumerate(top_10_values, 1):
    print(f"{i}. {value:.4f}")

# Parameters for 2D plot
ENERGY_LIMIT = 2.52  # Same limit as in 3D plot
F_clipped = np.clip(F, F.min(), ENERGY_LIMIT)

# Define custom colormap: blue (min) to red (max)
colors = [(0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)]
custom_cmap = LinearSegmentedColormap.from_list('blue_to_red', colors, N=256)

# Create 2D plot with contour and filled contour
fig = plt.figure(figsize=(12, 10))

# Option 1: Filled contour plot (recommended)
plt.subplot(2, 2, 1)
contourf = plt.contourf(X, Y, F_clipped, levels=50, cmap=custom_cmap, 
                         vmin=F.min(), vmax=ENERGY_LIMIT)
plt.colorbar(contourf, label='Free Energy (kcal/mol)', ticks=np.linspace(F.min(), ENERGY_LIMIT, 8))
plt.xlabel('PC1', fontsize=12, fontweight='bold')
plt.ylabel('PC2', fontsize=12, fontweight='bold')
plt.title('Filled Contour Plot (2D Energy Landscape)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')

# Option 2: Contour lines only
plt.subplot(2, 2, 2)
contour = plt.contour(X, Y, F_clipped, levels=20, cmap=custom_cmap, 
                      vmin=F.min(), vmax=ENERGY_LIMIT)
plt.colorbar(contour, label='Free Energy (kcal/mol)', ticks=np.linspace(F.min(), ENERGY_LIMIT, 8))
plt.xlabel('PC1', fontsize=12, fontweight='bold')
plt.ylabel('PC2', fontsize=12, fontweight='bold')
plt.title('Contour Lines Only', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')

# Option 3: Filled contour with contour lines on top
plt.subplot(2, 2, 3)
contourf2 = plt.contourf(X, Y, F_clipped, levels=50, cmap=custom_cmap,
                         vmin=F.min(), vmax=ENERGY_LIMIT, alpha=0.85)
contour2 = plt.contour(X, Y, F_clipped, levels=15, colors='black', 
                       linewidths=0.5, alpha=0.7, vmin=F.min(), vmax=ENERGY_LIMIT)
plt.colorbar(contourf2, label='Free Energy (kcal/mol)', ticks=np.linspace(F.min(), ENERGY_LIMIT, 8))
plt.xlabel('PC1', fontsize=12, fontweight='bold')
plt.ylabel('PC2', fontsize=12, fontweight='bold')
plt.title('Filled Contour with Lines', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')

# Option 4: High-quality filled contour with scatter plot of PCA data
plt.subplot(2, 2, 4)
contourf3 = plt.contourf(X, Y, F_clipped, levels=50, cmap=custom_cmap,
                         vmin=F.min(), vmax=ENERGY_LIMIT, alpha=0.8)
# Add scatter plot of original PCA data points
scatter = plt.scatter(pc1, pc2, c='black', s=1, alpha=0.3, label='PCA frames')
plt.colorbar(contourf3, label='Free Energy (kcal/mol)', ticks=np.linspace(F.min(), ENERGY_LIMIT, 8))
plt.xlabel('PC1', fontsize=12, fontweight='bold')
plt.ylabel('PC2', fontsize=12, fontweight='bold')
plt.title('Energy Landscape with Data Points', fontsize=14, fontweight='bold')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.show()

# Create a single high-quality 2D plot (best for publication)
fig2 = plt.figure(figsize=(10, 8))
contourf_main = plt.contourf(X, Y, F_clipped, levels=100, cmap=custom_cmap,
                              vmin=F.min(), vmax=ENERGY_LIMIT)
contour_main = plt.contour(X, Y, F_clipped, levels=20, colors='black', 
                           linewidths=0.8, alpha=0.6, vmin=F.min(), vmax=ENERGY_LIMIT)

# Add energy minima markers
min_energy_idx = np.unravel_index(np.argmin(F), F.shape)

# Add top 5 minima (excluding global minimum)
F_copy = F.copy()
F_copy[min_energy_idx] = np.inf
for i in range(5):
    local_min_idx = np.unravel_index(np.argmin(F_copy), F_copy.shape)
    if F_copy[local_min_idx] < ENERGY_LIMIT:
        plt.plot(X[0, local_min_idx[1]], Y[local_min_idx[0], 0], 
                'g*', markersize=10, markeredgecolor='black', 
                markeredgewidth=1)
        F_copy[local_min_idx] = np.inf

# Customize plot
cbar = plt.colorbar(contourf_main, label='Free Energy (kcal/mol)', 
                    ticks=np.linspace(F.min(), ENERGY_LIMIT, 10))
cbar.ax.set_yticklabels([f'{tick:.1f}' for tick in np.linspace(F.min(), ENERGY_LIMIT, 10)])
cbar.ax.tick_params(labelsize=11)

plt.xlabel('Principal Component 1 (PC1)', fontsize=14, fontweight='bold')
plt.ylabel('Principal Component 2 (PC2)', fontsize=14, fontweight='bold')
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()

# Print summary
print("\n" + "="*50)
print("ENERGY RANGE SUMMARY:")
print("="*50)
print(f"Minimum free energy: {F.min():.4f} kcal/mol")
print(f"Maximum free energy in data: {F.max():.4f} kcal/mol")
print(f"Display limit applied: {ENERGY_LIMIT} kcal/mol")
print(f"Percentage of data above limit: {100 * np.sum(F > ENERGY_LIMIT) / F.size:.2f}%")
print("\nGlobal minimum location:")
print(f"PC1 = {X[0, min_energy_idx[1]]:.4f}")
print(f"PC2 = {Y[min_energy_idx[0], 0]:.4f}")