import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import scipy.ndimage as ndimage

# Read data from pca.dat file
data = np.loadtxt('pca.dat', skiprows=1)
pca_modes = data[:, 1:]  # Skip frame numbers

# Use first two principal components for 3D free energy landscape
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

# Calculate 3D free energy landscape
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

# Create 3D plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Define custom colormap: blue (min) to red (max) with intermediate colors
colors = [(0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)]
custom_cmap = LinearSegmentedColormap.from_list('blue_to_red', colors, N=256)

# Set z-axis and colorbar limit
ENERGY_LIMIT = 7.2

# Option 1: Clip the data to show peaks truncated at 2.4
F_clipped = np.clip(F, F.min(), ENERGY_LIMIT)

# Plot surface with clipped data
surf = ax.plot_surface(X, Y, F_clipped, cmap=custom_cmap, rstride=1, cstride=1, 
                       linewidth=0.1, edgecolor='gray', alpha=0.95, 
                       antialiased=True, vmin=F.min(), vmax=ENERGY_LIMIT)

# Add lighting and view for silky smooth rendering
ax.view_init(elev=25, azim=45)
ax.dist = 10

# Labels and title with improved fonts
ax.set_xlabel('PC1', fontsize=14, fontweight='bold')
ax.set_ylabel('PC2', fontsize=14, fontweight='bold')
ax.set_zlabel('Free Energy (kcal/mol)', fontsize=14, fontweight='bold')
ax.set_title('Your Data Gibbs Energy Landscape', fontsize=16, fontweight='bold')

# Adjust axis limits with 2.4 kcal/mol limit
ax.set_xlim(X.min(), X.max())
ax.set_ylim(Y.min(), Y.max())
ax.set_zlim(F.min(), ENERGY_LIMIT)  # Set z-axis limit to 2.4

# Add grid for better orientation
ax.grid(True, linestyle='--', alpha=0.5)

# Add color bar with decimal-only ticks limited to 2.4
f_min, f_max = F.min(), F.max()
num_ticks = 8  # Number of ticks on colorbar
step_size = (ENERGY_LIMIT - f_min) / (num_ticks - 1)  # Evenly spaced ticks
cbar_ticks = np.linspace(f_min, ENERGY_LIMIT, num_ticks)
cbar = plt.colorbar(surf, label='Gibbs Energy (kcal/mol)', shrink=0.5, aspect=10)
cbar.set_ticks(cbar_ticks)
cbar.ax.set_yticklabels([f'{tick:.1f}' for tick in cbar_ticks])
cbar.ax.tick_params(labelsize=12)

# Add a label indicating the limit
#cbar.ax.text(1.5, 1.02, f'Max: {ENERGY_LIMIT} kcal/mol', 
#             transform=cbar.ax.transAxes, fontsize=10, 
#             verticalalignment='bottom')

# Show plot
plt.tight_layout()
plt.show()

# Print summary of energy range
print("\n" + "="*50)
print("ENERGY RANGE SUMMARY:")
print("="*50)
print(f"Minimum free energy: {F.min():.4f} kcal/mol")
print(f"Maximum free energy in data: {F.max():.4f} kcal/mol")
print(f"Display limit applied: {ENERGY_LIMIT} kcal/mol")
print(f"Percentage of data above limit: {100 * np.sum(F > ENERGY_LIMIT) / F.size:.2f}%")
