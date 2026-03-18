import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.colors as mcolors

# ── crystal structure ──────────────────────────────────────────────────────────

def init_supercell(a, nx=9, ny=5, nz=5):
    L_x = a
    L_y = np.sqrt(3) * a
    L_z = np.sqrt(8 / 3) * a
    basis_frac = np.array([
        [1/4, 5/6, 1/4],
        [1/4, 1/6, 3/4],
        [3/4, 1/3, 1/4],
        [3/4, 2/3, 3/4],
    ], dtype=float)

    cell_length = np.array([L_x, L_y, L_z])
    basis_cart  = basis_frac * cell_length

    pos = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                shift = np.array([i * L_x, j * L_y, k * L_z])
                for atom in basis_cart:
                    pos.append(atom + shift)

    pos = np.array(pos, dtype=float)
    box = np.array([nx * L_x, ny * L_y, nz * L_z], dtype=float)
    return pos, box


# ── bond finder ───────────────────────────────────────────────────────────────

def find_bonds(pos, bond_cutoff):
    """Return list of (i, j) index pairs closer than bond_cutoff."""
    bonds = []
    n = len(pos)
    # Use a simple distance check — fast enough for ~2000 atoms
    for i in range(n):
        diffs = pos[i+1:] - pos[i]
        dists = np.linalg.norm(diffs, axis=1)
        js    = np.where(dists < bond_cutoff)[0] + i + 1
        bonds.extend((i, j) for j in js)
    return bonds


# ── main ──────────────────────────────────────────────────────────────────────

a   = 1.0
pos, box = init_supercell(a, nx=4, ny=3, nz=3)   # smaller for clarity

# nearest-neighbour distance in this lattice ≈ a * sqrt(2/3) * … tuned visually
bond_cutoff = 0.72 * a * np.sqrt(3)

bonds = find_bonds(pos, bond_cutoff)

# ── colour atoms by z-depth (gives a "crystal glow" look) ─────────────────────
z_norm = (pos[:, 2] - pos[:, 2].min()) / (pos[:, 2].max() - pos[:, 2].min() + 1e-9)

# palette: ForestGreen (low-z) → RoyalBlue (mid, dominant) → IndianRed (high-z)
import matplotlib.colors as mc
_cmap_colors = ["forestgreen", "royalblue", "royalblue", "indianred"]
_cmap_nodes  = [0.0, 0.35, 0.65, 1.0]
atom_cmap = mc.LinearSegmentedColormap.from_list("crystal", list(zip(_cmap_nodes, _cmap_colors)))
atom_colors = atom_cmap(z_norm)

# ── figure ────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(14, 10), facecolor="white")
ax  = fig.add_subplot(111, projection="3d", facecolor="white")

# --- bonds (thin grey sticks) ---
if bonds:
    segs = [(pos[i], pos[j]) for i, j in bonds]
    lc = Line3DCollection(
        segs,
        linewidths=1.2,
        colors=(0.55, 0.55, 0.60, 0.45),
        zorder=1,
    )
    ax.add_collection3d(lc)

# --- atoms ---
x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]

# outer glow: large, very transparent sphere
ax.scatter(x, y, z,
           s=600,
           c=z_norm, cmap=atom_cmap,
           alpha=0.07,
           edgecolors="none",
           depthshade=False,
           zorder=2)

# mid halo
ax.scatter(x, y, z,
           s=280,
           c=z_norm, cmap=atom_cmap,
           alpha=0.18,
           edgecolors="none",
           depthshade=False,
           zorder=3)

# core atom
sc = ax.scatter(x, y, z,
                s=220,
                c=z_norm, cmap=atom_cmap,
                alpha=0.97,
                edgecolors=(0.15, 0.05, 0.3, 0.7),
                linewidths=0.6,
                depthshade=True,
                zorder=4)

# ── cosmetics ──────────────────────────────────────────────────────────────────

ax.set_xlabel("x  (Å)", color="#333333", labelpad=8, fontsize=9)
ax.set_ylabel("y  (Å)", color="#333333", labelpad=8, fontsize=9)
ax.set_zlabel("z  (Å)", color="#333333", labelpad=8, fontsize=9)

ax.tick_params(colors="#444444", labelsize=7)
for spine in ax.spines.values():
    spine.set_edgecolor("#cccccc")

ax.xaxis.pane.fill = True
ax.yaxis.pane.fill = True
ax.zaxis.pane.fill = True
ax.xaxis.pane.set_facecolor("#f5f5f7")
ax.yaxis.pane.set_facecolor("#f5f5f7")
ax.zaxis.pane.set_facecolor("#f5f5f7")
ax.xaxis.pane.set_edgecolor("#dddddd")
ax.yaxis.pane.set_edgecolor("#dddddd")
ax.zaxis.pane.set_edgecolor("#dddddd")
ax.grid(True, color="#cccccc", linewidth=0.5, linestyle="--", alpha=0.6)

ax.set_title(
    "Gadolinium crystal model ·  {0} atoms".format(len(pos)),
    color="#1a1a2e", fontsize=13, fontweight="bold", pad=14,
    fontfamily="monospace",
)

cbar = fig.colorbar(sc, ax=ax, shrink=0.45, pad=0.02, aspect=20)
cbar.set_label("z-depth", color="#333333", fontsize=8)
cbar.ax.yaxis.set_tick_params(color="#333333", labelsize=7)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#333333")
cbar.outline.set_edgecolor("#cccccc")

ax.view_init(elev=22, azim=35)

plt.tight_layout()
plt.savefig("./figs/crystal.png", dpi=180, bbox_inches="tight",
            facecolor='white')
plt.show()
print("Saved → crystal.png")