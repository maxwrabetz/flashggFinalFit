import matplotlib.pyplot as plt
import matplotlib.patches as patches

color_reco = "#4C72B0"
color_fid = "#DD8452"
color_overlap = "#55A868"

fig, ax = plt.subplots(figsize=(7, 6))

# Outer box
total = patches.Rectangle((-6, -4), 12, 8, linewidth=2,
                          edgecolor="black", facecolor="lightgray", alpha=0.15,
                          label="Total phase space")
ax.add_patch(total)

# rectangle
reco = patches.Rectangle((-4, -2), 8, 4, linewidth=2,
                         edgecolor=color_reco, facecolor=color_reco, alpha=0.35,
                         label="Reconstruction level")
ax.add_patch(reco)

# ellipse
fiducial = patches.Ellipse((0, 0), width=9, height=5,
                           linewidth=2, edgecolor=color_fid, facecolor=color_fid,
                           alpha=0.35, label="Fiducial phase space")
ax.add_patch(fiducial)

ax.text(0, 3.6, "Total phase space", color="black", fontsize=13,
        ha="center", va="bottom")
ax.text(3.6, -1.65, "OOA", color="black", fontsize=12,
        ha="center", va="top")
ax.text(0, 2.1, "NR", color="black", fontsize=12,
        ha="center", va="bottom")

ax.set_xlim(-6.5, 6.5)
ax.set_ylim(-4.5, 4.5)
ax.set_aspect('equal')
ax.axis("off")
ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig("phase_space.pdf", bbox_inches="tight")
