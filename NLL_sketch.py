import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 14,
    "lines.linewidth": 2,
})

# X axis (parameter of interest)
x = np.linspace(-1, 3, 500)

# Beispielkurven (Background PDFs)
y_bestfit = (x - 1.0)**2
y_alt1 = (x - 1.2)**2 * 1.1
y_alt2 = (x - 0.8)**2 * 1.2  # neue zusätzliche Kurve

# Hüllkurve: Minimum aller PDFs
y_env = np.minimum.reduce([y_bestfit, y_alt1, y_alt2])

# Normalisieren (Minimum = 0)
y_bestfit -= np.min(y_bestfit)
y_alt1 -= np.min(y_alt1)
y_alt2 -= np.min(y_alt2)
y_env -= np.min(y_env)

# 1σ Schwelle
sigma_level = 1.0
xmin = x[np.where(y_env <= sigma_level)[0][0]]
xmax = x[np.where(y_env <= sigma_level)[0][-1]]

# Hintergrund-PDFs (gestrichelt, transparent, niedriger zorder)
plt.plot(x, y_bestfit, "g--", alpha=0.8, zorder=1, label="Best fit background PDF")
plt.plot(x, y_alt1, "r--", alpha=0.8, zorder=1, label="Alternative background PDF 1")
plt.plot(x, y_alt2, "b--", alpha=0.8, zorder=1, label="Alternative background PDF 2")

# Hüllkurve oben drauf
plt.plot(x, y_env, "k-", zorder=2, label="Likelihood envelope")

# 1σ Intervall (graues Rechteck)
plt.hlines(sigma_level, xmin, xmax, color="gray", linestyle="--", zorder=0)
plt.vlines([xmin, xmax], 0, sigma_level, color="gray", linestyle="--", zorder=0, label=r"$1\sigma$ interval")

# Achsen etc.
plt.xlabel("Parameter of interest")
plt.ylabel(r"$-2\Delta LL$")
plt.ylim(0, 4.2)
plt.xlim(-1, 3)
plt.legend(frameon=True, loc="upper center", fontsize=10)

plt.tight_layout()
plt.savefig("NLL_sketch.pdf")