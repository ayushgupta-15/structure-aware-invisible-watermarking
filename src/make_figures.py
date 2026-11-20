"""Plot mean BER per attack/variant from results/*_results.csv (run from code/)."""
import csv, glob, sys
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

rows = list(csv.DictReader(open("results/conventional_results.csv")))
try:  # diffusion rows only (the diffusion CSV's conventional rows cover fewer images)
    rows += [r for r in csv.DictReader(open("results/diffusion_results.csv"))
             if r["attack"].startswith("Diffusion")]
except FileNotFoundError:
    pass
attacks = []
agg = defaultdict(list)
for r in rows:
    if r["attack"] not in attacks and r["attack"] != "None":
        attacks.append(r["attack"])
    agg[(r["attack"], r["variant"])].append(float(r["ber"]))
variants = ["Uniform", "Variance", "Edge"]
colors = ["#6b7280", "#2563eb", "#d97706"]
x = np.arange(len(attacks)); w = 0.26
fig, ax = plt.subplots(figsize=(10, 4))
for i, v in enumerate(variants):
    m = [np.mean(agg[(a, v)]) for a in attacks]
    e = [np.std(agg[(a, v)]) / np.sqrt(len(agg[(a, v)])) for a in attacks]
    ax.bar(x + (i - 1) * w, m, w, yerr=e, label=v, color=colors[i], capsize=2)
ax.axhline(0.5, color="k", ls="--", lw=0.8); ax.text(len(attacks) - 0.5, 0.51, "random guess", ha="right", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(attacks, rotation=20, ha="right")
ax.set_ylabel("Mean BER (lower = better)"); ax.set_ylim(0, 0.7); ax.legend(frameon=False)
ax.set_title("Watermark bit error rate by attack (100 images conventional, 10 diffusion; error bars = s.e.m.)", fontsize=10)
fig.tight_layout(); fig.savefig("results/ber_comparison.png", dpi=170)
print("saved results/ber_comparison.png")
