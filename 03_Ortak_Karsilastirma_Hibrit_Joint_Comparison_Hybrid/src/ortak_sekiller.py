"""Ortak karşılaştırma şekilleri (TR/EN): yakınsama, kutu grafiği, yöntem akışı."""
import os, sys, json
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
DIL = sys.argv[1]; T = (lambda tr, en: tr) if DIL == "tr" else (lambda tr, en: en)
B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(B), "04_Sekiller_Figures", DIL); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 10, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300, "savefig.bbox": "tight"})
R = {"GA": "#2a6fdb", "ACO_R": "#1a9e5c", "PSO": "#e07b00", "GWO": "#7b3fa0",
     "GA-ACO_R": "#00a5b5", "PSO-GWO": "#c0392b", "Rastgele Arama": "#8a8a8a"}
LS = {"GA-ACO_R": "--", "PSO-GWO": "--"}
AD = lambda a: T(a, "Random search") if a == "Rastgele Arama" else a
BAS = {"kalkis": T("Kalkış (GA/ACO_R problemi)", "Ascent"), "inis": T("İniş (PSO/GWO problemi)", "Landing")}

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for i, p in enumerate(["kalkis", "inis"]):
    yk = json.load(open(os.path.join(B, "sonuclar_results", {"kalkis": "kalkis_ascent", "inis": "inis_landing"}[p], "yakinsama.json")))
    iz = np.array(yk["izgara"]); Jmin = min(min(min(e) for e in E) for E in yk["egriler"].values())
    for a, E in yk["egriler"].items():
        E = np.array(E); m, s = E.mean(0), E.std(0)
        ax[i].plot(iz, m, color=R[a], ls=LS.get(a, "-"), lw=1.6, label=AD(a))
        ax[i].fill_between(iz, m - s, m + s, color=R[a], alpha=0.10)
    ax[i].set(xlabel=T("Amaç fonksiyonu değerlendirme sayısı", "Objective-function evaluations"),
              ylabel=T("En iyi J (10 koşu ort. ± std)", "Best J (mean ± std, 10 runs)"),
              ylim=(Jmin * 0.995, Jmin * (1.12 if p == "kalkis" else 1.10)), title=f"({'ab'[i]}) {BAS[p]}")
ax[0].legend(frameon=False, fontsize=8, ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "yakinsama.png")); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
for i, p in enumerate(["kalkis", "inis"]):
    df = pd.read_csv(os.path.join(B, "sonuclar_results", {"kalkis": "kalkis_ascent", "inis": "inis_landing"}[p], "tum_kosular.csv"))
    adl = [a for a in R if a != "Rastgele Arama"]
    Js = df.J_egitim.min()
    bp = ax[i].boxplot([(df[df.yontem == a].J_egitim.values - Js) / Js * 100 + 1e-6 for a in adl], patch_artist=True, widths=0.55)
    ax[i].set_yscale("log")
    for pa, a in zip(bp["boxes"], adl): pa.set_facecolor(R[a]); pa.set_alpha(0.55)
    ax[i].set_xticks(range(1, len(adl) + 1), adl, fontsize=8.5)
    ax[i].set(ylabel=T("En iyi J'ye göre fark [%] (log)", "Gap to best J [%] (log)"), title=f"({'ab'[i]}) {BAS[p]}")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "kutu.png")); plt.close(fig)

# Yöntem akış şeması
fig, ax = plt.subplots(figsize=(11, 2.0)); ax.axis("off"); ax.set_xlim(-0.05, 11.2); ax.set_ylim(0, 2.3)
kut = [T("Kaggle IFT-6/7/8\nOCR telemetrisi", "Kaggle IFT-6/7/8\nOCR telemetry"),
       T("Ön işleme ve\nreferans profil\n(v_ref, γ, q)", "Pre-processing &\nreference profile\n(v_ref, γ, q)"),
       T("Kalkış / iniş\nsimülasyonu +\n2 fazlı PID", "Ascent / landing\nsimulation +\n2-phase PID"),
       T("Monte Carlo\namaç fonksiyonu J", "Monte Carlo\nobjective J"),
       T("GA · ACO_R · PSO · GWO\nGA-ACO_R · PSO-GWO\n(1800 değ., 10 koşu)", "GA · ACO_R · PSO · GWO\nGA-ACO_R · PSO-GWO\n(1800 evals, 10 runs)"),
       T("200 yeni senaryoda\ntest + istatistik", "Test on 200 unseen\nscenarios + statistics")]
w, g = 1.66, 0.22
for k, s in enumerate(kut):
    x = 0.05 + k * (w + g)
    ax.add_patch(FancyBboxPatch((x, 0.55), w, 1.2, boxstyle="round,pad=0.03", fc="#eef3fb" if k != 4 else "#fdeeee", ec="#1f4e8c", lw=1.1))
    ax.text(x + w / 2, 1.15, s, ha="center", va="center", fontsize=7.8)
    if k < len(kut) - 1:
        ax.annotate("", (x + w + g - 0.02, 1.15), (x + w + 0.02, 1.15), arrowprops=dict(arrowstyle="->", lw=1.2))
ax.annotate("", (0.05 + 3 * (w + g) + w / 2, 0.52), (0.05 + 4 * (w + g) + w / 2, 0.52),
            arrowprops=dict(arrowstyle="->", lw=1, ls="--", connectionstyle="arc3,rad=-0.22", color="#c0392b"))
ax.text(0.05 + 3.5 * (w + g) + w / 2, 0.02, T("aday kazançlar x = [Kp1..Kd2]", "candidate gains x = [Kp1..Kd2]"),
        ha="center", fontsize=8, color="#c0392b")
fig.savefig(os.path.join(OUT, "yontem_akisi.png"), pad_inches=0.05); plt.close(fig)
print("ok")
