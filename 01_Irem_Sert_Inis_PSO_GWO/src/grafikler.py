"""
Bildiri şekilleri. Önce deneyler.py çalıştırılmalı.
Çalıştırma: python grafikler.py   ->  ../sonuclar/sekiller/*.png (300 dpi)
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C
import model as M
from referans import veri_yukle
from deneyler import MANUEL, N_TEST, TEST_SEED

REF = M.REF
RENK = {"PSO": "#e07b00", "GWO": "#2a6fdb",
        "Rastgele Arama": "#8a8a8a", "Manuel PID": "#c0392b", "IFT-8": "#222222"}
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight"})
os.makedirs(C.SEKIL_KLASORU, exist_ok=True)


def kaydet(fig, ad):
    fig.savefig(os.path.join(C.SEKIL_KLASORU, ad))
    plt.close(fig)
    print("kaydedildi:", ad)


df = pd.read_csv(os.path.join(C.SONUC_KLASORU, "tum_kosular.csv"))
with open(os.path.join(C.SONUC_KLASORU, "yakinsama.json")) as fp:
    yk = json.load(fp)
en_iyi = {a: df[df.yontem == a].sort_values("J_egitim").iloc[0][C.PARAM_ADLARI].values.astype(float)
          for a in df.yontem.unique()}

# ---- Şekil 1: IFT-8 verisi ve geri hesaplanan irtifa
t, v, h, d = veri_yukle()
ocr_h = d["booster_altitude"].astype(float).values * 1000
fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
ax[0].plot(t, v, "o-", ms=3, color=RENK["IFT-8"])
ax[0].axvspan(0, C.T_SWITCH, color="#f5b041", alpha=0.25, label="13 motor")
ax[0].axvspan(C.T_SWITCH, t[-1], color="#85c1e9", alpha=0.25, label="3 motor")
ax[0].set(xlabel="Ateşlemeden itibaren süre [s]", ylabel="Hız [m/s]", title="(a) IFT-8 Booster hızı")
ax[0].legend(frameon=False)
ax[1].plot(t, h, "o-", ms=3, color=RENK["IFT-8"], label="Hızdan integre edilen")
ax[1].step(t, ocr_h, where="post", color="#c0392b", alpha=0.7, label="OCR (tam sayı km)")
ax[1].set(xlabel="Ateşlemeden itibaren süre [s]", ylabel="İrtifa [m]", title="(b) İrtifa geri hesaplama")
ax[1].legend(frameon=False)
kaydet(fig, "sekil1_ift8_veri.png")

# ---- Şekil 2: yakınsama eğrileri
iz = np.array(yk["izgara"])
fig, ax = plt.subplots(figsize=(5.5, 3.6))
for ad, E in yk["egriler"].items():
    E = np.array(E)
    m, s = E.mean(0), E.std(0)
    ax.plot(iz, m, color=RENK[ad], lw=1.8, label=ad)
    ax.fill_between(iz, m - s, m + s, color=RENK[ad], alpha=0.15)
ax.set(xlabel="Amaç fonksiyonu değerlendirme sayısı", ylabel="En iyi J (10 koşu ort. ± std)",
       ylim=(1.33, 1.70), title="Yakınsama karşılaştırması")
ax.legend(frameon=False)
kaydet(fig, "sekil2_yakinsama.png")

# ---- Şekil 3: kutu grafiği
fig, ax = plt.subplots(figsize=(5.5, 3.4))
adlar = list(yk["egriler"].keys())
veri = [df[df.yontem == a]["J_egitim"].values for a in adlar]
bp = ax.boxplot(veri, patch_artist=True, widths=0.5)
ax.set_xticks(range(1, len(adlar) + 1), adlar)
for p, a in zip(bp["boxes"], adlar):
    p.set_facecolor(RENK[a]); p.set_alpha(0.5)
ax.set(ylabel="Son J değeri", title=f"{C.N_RUNS} bağımsız koşu sonunda J dağılımı")
kaydet(fig, "sekil3_kutu.png")

# en iyi yöntem (eğitim J'si en düşük olan)
EN = df.groupby("yontem")["J_egitim"].min().drop("Rastgele Arama").idxmin()
RENK["En iyi"] = RENK[EN]

# ---- Şekil 4: nominal senaryoda yörünge (manuel vs en iyi vs IFT-8)
nom = dict(fm=np.ones(1), fcd=np.ones(1), ft=np.ones(1), dv0=np.zeros(1), dh0=np.zeros(1), seed=7)
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
for ad, x in [("Manuel PID", MANUEL), (EN, en_iyi[EN])]:
    L = M.simule(x[None], nom, kayit=True)["log"]
    tt = L["t"]
    ax[0].plot(tt, L["v"][:, 0, 0], color=RENK[ad], label=ad)
    ax[1].plot(L["h"][:, 0, 0], L["v"][:, 0, 0], color=RENK[ad], label=ad)
    ax[2].plot(tt, L["u"][:, 0, 0] * 100, color=RENK[ad], lw=1, label=ad)
ax[0].plot(t[t >= 0], v[t >= 0], "o", ms=3, color=RENK["IFT-8"], label="IFT-8 verisi")
hh = np.linspace(0, REF.h0, 300)
ax[1].plot(hh, REF.v_ref(hh), "--", color=RENK["IFT-8"], label="Referans $v_{ref}(h)$")
ax[0].set(xlabel="Süre [s]", ylabel="Hız [m/s]", title="(a) Hız", xlim=(0, 30))
ax[1].set(xlabel="İrtifa [m]", ylabel="Hız [m/s]", title="(b) Hız–irtifa profili")
ax[1].invert_xaxis()
ax[2].set(xlabel="Süre [s]", ylabel="Gaz kolu [%]", title="(c) Kontrol sinyali", xlim=(0, 30))
for a in ax:
    a.legend(frameon=False, fontsize=8)
kaydet(fig, "sekil4_yorunge.png")

# ---- Şekil 5: Monte Carlo test (200 senaryo)
sc = M.senaryolar(N_TEST, TEST_SEED)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
for ad, x in [("Manuel PID", MANUEL), (EN, en_iyi[EN])]:
    r = M.simule(x[None], sc, kayit=True)
    L = r["log"]
    for k in range(0, N_TEST, 4):
        ax[0].plot(L["h"][:, 0, k], L["v"][:, 0, k], color=RENK[ad], alpha=0.12, lw=0.7)
    ax[0].plot([], [], color=RENK[ad], label=ad)
    ax[1].hist(r["v_td"][0], bins=np.linspace(0, 3.5, 36), color=RENK[ad], alpha=0.55, label=ad)
ax[0].plot(hh, REF.v_ref(hh), "--", color=RENK["IFT-8"], label="Referans")
ax[0].set(xlabel="İrtifa [m]", ylabel="Hız [m/s]", title=f"(a) {N_TEST} Monte Carlo senaryosu",
          xlim=(400, 0), ylim=(0, 80))
ax[1].axvline(C.V_TOUCHDOWN_TARGET, color="k", ls=":", lw=1, label="Hedef 1 m/s")
ax[1].set(xlabel="Temas hızı [m/s]", ylabel="Senaryo sayısı", title="(b) Temas hızı dağılımı")
for a in ax:
    a.legend(frameon=False, fontsize=8)
kaydet(fig, "sekil5_monte_carlo.png")
