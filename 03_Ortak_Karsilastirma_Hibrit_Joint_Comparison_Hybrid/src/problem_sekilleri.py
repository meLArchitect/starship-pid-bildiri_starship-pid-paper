"""Probleme özgü şekiller (TR/EN). Kullanım: python problem_sekilleri.py kalkis|inis tr|en"""
import os, sys, json
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
PROB, DIL = sys.argv[1], sys.argv[2]
KOK3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KOK = os.path.dirname(KOK3)
sys.path.insert(0, os.path.join(KOK, "02_MD_Kalkis_GA_ACO_Ascent" if PROB == "kalkis" else "01_IS_Inis_PSO_GWO_Landing", "src"))
import config as C, model as M
from deneyler import MANUEL, N_TEST, TEST_SEED
T = (lambda tr, en: tr) if DIL == "tr" else (lambda tr, en: en)
OUT = os.path.join(os.path.dirname(KOK3), "04_Sekiller_Figures", DIL); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 10, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300, "savefig.bbox": "tight", "mathtext.fontset": "stix"})
RES = os.path.join(KOK3, "sonuclar_results", {"kalkis": "kalkis_ascent", "inis": "inis_landing"}[PROB])
df = pd.read_csv(os.path.join(RES, "tum_kosular.csv"))
oz = pd.read_csv(os.path.join(RES, "ozet_egitim.csv"), index_col=0)
EN = oz.drop("Rastgele Arama")["mean"].idxmin()
xbest = df[df.yontem == EN].sort_values("J_egitim").iloc[0][C.PARAM_ADLARI].values.astype(float)
MAN, OPT = T("Elle ayarlı PID", "Hand-tuned PID"), T(f"Optimize PID ({EN})", f"Optimized PID ({EN})")
CM, CO = "#c0392b", "#1f4e8c"
def kaydet(fig, ad):
    fig.savefig(os.path.join(OUT, ad)); plt.close(fig); print(ad)
S = T("Süre [s]", "Time [s]")

if PROB == "kalkis":
    from referans import REFS
    UR = ["#7b3fa0", "#e07b00", "#222222"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.1))
    for R, c in zip(REFS, UR):
        ax[0].plot(R.t, R.v_veri, ".", ms=2, color=c, alpha=0.5)
        ax[0].plot(R.t, R.v_ref(R.t), color=c, lw=1.4, label=R.ad)
        rho = C.RHO0 * np.exp(-R.h_ref / C.H_SCALE)
        ax[1].plot(R.t, 0.5 * rho * R.v_ref(R.t) ** 2 / 1e3, color=c, lw=1.4, label=R.ad)
    ax[0].set(xlabel=S, ylabel=T("Hız [m/s]", "Velocity [m/s]"),
              title=T("(a) Kalkış hızı: OCR (nokta), referans (çizgi)", "(a) Ascent velocity: OCR (dots), reference (line)"))
    ax[1].set(xlabel=S, ylabel=T("Dinamik basınç q [kPa]", "Dynamic pressure q [kPa]"), title=T("(b) Dinamik basınç (max-Q)", "(b) Dynamic pressure (max-Q)"))
    for a in ax: a.axvline(C.T_SWITCH, color="k", ls=":", lw=0.8); a.legend(frameon=False, fontsize=8)
    kaydet(fig, "veri_kalkis.png")

    sen = dict(ucus=np.array([2]), fm=np.ones(1), fcd=np.ones(1), ft=np.ones(1), t_ariza=np.array([60.0]), seed=7)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
    for ad, x, c in [(MAN, MANUEL, CM), (OPT, xbest, CO)]:
        L = M.simule(x[None], sen, kayit=True)["log"]; tt = L["t"]
        ax[0].plot(tt, (L["v"] - L["vref"])[:, 0, 0], color=c, lw=1, label=ad)
        ax[1].plot(tt, L["u"][:, 0, 0] * 100, color=c, lw=1, label=ad)
        ax[2].plot(tt, L["q"][:, 0, 0] / 1e3, color=c, lw=1.2, label=ad)
    ax[2].axhline(C.Q_MARJ * REFS[2].q_max / 1e3, color="k", ls="--", lw=1, label=T("max-Q sınırı", "max-Q limit"))
    for a in ax: a.axvline(60, color="#888", ls=":", lw=1)
    ax[0].set(xlabel=S, ylabel=r"$v - v_{ref}$ [m/s]", title=T("(a) Hız takip hatası", "(a) Velocity tracking error"))
    ax[1].set(xlabel=S, ylabel=T("Gaz kolu [%]", "Throttle [%]"), title=T("(b) Gaz kolu", "(b) Throttle command"))
    ax[2].set(xlabel=S, ylabel="q [kPa]", title=T("(c) Dinamik basınç", "(c) Dynamic pressure"))
    for a in ax: a.legend(frameon=False, fontsize=7.5)
    kaydet(fig, "kalkis_motor_arizasi.png")
else:
    from referans import veri_yukle; REF = M.REF
    t, v, h, d = veri_yukle()
    ocr_h = d["booster_altitude"].astype(float).values * 1000
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.1))
    ax[0].plot(t, v, "o-", ms=2.5, color="#222")
    ax[0].axvspan(0, C.T_SWITCH, color="#f5b041", alpha=0.25, label=T("13 motor", "13 engines"))
    ax[0].axvspan(C.T_SWITCH, t[-1], color="#85c1e9", alpha=0.25, label=T("3 motor", "3 engines"))
    ax[0].set(xlabel=T("Ateşlemeden itibaren süre [s]", "Time from ignition [s]"), ylabel=T("Hız [m/s]", "Velocity [m/s]"),
              title=T("(c) IFT-8 iniş yanması hızı", "(c) IFT-8 landing-burn velocity"))
    ax[1].plot(t, h, "o-", ms=2.5, color="#222", label=T("Hızdan integre edilen", "Integrated from velocity"))
    ax[1].step(t, ocr_h, where="post", color=CM, alpha=0.7, label=T("OCR (tam sayı km)", "OCR (integer km)"))
    ax[1].set(xlabel=T("Ateşlemeden itibaren süre [s]", "Time from ignition [s]"), ylabel=T("İrtifa [m]", "Altitude [m]"),
              title=T("(d) İrtifanın geri hesaplanması", "(d) Altitude reconstruction"))
    for a in ax: a.legend(frameon=False, fontsize=8)
    kaydet(fig, "veri_inis.png")

    sc = M.senaryolar(N_TEST, TEST_SEED); hh = np.linspace(0, REF.h0, 300)
    nom = dict(fm=np.ones(1), fcd=np.ones(1), ft=np.ones(1), dv0=np.zeros(1), dh0=np.zeros(1), seed=7)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
    for ad, x, c in [(MAN, MANUEL, CM), (OPT, xbest, CO)]:
        r = M.simule(x[None], sc, kayit=True); L = r["log"]
        for k in range(0, N_TEST, 4):
            ax[0].plot(L["h"][:, 0, k], L["v"][:, 0, k], color=c, alpha=0.12, lw=0.7)
        ax[0].plot([], [], color=c, label=ad)
        ax[1].hist(r["v_td"][0], bins=np.linspace(0, 3.5, 36), color=c, alpha=0.55, label=ad)
        Ln = M.simule(x[None], nom, kayit=True)["log"]
        ax[2].plot(Ln["t"], Ln["u"][:, 0, 0] * 100, color=c, lw=1, label=ad)
    ax[0].plot(hh, REF.v_ref(hh), "--", color="k", label=T("Referans", "Reference"))
    ax[0].set(xlabel=T("İrtifa [m]", "Altitude [m]"), ylabel=T("Hız [m/s]", "Velocity [m/s]"),
              title=T(f"(a) {N_TEST} test senaryosu", f"(a) {N_TEST} test scenarios"), xlim=(400, 0), ylim=(0, 80))
    ax[1].axvline(C.V_TOUCHDOWN_TARGET, color="k", ls=":", lw=1, label=T("Hedef 1 m/s", "Target 1 m/s"))
    ax[1].set(xlabel=T("Temas hızı [m/s]", "Touchdown velocity [m/s]"), ylabel=T("Senaryo sayısı", "Scenarios"),
              title=T("(b) Temas hızı dağılımı", "(b) Touchdown velocity"))
    ax[2].set(xlabel=S, ylabel=T("Gaz kolu [%]", "Throttle [%]"), title=T("(c) Nominal senaryoda gaz kolu", "(c) Throttle, nominal case"), xlim=(0, 30))
    for a in ax: a.legend(frameon=False, fontsize=7.5)
    kaydet(fig, "inis_monte_carlo.png")
