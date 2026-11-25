"""
Ortak deney: altı yöntem + rastgele arama, iki problemde (kalkış / iniş).
Kullanım:  python ortak_deney.py kalkis     |    python ortak_deney.py inis
Yazarların kendi model/config/deneyler dosyaları değiştirilmeden kullanılır
(aynı eğitim senaryoları, aynı tohumlar, aynı bütçe, aynı test senaryoları).
"""
import os, sys, json, time
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, friedmanchisquare

PROB = sys.argv[1]
KOK3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KOK = os.path.dirname(KOK3)
SRC = os.path.join(KOK, "02_Melih_Demir_Kalkis_GA_ACO" if PROB == "kalkis" else "01_Irem_Sert_Inis_PSO_GWO", "src")
sys.path.insert(0, SRC); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C, model as M, deneyler as Dn
import algoritmalar as A

OUT = os.path.join(KOK3, "sonuclar", PROB)
os.makedirs(OUT, exist_ok=True)
izgara = np.arange(C.POP, C.POP * C.ITER + 1, C.POP)
satir, egriler = [], {}
for ad, alg in A.ALGORITMALAR.items():
    egriler[ad] = []
    for s in range(1, C.N_RUNS + 1):
        f = Dn.KayitliAmac(); t0 = time.time()
        x, J, _ = alg(f, C.LOWER, C.UPPER, C.POP, C.ITER, seed=s)
        assert f.n_eval == C.POP * C.ITER, (ad, f.n_eval)
        ev, best = np.array(f.gecmis).T
        egriler[ad].append(np.interp(izgara, ev, best).tolist())
        satir.append(dict(yontem=ad, kosu=s, J_egitim=float(J), sure_s=time.time() - t0,
                          **dict(zip(C.PARAM_ADLARI, map(float, x)))))
        print(PROB, ad, s, round(float(J), 6), flush=True)
df = pd.DataFrame(satir); df.to_csv(os.path.join(OUT, "tum_kosular.csv"), index=False)

Jstar = df.J_egitim.min()
oz = df.groupby("yontem", sort=False).J_egitim.agg(["min", "mean", "std", "max"])
for esik in (300, 900):
    k = list(izgara).index(esik)
    oz[f"J_ort_{esik}"] = [np.mean([e[k] for e in egriler[a]]) for a in oz.index]
n1, bas = [], []
for a in oz.index:
    n = [izgara[np.argmax(np.array(e) <= Jstar * 1.01)] for e in egriler[a] if min(e) <= Jstar * 1.01]
    n1.append(np.mean(n) if n else np.nan); bas.append(f"{len(n)}/{C.N_RUNS}")
oz["eval_%1_ort"], oz["basari_%1"] = n1, bas
oz["sure_ort_s"] = df.groupby("yontem", sort=False).sure_s.mean()
# Mann-Whitney: her yöntem, ortalaması en iyi olan yönteme karşı
eniyi = oz.drop("Rastgele Arama")["mean"].idxmin()
ref = df[df.yontem == eniyi].J_egitim
oz["p_eniyi_ile"] = [np.nan if a == eniyi else mannwhitneyu(ref, df[df.yontem == a].J_egitim).pvalue for a in oz.index]
# ortalama sıra (Friedman) - koşu bazında, 6 yöntem
pv = df[df.yontem != "Rastgele Arama"].pivot(index="kosu", columns="yontem", values="J_egitim")
fr = friedmanchisquare(*[pv[c] for c in pv.columns])
oz["ort_sira"] = pv.rank(axis=1).mean().reindex(oz.index)
oz.to_csv(os.path.join(OUT, "ozet_egitim.csv"))
print(oz.round(5).to_string()); print("Friedman", fr)

test = [dict(yontem="Manuel PID", **dict(zip(C.PARAM_ADLARI, Dn.MANUEL)), **Dn.test_metrikleri(Dn.MANUEL))]
for ad in A.ALGORITMALAR:
    x = df[df.yontem == ad].sort_values("J_egitim").iloc[0][C.PARAM_ADLARI].values.astype(float)
    test.append(dict(yontem=ad, **dict(zip(C.PARAM_ADLARI, x)), **Dn.test_metrikleri(x)))
pd.DataFrame(test).to_csv(os.path.join(OUT, "test_sonuclari.csv"), index=False)
json.dump(dict(izgara=izgara.tolist(), egriler=egriler, eniyi=eniyi,
               friedman=[float(fr.statistic), float(fr.pvalue)]),
          open(os.path.join(OUT, "yakinsama.json"), "w"))
print(pd.DataFrame(test).round(4).to_string())
