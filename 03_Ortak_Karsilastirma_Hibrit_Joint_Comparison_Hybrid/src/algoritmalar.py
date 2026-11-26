"""
Ortak algoritma kütüphanesi / Shared algorithm library
------------------------------------------------------
İki çalışmadaki dört algoritma (GA, ACO_R -> Melih; PSO, GWO -> İrem) ile iki hibrit
yöntem (GA-ACO_R ve PSO-GWO) tek arayüzde toplanmıştır:
    algoritma(f, lb, ub, pop, iters, seed) -> (x_best, f_best, egri)
Her yöntem tam olarak pop*iters amaç fonksiyonu değerlendirmesi kullanır (eşit bütçe).
GA/ACO_R/PSO/GWO kodları yazarların kendi optimizasyon.py dosyalarından değiştirilmeden alınmıştır.
"""
import os, sys, importlib.util
import numpy as np

# sonfinal düzeni: 01_IS.../src, 02_MD.../src, 03_Ortak.../src
_KOK3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # 03_Ortak... klasörü
_SF = os.path.dirname(_KOK3)                                                 # sonfinal
_DIR = {"melih": "02_MD_Kalkis_GA_ACO_Ascent", "irem": "01_IS_Inis_PSO_GWO_Landing"}


def _yukle(ad, yol):
    spec = importlib.util.spec_from_file_location(ad, yol)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


_mel = _yukle("opt_melih", os.path.join(_SF, _DIR["melih"], "src", "optimizasyon.py"))
_irm = _yukle("opt_irem", os.path.join(_SF, _DIR["irem"], "src", "optimizasyon.py"))
ga, aco, rastgele = _mel.ga, _mel.aco, _mel.rastgele
pso, gwo = _irm.pso, _irm.gwo


# ---------------------------------------------------------------- Hibrit 1: GA -> ACO_R
def ga_aco(f, lb, ub, pop=30, iters=60, seed=0, oran=1/3, k=30, q=0.2, xi=0.85):
    """Ardışık (relay) hibrit: bütçenin ilk 1/3'ünde GA ile hızlı küresel arama,
    GA'nın son popülasyonu ACO_R arşivini başlatır, kalan 2/3 bütçede ACO_R ile ince ayar.
    Gerekçe: GA erken aşamada hızlı, ACO_R son aşamada hassas (Bölüm 4)."""
    r = np.random.default_rng(seed)
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    D, aralik = len(lb), ub - lb
    butce = pop * iters
    butce_ga = int(round(butce * oran / pop)) * pop
    # --- GA evresi (operatörler ga() ile aynı) ---
    X = lb + r.random((pop, D)) * aralik
    F = f(X); egri = [F.min()]; n_ev = pop
    while n_ev < butce_ga:
        sira = np.argsort(F)
        def secim():
            a = r.integers(0, pop, 3); return X[a[np.argmin(F[a])]]
        cocuk = []
        while len(cocuk) < pop - 2:
            p1, p2 = secim(), secim()
            if r.random() < 0.9:
                lo, hi = np.minimum(p1, p2), np.maximum(p1, p2)
                c = r.uniform(lo - 0.5 * (hi - lo), hi + 0.5 * (hi - lo))
            else:
                c = p1.copy()
            m = r.random(D) < 0.15
            c[m] += r.normal(0, 0.1 * aralik[m])
            cocuk.append(np.clip(c, lb, ub))
        Xc = np.array(cocuk); Fc = f(Xc); n_ev += len(Xc)
        X = np.vstack([X[sira[:2]], Xc]); F = np.concatenate([F[sira[:2]], Fc])
        egri.append(F.min())
    # --- ACO_R evresi: arşiv = GA popülasyonu ---
    s = np.argsort(F)[:k]; A, FA = X[s], F[s]
    w = np.exp(-np.arange(k) ** 2 / (2 * (q * k) ** 2)); p = w / w.sum()
    while n_ev < butce:
        m = min(pop, butce - n_ev)
        l = r.choice(k, size=m, p=p)
        sigma = xi * np.abs(A[None] - A[l][:, None]).sum(1) / (k - 1)
        Xn = np.clip(r.normal(A[l], sigma + 1e-12), lb, ub); Fn = f(Xn); n_ev += m
        A = np.vstack([A, Xn]); FA = np.concatenate([FA, Fn])
        s = np.argsort(FA)[:k]; A, FA = A[s], FA[s]
        egri.append(FA[0])
    return A[0], FA[0], np.minimum.accumulate(egri)


# ---------------------------------------------------------------- Hibrit 2: PSO-GWO
def pso_gwo(f, lb, ub, pop=30, iters=60, seed=0, w_max=0.9, w_min=0.4, c1=0.5, c2=1.5, vmax_oran=0.2):
    """Bütünleşik hibrit (Singh & Singh, 2017 yaklaşımından esinlenmiştir):
    parçacıklar PSO'daki gibi hız ve kişisel bellek tutar, ancak sosyal çekim noktası
    tek bir g_best yerine GWO'nun alfa-beta-delta liderlerinden üretilen konumdur:
        X_gwo = (X1 + X2 + X3)/3   (GWO denklemleri, a: 2 -> 0)
        V <- w V + c1 r1 (p_best - X) + c2 r2 (X_gwo - X)
    Böylece PSO'nun belleği ile GWO'nun çok liderli yönlendirmesi birleşir."""
    r = np.random.default_rng(seed)
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    D, aralik = len(lb), ub - lb
    vmax = vmax_oran * aralik
    X = lb + r.random((pop, D)) * aralik
    V = r.uniform(-vmax, vmax, (pop, D))
    F = f(X); pX, pF = X.copy(), F.copy()
    egri = [pF.min()]
    for it in range(1, iters):
        frac = it / (iters - 1)
        w = w_max - (w_max - w_min) * frac
        a = 2.0 - 2.0 * frac
        s = np.argsort(pF)[:3]
        Xg = np.zeros_like(X)
        for L in pX[s]:
            A = 2 * a * r.random((pop, D)) - a
            Cc = 2 * r.random((pop, D))
            Xg += L - A * np.abs(Cc * L - X)
        Xg /= 3.0
        r1, r2 = r.random((pop, D)), r.random((pop, D))
        V = np.clip(w * V + c1 * r1 * (pX - X) + c2 * r2 * (Xg - X), -vmax, vmax)
        X = np.clip(X + V, lb, ub)
        F = f(X)
        iyi = F < pF; pX[iyi], pF[iyi] = X[iyi], F[iyi]
        egri.append(pF.min())
    i = np.argmin(pF)
    return pX[i], pF[i], np.array(egri)


ALGORITMALAR = {"GA": ga, "ACO_R": aco, "PSO": pso, "GWO": gwo,
                "GA-ACO_R": ga_aco, "PSO-GWO": pso_gwo, "Rastgele Arama": rastgele}
