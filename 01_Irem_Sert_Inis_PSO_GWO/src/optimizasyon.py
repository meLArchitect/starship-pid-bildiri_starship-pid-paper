"""
Sürü zekâsı tabanlı optimizasyon algoritmaları (sıfırdan, yalnızca numpy).
Hepsi aynı arayüz:  algoritma(f, lb, ub, pop, iters, seed) -> (x_best, f_best, egri)
  f    : (P,D) -> (P,) toplu (vektörize) amaç fonksiyonu
  egri : her iterasyon sonundaki en iyi J değeri (yakınsama eğrisi)
Her iterasyonda tam olarak `pop` değerlendirme yapılır -> eşit bütçe (pop*iters).
"""
import numpy as np


# ---------------------------------------------------------------- PSO
def pso(f, lb, ub, pop=30, iters=60, seed=0, w_max=0.9, w_min=0.4, c1=2.0, c2=2.0, vmax_oran=0.2):
    """Parçacık Sürü Optimizasyonu (Kennedy & Eberhart, 1995; Shi & Eberhart, 1998).
    Atalet ağırlığı w iterasyonla 0.9 -> 0.4 doğrusal azalır.
        V <- w V + c1 r1 (pbest - X) + c2 r2 (gbest - X)
        X <- X + V
    """
    r = np.random.default_rng(seed)
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    D, aralik = len(lb), ub - lb
    vmax = vmax_oran * aralik
    X = lb + r.random((pop, D)) * aralik
    V = r.uniform(-vmax, vmax, (pop, D))
    F = f(X)
    pX, pF = X.copy(), F.copy()
    g = pX[np.argmin(pF)].copy()
    egri = [pF.min()]
    for it in range(1, iters):
        w = w_max - (w_max - w_min) * it / (iters - 1)
        r1, r2 = r.random((pop, D)), r.random((pop, D))
        V = np.clip(w * V + c1 * r1 * (pX - X) + c2 * r2 * (g - X), -vmax, vmax)
        X = np.clip(X + V, lb, ub)
        F = f(X)
        iyi = F < pF
        pX[iyi], pF[iyi] = X[iyi], F[iyi]
        g = pX[np.argmin(pF)].copy()
        egri.append(pF.min())
    return g, pF.min(), np.array(egri)


# ---------------------------------------------------------------- GWO
def gwo(f, lb, ub, pop=30, iters=60, seed=0):
    """Gri Kurt Optimizasyonu (Mirjalili, Mirjalili & Lewis, 2014).
    Sürünün en iyi üç kurdu alfa, beta, delta avı (optimumu) kuşatır; diğerleri onları izler.
        a  : 2 -> 0 doğrusal azalır (keşif -> sömürü)
        A  = 2 a r1 - a ,  C = 2 r2
        X1 = X_alfa  - A1 |C1 X_alfa  - X|   (X2, X3 beta ve delta için benzer)
        X  <- (X1 + X2 + X3) / 3
    |A| > 1 iken kurtlar avdan uzaklaşır (keşif), |A| < 1 iken avın üzerine gider (sömürü).
    """
    r = np.random.default_rng(seed)
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    D = len(lb)
    X = lb + r.random((pop, D)) * (ub - lb)
    F = f(X)
    s = np.argsort(F)
    lider = X[s[:3]].copy()          # alfa, beta, delta
    liderF = F[s[:3]].copy()
    egri = [liderF[0]]
    for it in range(1, iters):
        a = 2.0 - 2.0 * it / (iters - 1)
        Xyeni = np.zeros_like(X)
        for L in lider:
            A = 2 * a * r.random((pop, D)) - a
            Cc = 2 * r.random((pop, D))
            Xyeni += L - A * np.abs(Cc * L - X)
        X = np.clip(Xyeni / 3.0, lb, ub)
        F = f(X)
        # liderleri güncelle: eski liderler + yeni sürüden en iyi üç
        havuzX = np.vstack([lider, X])
        havuzF = np.concatenate([liderF, F])
        s = np.argsort(havuzF)[:3]
        lider, liderF = havuzX[s].copy(), havuzF[s].copy()
        egri.append(liderF[0])
    return lider[0], liderF[0], np.array(egri)


# ---------------------------------------------------------------- Rastgele arama (taban çizgisi)
def rastgele(f, lb, ub, pop=30, iters=60, seed=0):
    r = np.random.default_rng(seed)
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    gF, g, egri = np.inf, None, []
    for _ in range(iters):
        X = lb + r.random((pop, len(lb))) * (ub - lb)
        F = f(X)
        if F.min() < gF:
            gF, g = F.min(), X[np.argmin(F)].copy()
        egri.append(gF)
    return g, gF, np.array(egri)


ALGORITMALAR = {"PSO": pso, "GWO": gwo, "Rastgele Arama": rastgele}
KARSILASTIR = ("PSO", "GWO")    # istatistiksel testte karşılaştırılan iki ana yöntem
