"""
Evrimsel ve karınca kolonisi tabanlı optimizasyon algoritmaları (sıfırdan, yalnızca numpy).
Hepsi aynı arayüz:  algoritma(f, lb, ub, pop, iters, seed) -> (x_best, f_best, egri)
  f    : (P,D) -> (P,) toplu (vektörize) amaç fonksiyonu
Toplam değerlendirme bütçesi her yöntemde pop*iters -> adil karşılaştırma.
"""
import numpy as np


# ---------------------------------------------------------------- GA
def ga(f, lb, ub, pop=30, iters=60, seed=0, pc=0.9, pm=0.15, elit=2, tur=3, alfa=0.5):
    """Gerçel kodlu Genetik Algoritma (Holland, 1975; Goldberg, 1989).
    Turnuva seçimi (k=3) + BLX-α çaprazlama (Eshelman & Schaffer, 1993)
    + Gauss mutasyonu + elitizm (en iyi 2 birey korunur)."""
    r = np.random.default_rng(seed)
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    D, aralik = len(lb), ub - lb
    X = lb + r.random((pop, D)) * aralik
    F = f(X)
    egri = [F.min()]
    n_ev, butce = pop, pop * iters
    while n_ev < butce:
        sira = np.argsort(F)

        def secim():
            aday = r.integers(0, pop, tur)
            return X[aday[np.argmin(F[aday])]]

        n_c = min(pop - elit, butce - n_ev)
        cocuk = []
        while len(cocuk) < n_c:
            p1, p2 = secim(), secim()
            if r.random() < pc:
                lo, hi = np.minimum(p1, p2), np.maximum(p1, p2)
                c = r.uniform(lo - alfa * (hi - lo), hi + alfa * (hi - lo))
            else:
                c = p1.copy()
            mut = r.random(D) < pm
            c[mut] += r.normal(0, 0.1 * aralik[mut])
            cocuk.append(np.clip(c, lb, ub))
        Xc = np.array(cocuk)
        Fc = f(Xc)
        n_ev += n_c
        kalan = sira[elit:pop - n_c + elit] if n_c < pop - elit else sira[:0]
        X = np.vstack([X[sira[:elit]], X[kalan], Xc])
        F = np.concatenate([F[sira[:elit]], F[kalan], Fc])
        egri.append(F.min())
    i = np.argmin(F)
    return X[i], F[i], np.minimum.accumulate(egri)


# ---------------------------------------------------------------- ACO_R
def aco(f, lb, ub, pop=30, iters=60, seed=0, k=30, q=0.2, xi=0.85):
    """Sürekli uzay için Karınca Kolonisi Optimizasyonu, ACO_R (Socha & Dorigo, 2008).
    Feromon = k iyi çözümden oluşan arşiv. Her karınca:
      1) arşivden bir çözüm l seçer, olasılık ∝ w_l = exp(-(l-1)^2 / (2 q^2 k^2))   (sıra ağırlığı)
      2) her boyutta Gauss örnekler:  x_i ~ N( s_l,i , σ_i ),
         σ_i = ξ · Σ_e |s_e,i - s_l,i| / (k-1)       (arşivin o boyuttaki yayılımı)
    Yeni çözümler arşive eklenir, en iyi k tanesi kalır (feromon güncelleme / buharlaşma).
    q küçük -> en iyi çözüme yakın arama (sömürü); ξ büyük -> yavaş yakınsama (keşif)."""
    r = np.random.default_rng(seed)
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    D = len(lb)
    A = lb + r.random((k, D)) * (ub - lb)          # başlangıç arşivi
    FA = f(A)
    s = np.argsort(FA); A, FA = A[s], FA[s]
    rank = np.arange(k)
    w = np.exp(-rank ** 2 / (2 * (q * k) ** 2))
    p = w / w.sum()
    egri = [FA[0]]
    n_ev, butce = k, pop * iters
    while n_ev < butce:
        m = min(pop, butce - n_ev)                   # bu iterasyondaki karınca sayısı
        l = r.choice(k, size=m, p=p)
        sigma = xi * np.abs(A[None, :, :] - A[l][:, None, :]).sum(axis=1) / (k - 1)
        Xn = np.clip(r.normal(A[l], sigma + 1e-12), lb, ub)
        Fn = f(Xn)
        n_ev += m
        A = np.vstack([A, Xn]); FA = np.concatenate([FA, Fn])
        s = np.argsort(FA)[:k]
        A, FA = A[s], FA[s]
        egri.append(FA[0])
    return A[0], FA[0], np.array(egri)


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


ALGORITMALAR = {"GA": ga, "ACO": aco, "Rastgele Arama": rastgele}
KARSILASTIR = ("GA", "ACO")
