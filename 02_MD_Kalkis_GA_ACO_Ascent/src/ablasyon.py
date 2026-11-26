"""
Ablasyon çalışması: kazanç programlamanın (iki faz) katkısı.
Tek faz: tüm uçuş boyunca aynı [Kp, Ki, Kd]  (3 parametre)
İki faz: t < 75 s -> G1, t >= 75 s -> G2      (6 parametre, ana çalışma)
Çalıştırma: python ablasyon.py  ->  ../sonuclar_results/ablasyon.csv
"""
import os
import numpy as np
import pandas as pd
import config as C
import model as M
import optimizasyon as O
from deneyler import test_metrikleri

IKI_FAZ_ACO = np.array([0.23024604452831987, 0.08224002597600648, 0.0,
                        0.15463379643395, 0.08246728218757685, 0.0])   # test_sonuclari.csv

if __name__ == "__main__":
    f = M.AmacFonksiyonu(C.N_SCEN, seed=0)
    f3 = lambda X: f(np.hstack([np.atleast_2d(X)] * 2))      # aynı 3 kazanç iki fazda da
    lb, ub = np.array(C.LOWER[:3]), np.array(C.UPPER[:3])
    sonuc = [O.aco(f3, lb, ub, pop=C.POP, iters=C.ITER, seed=s) for s in range(5)]
    x3 = min(sonuc, key=lambda r: r[1])[0]
    tek = np.hstack([x3, x3])
    satirlar = []
    for ad, g, n in (("Tek faz", tek, 3), ("İki faz", IKI_FAZ_ACO, 6)):
        r = test_metrikleri(g)
        satirlar.append(dict(yapi=ad, parametre=n, Kp1=g[0], Ki1=g[1], Kd1=g[2], Kp2=g[3], Ki2=g[4], Kd2=g[5],
                             J_egitim=float(f(g[None])[0]), **r))
    df = pd.DataFrame(satirlar)
    os.makedirs(C.SONUC_KLASORU, exist_ok=True)
    df.to_csv(os.path.join(C.SONUC_KLASORU, "ablasyon.csv"), index=False)
    print(df[["yapi", "parametre", "Kp1", "Kp2", "J_egitim", "J_test", "IAE_ort", "TV_ort", "maxQ_ihlal"]].round(4))
