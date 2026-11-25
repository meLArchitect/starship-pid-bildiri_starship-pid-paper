"""
Deney düzeni / Experimental setup
  * 3 yöntem: PSO, Gri Kurt Optimizasyonu (GWO), Rastgele Arama (taban çizgisi)
  * Her yöntem N_RUNS bağımsız koşu (tohum 1..N_RUNS), eşit bütçe (POP*ITER değerlendirme)
  * Eğitim: sabit 8 Monte Carlo senaryosu (ortak rastgele sayılar -> adil karşılaştırma)
  * Test : eğitimde görülmemiş 200 senaryo (aşırı uyum kontrolü)
  * Karşılaştırma için elle ayarlanmış PID (mühendislik sezgisi) de test edilir

Çalıştırma:  python deneyler.py         (~3-4 dk)
Çıktılar  :  ../sonuclar/*.csv, *.json
"""
import json
import os
import time
import numpy as np
import pandas as pd
import config as C
import model as M
import optimizasyon as O

MANUEL = np.array([1.0, 0.1, 0.1, 1.0, 0.1, 0.1])    # elle ayarlanmış başlangıç PID'i
N_TEST, TEST_SEED = 200, 12345


class KayitliAmac(M.AmacFonksiyonu):
    """Her çağrıda (değerlendirme sayısı, şimdiye kadarki en iyi J) kaydeder."""

    def __init__(self):
        super().__init__(C.N_SCEN, seed=0)
        self.gecmis, self.en_iyi = [], np.inf

    def __call__(self, X):
        J = super().__call__(X)
        self.en_iyi = min(self.en_iyi, J.min())
        self.gecmis.append((self.n_eval, self.en_iyi))
        return J


def test_metrikleri(x):
    sc = M.senaryolar(N_TEST, TEST_SEED)
    r = M.simule(np.asarray(x)[None], sc)
    v = r["v_td"][0]
    return dict(
        J_test=float(M.maliyet(r)[0]),
        IAE_ort=float(r["iae"][0].mean()),
        v_temas_ort=float(np.nanmean(v)),
        v_temas_maks=float(np.nanmax(v)),
        guvenli_inis_orani=float(np.mean(np.nan_to_num(v, nan=99) <= 3.0)),   # ≤3 m/s
        yakit_ort_ton=float(r["yakit"][0].mean() / 1e3),
        sure_ort=float(np.nanmean(r["t_td"][0])),
        inemedi=int(r["inmedi"][0].sum()),
    )


def main():
    os.makedirs(C.SONUC_KLASORU, exist_ok=True)
    izgara = np.arange(C.POP, C.POP * C.ITER + 1, C.POP)
    satirlar, egriler = [], {}
    for ad, alg in O.ALGORITMALAR.items():
        egriler[ad] = []
        for s in range(1, C.N_RUNS + 1):
            f = KayitliAmac()
            t0 = time.time()
            x, J, _ = alg(f, C.LOWER, C.UPPER, C.POP, C.ITER, seed=s)
            sure = time.time() - t0
            ev, best = np.array(f.gecmis).T
            egriler[ad].append(np.interp(izgara, ev, best).tolist())
            satirlar.append(dict(yontem=ad, kosu=s, J_egitim=float(J), sure_s=sure,
                                 n_eval=f.n_eval, **dict(zip(C.PARAM_ADLARI, map(float, x)))))
            print(f"{ad:15s} koşu {s:2d}  J={J:.4f}  {sure:.1f}s  x={np.round(x, 3)}")

    df = pd.DataFrame(satirlar)
    df.to_csv(os.path.join(C.SONUC_KLASORU, "tum_kosular.csv"), index=False)

    # --- özet istatistikler (eğitim J) ---
    ozet = df.groupby("yontem", sort=False)["J_egitim"].agg(["min", "mean", "std", "max"])
    ozet["sure_ort_s"] = df.groupby("yontem", sort=False)["sure_s"].mean()
    # %1 tolerans içinde en iyiye ulaşma için gereken değerlendirme (ortalama)
    J_star = df["J_egitim"].min()
    gerekli = {}
    for ad, E in egriler.items():
        n = [izgara[np.argmax(np.array(e) <= J_star * 1.01)] if min(e) <= J_star * 1.01 else np.nan for e in E]
        n = np.array(n, float)
        gerekli[ad] = (np.nanmean(n) if np.any(~np.isnan(n)) else np.nan, int(np.sum(~np.isnan(n))))
    ozet["eval_%1_ort"] = [gerekli[a][0] for a in ozet.index]
    ozet["basari_%1"] = [f"{gerekli[a][1]}/{C.N_RUNS}" for a in ozet.index]

    # --- istatistiksel test (Mann-Whitney U, iki ana yöntem ve taban çizgisi) ---
    try:
        from scipy.stats import mannwhitneyu
        a1, a2 = O.KARSILASTIR
        ref = df[df.yontem == a2]["J_egitim"]
        ozet[f"p_{a2}_ile"] = [np.nan if a == a2 else
                              mannwhitneyu(ref, df[df.yontem == a]["J_egitim"]).pvalue for a in ozet.index]
    except ImportError:
        print("scipy yok -> istatistiksel test atlandı (pip install scipy)")
    ozet.to_csv(os.path.join(C.SONUC_KLASORU, "ozet_egitim.csv"))
    print("\n", ozet.round(5))

    # --- test senaryolarında doğrulama: her yöntemin en iyi koşusu + manuel ---
    test = [dict(yontem="Manuel PID", **dict(zip(C.PARAM_ADLARI, MANUEL)), **test_metrikleri(MANUEL))]
    for ad in O.ALGORITMALAR:
        en = df[df.yontem == ad].sort_values("J_egitim").iloc[0]
        x = en[C.PARAM_ADLARI].values.astype(float)
        test.append(dict(yontem=ad, **dict(zip(C.PARAM_ADLARI, x)), **test_metrikleri(x)))
    dt = pd.DataFrame(test)
    dt.to_csv(os.path.join(C.SONUC_KLASORU, "test_sonuclari.csv"), index=False)
    print("\n", dt.round(3).to_string(index=False))

    with open(os.path.join(C.SONUC_KLASORU, "yakinsama.json"), "w") as fp:
        json.dump(dict(izgara=izgara.tolist(), egriler=egriler), fp)


if __name__ == "__main__":
    main()
