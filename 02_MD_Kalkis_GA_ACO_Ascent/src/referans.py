"""
IFT-6, IFT-7, IFT-8 Booster kalkış verisinden referans profiller.
Her uçuş için: v_ref(t), a_ref(t) = dv_ref/dt, sin(gamma)(t) (uçuş yolu açısı), h_ref(t), max-Q.

Adımlar
  1) Tekrarlanan zaman damgaları atılır, 1 s ızgaraya interpolasyon (OCR kaynaklı atlamalar).
  2) Hız km/h -> m/s; 7. derece polinomla yumuşatılır (OCR gürültüsü).
  3) İrtifa OCR'da tam sayı km (aşağı yuvarlanmış) -> +500 m kaydırılıp 6. derece polinom.
  4) sin(gamma) = (dh/dt) / v ; düşük hızda (v < 60 m/s) dikey tırmanış: sin(gamma) = 1.
"""
import os
import numpy as np
import pandas as pd
import config as C


def _saniye(s):
    h, m, x = map(int, s.split(":"))     #CSVDEnki saat:dakika:saniye yazısını tek bir sayısya yani toplam saniyeye çeviriyor.
    return h * 3600 + m * 60 + x


def ham_veri(ucus):
    d = pd.read_csv(os.path.join(C.VERI_KLASORU, f"{ucus}_results.csv"), encoding="utf-8-sig")
    d = d.drop_duplicates(subset="time", keep="first")# aynı tekrar siliniyor.
    d["ts"] = d["time"].map(_saniye)# saniye sütunu ekliyoruz.
    d = d[d.ts <= C.T_END + 10].groupby("ts").first().reset_index() #üç iş birden yapılıyor, kalkış kısmını alıyor,satırları saniyeye göre sıralıyor, satır numaralarını baştan düzenliyor.
    return d
#Yukardaki fonsiyon CVS dokümanını açıyor, temizliyor ve kalkış kısmını ayırıp geri veriyor.

#Aşağıdaki yöntem bir kalıp, kalıba IFT6 veriyorum PID nin  takip edeceği hedef yolu çıkarıyor.
class Referans:
    def __init__(self, ucus):
        self.ad = ucus.upper().replace("IFT", "IFT-") # ad ve ham veri
        d = ham_veri(ucus)

        t_ham = d.ts.values.astype(float) #her saniye bir değer, birim çevirme
        self.t = np.arange(0, C.T_END + 1.0)
        self.v_veri = np.interp(self.t, t_ham, d.booster_speed.values / 3.6) #interpolasyon ve  metreye çevirme
        self.h_veri = np.interp(self.t, t_ham, d.booster_altitude.values * 1000.0)

        self.pv = np.polyfit(self.t, self.v_veri, 7) # Hızı, 7. derece polinom yumusatma
        ph = np.polyfit(self.t, self.h_veri + 500.0 * (self.h_veri > 0), 6) # İrtifayı, 6. derece polinom yumusatma

        v = np.polyval(self.pv, self.t) # Uçuş yolu açısı gamma hesabı
        hd = np.polyval(np.polyder(ph), self.t)
        sg = np.clip(hd / np.maximum(v, 1.0), 0.05, 1.0)
        sg[v < 60] = 1.0

        self._sg = sg #kontrol, hesaplanan irtifa ekranla tutuyor mu?
        # referans irtifa: v*sin(gamma) integrali (OCR ile tutarlılık kontrolü için)
        self.h_ref = np.concatenate([[0], np.cumsum(0.5 * (v[1:] * sg[1:] + v[:-1] * sg[:-1]))])

        rho = C.RHO0 * np.exp(-self.h_ref / C.H_SCALE) # Max-Q, her saniye için q hesaplanıyor en büyüğü ve zamanı saklanıyor.
        q = 0.5 * rho * v ** 2
        self.q_max, self.t_qmax = q.max(), self.t[q.argmax()]

#model.py'ın sorduğu 3 soru
    def v_ref(self, t):
        return np.polyval(self.pv, t)             #hedef hız, hızım ne olmalı?

    def a_ref(self, t):
        return np.polyval(np.polyder(self.pv), t)  #hedef ivme, ne kadar hızlanmalıyım?

    def sin_gamma(self, t):
        return np.interp(t, self.t, self._sg)      #eğim, ne kadar yatığım?



REFS = [Referans(u) for u in C.UCUSLAR] # aynı kalıbı üç kez yapıyor ıft6,ıft7,ıft8 için

if __name__ == "__main__": #özet yazdırılır
    for R in REFS:
        print(f"{R.ad}: v(150)={R.v_ref(150):.0f} m/s, h_ref(150)={R.h_ref[-1]/1e3:.1f} km "
              f"(OCR {R.h_veri[-1]/1e3:.0f} km), max-Q={R.q_max/1e3:.1f} kPa @ t={R.t_qmax:.0f} s, "
              f"hız RMSE={np.sqrt(np.mean((R.v_ref(R.t)-R.v_veri)**2)):.2f} m/s")

#r.ad, IFT8 Hangi uçuş?
#R.v_ref(150), v(150)=1213m/s Yolun sonundaki hedef hız mantıklı mı? 150. saniyedeki yumuşatılmış eğreiden elde ettiğimiz değer.

#R.h_ref[-1], h_ref(150)=54.7 km gamma doğru mu? Bizim hesapladığımız dikey hızı topladık
#R.h_veri[-1], (OCR 54 km) Ekrandaki değerle karşılaştırma csv deki ham veriden elde edilen değer
#yukardaki iki hesap var karşılaştırıyoruz yakınsa gamma doğrudur.

#R.q_max, R.t_qmax; max-Q=21.8 kPa @ t=67 s Max-Q gerçekçi mi?
#print satırın içinde hesaplanıyor, hız RMSE=2.99 m/s Polinom veriyi iyi takip ediyor mu? İp 150 noktaya bağladığımızda her saniyede ne kadar ip ana eğriye uzak bunların ortalamasını alıyoruz. İşlem yapılırken rms gibi kare ortalamalarının kare kökü hesaplanır çünkü negatif ederler korunmalı.
#İyi bir yumuşatmada RMSE=verinin gürültü seviyesi olmalı. RMSE gürültüden büyükse kötü; küçükse eğri gürültüyü kopyalar kötü; gürültü seviyesine eşitse iyi
#burada gürültüyü hesaplamaya gerek yok hız ile karşılaştırdık:
#ift6 için 2.31/1394 *100 yaptığımızda %0,17=%o 1.7, yani veriden binde 2-3 sapıyor bu mühendislikte çok iyi uyum sayılır.
