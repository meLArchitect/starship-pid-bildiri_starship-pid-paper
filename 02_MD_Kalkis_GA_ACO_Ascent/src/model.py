"""
Super Heavy + Starship kalkış fazı: uçuş yolu boyunca 1-B dinamik model + PID gaz kolu kontrolü.
Vektörize: P aday PID x K senaryo aynı anda simüle edilir.

Durum (uçuş yolu boyunca, yukarı/ileri pozitif):
    dv/dt = (T - D) / m - g * sin(gamma(t))
    dh/dt = v * sin(gamma(t))                 gamma(t): uçuşun eğim programı (veriden)
    dm/dt = -T / (Isp(h) * g0)
    dT/dt = (T_cmd - T) / tau
    D     = 0.5 * rho(h) * Cd * A * v^2
    Isp(h)= Isp_vac - (Isp_vac - Isp_sl) * p(h)/p0      (irtifayla artan itki)

Kontrol (ileri beslemeli, kazanç programlamalı PID, hız takibi):
    e     = v_ref(t) - v_ölçülen                 (pozitif: gereğinden yavaş)
    a_cmd = a_ref(t) + g sin(gamma) + D_tahmin/m_tahmin + Kp e + Ki ∫e + Kd de/dt
    u     = sat( m_tahmin * a_cmd / (N * T_motor(h)), 0.40, 1.0 )
    faz-1 (t < 75 s, max-Q bölgesi) ve faz-2 (t >= 75 s) için ayrı [Kp, Ki, Kd]
"""
import numpy as np
import config as C
from referans import REFS

N_ADIM = int(round(C.T_END / C.DT)) + 1                # kaç an var, sayıyor 150/0,1=1500 ve 0.anda var 1501. 0,1 döngü delta t
_tg = np.arange(N_ADIM) * C.DT                         # 1501 anın saat listesini yazıyor 0;0,1;0,2;0,3...149,9;150
VREF = np.array([R.v_ref(_tg) for R in REFS])          # referansa gidiyor ve her uçusa soruyor bu 1501 anın her birinde hızın ne olmalı. cevaplar bir tabloya array yazılıyor.
AREF = np.array([R.a_ref(_tg) for R in REFS])          # referansa gidiyor ve her uçusa soruyor bu 1501 anın her birinde ivmesi ne olmalı. cevaplar bir tabloya array yazılıyor.
SG = np.array([R.sin_gamma(_tg) for R in REFS])        # referansa gidiyor ve her uçusa soruyor bu 1501 anın her birinde ne kadar yatıksın. cevaplar bir tabloya array yazılıyor.
QMAX_REF = np.array([R.q_max for R in REFS])           # referansa gidiyor ve her uçusa soruyor bu 1501 anın her birinde en yüksek basıncın neydi. cevaplar bir tabloya array yazılıyor.


# 3 tane uçuş verimiz var. Burda her yolu 3 kez 3 farklı koşulda gitmesini istiyoruz. Böylece 9 kez test ediyoruz.
# Eğer tek test edilse 1 sürüş yapılsa optimizasyon ezber yapabilirdi yeterli olmaz.
# ıft6 test 1,4,7
# ıft7 test 2,5,8
# ıft8 test 3,6,9 maruz kalıyor.
#test ettğimiz roketin pıd değerleri (2 faz olduğu için 6 tane kp1,kp2...)
#test senaryolarında hava direnci var biz %30 artı eksi değiştirdik
#motor gücü var biz biz%3 artı eksi değiştirdik
#motor arızası kullandığımız veride yok, tamamen biz ekledik
#sensör hatası gerçek sensörlerde var. biz %1 artı eksi ekledik
#biz ya tahminimiz biraz yanlışsa diye değerlerle oynadık.


#k örneğin kaç soruluk bir sınav olsun 20, seed kitapçık numarası ne olsun a,b,c
#pıd ayarları örenciler
#soru kağıdı 9 test
#seed 0 herkese aynı kağıdı ver demek. seed değişirse herkes aynı sınava girer ama sorular yani durumlar değişir.


def senaryolar(K, seed):
    """Belirsizlik senaryoları. Her senaryo bir referans uçuşa (IFT-6/7/8) atanır."""
    r = np.random.default_rng(seed)
    ariza = r.random(K) < C.P_MOTOR_ARIZA # bir senaryoda tek motor arızası olasılığı (33 -> 32) 9 testen tam 3 ü arıza oluyor lisre tutuyor evet hayır diye
    return dict(
        ucus=np.arange(K) % len(REFS),
        fm=r.uniform(0.97, 1.03, K),             # kalkış kütlesi ±%3
        fcd=r.uniform(0.70, 1.30, K),            # sürükleme katsayısı ±%30
        ft=r.uniform(0.97, 1.03, K),             # motor itki verimi ±%3
        t_ariza=np.where(ariza, r.uniform(20, 140, K), np.inf),   # tek motor arızası hangi saniyede olacağı
        seed=int(seed),
    )


def _isp(h):
    return C.ISP_VAC - (C.ISP_VAC - C.ISP_SL) * np.exp(-np.maximum(h, 0) / C.H_SCALE) #Roket yükseldikçe motorun ne kadar verimli çalıştığını hesaplıyor._isp, roket yükseldikçe motorun daha güçlü ittiğini hesaplar. Simülasyon bu değeri her 0,1 saniyede itkiyi, gaz kolunu ve yanan yakıtı doğru bulmak için kullanır.

#Roketi 150 saniye uçuruyor ve olanları deftere yazıyor.
def simule(gains, sc, kayit=False):
    gains = np.atleast_2d(np.asarray(gains, float)) #Gelen PID ayarlarını tabloya çeviriyor (her satır bir ayar).
    if gains.shape[1] == 3:
        gains = np.hstack([gains, gains])
    P, K = gains.shape[0], len(sc["fm"])
    G1 = [gains[:, i][:, None] for i in range(3)] #6 sayıyı ikiye ayırıyor: ilk 3'ü Faz-1 (Kp1, Ki1, Kd1), son 3'ü Faz-2.
    G2 = [gains[:, i][:, None] for i in range(3, 6)]
    fl = sc["ucus"] #Her testin hangi yolu kullanacağını alıyor (IFT-6, 7, 8…).

    m0 = np.ones((P, 1)) * (C.M0 * sc["fm"])[None, :]
    cdA = C.CD * C.AREA * sc["fcd"][None, :]
    ft = sc["ft"][None, :] #Test koşullarını uyguluyor: ağırlık, hava direnci, motor gücü.

    v = np.zeros((P, K)) + 0.5
    h = np.zeros((P, K))
    m = m0.copy()
    T = np.zeros((P, K)) #Roketin başlangıç durumu: hız ~0, yükseklik 0, kütle dolu, itki 0.

    I = np.zeros((P, K))
    d_f = np.zeros((P, K))
    e_prev = None #PID’in hafızasını sıfırlıyor: birikim (I), türev (D), önceki hata, önceki gaz.
    u_prev = np.zeros((P, K))

    noise = np.random.default_rng(sc["seed"] + 999).normal(0, C.SENSOR_NOISE, (N_ADIM, K)) #Bütün uçuş için sensör hatalarını önceden üretiyor (±1 m/s).

    iae = np.zeros((P, K)); tv = np.zeros((P, K))
    qmax = np.zeros((P, K)); amax = np.zeros((P, K))
    log = {"t": [], "v": [], "vref": [], "u": [], "q": [], "h": []} if kayit else None #Gözlemci defteri: sapma, gaz oynaması, en yüksek basınç, en yüksek ivme; hepsi 0'dan başlıyor.

    for k in range(N_ADIM - 1): #Döngü (her 0,1 saniyede, 1500 kez)
        t = k * C.DT #Saat ilerliyor: 0; 0,1; 0,2 … 149,9.

        Kp, Ki, Kd = G1 if t < C.T_SWITCH else G2#Kazanç programlama: 75 saniyeden önce Faz-1 ayarı, sonra Faz-2 ayarı.
        vref, aref, sg = VREF[fl, k][None, :], AREF[fl, k][None, :], SG[fl, k][None, :]   #Tablodan o anın hedef hızını, hedef ivmesini ve eğimini okuyor.

        rho = C.RHO0 * np.exp(-np.maximum(h, 0) / C.H_SCALE)
        isp = _isp(h)
        T_motor = C.MDOT_ENG * isp * C.G0                     # motor başına anlık max itki, O yükseklikte havanın yoğunluğunu, motor verimini ve motor başına itkiyi hesaplıyor.

        v_olc = v + noise[k][None, :]
        e = vref - v_olc #Sensör ölçüyor (gerçek hız + küçük hata), sonra hata hesaplanıyor: hedef − ölçülen.

        if e_prev is None:
            e_prev = e.copy()
        d_f += ((e - e_prev) / C.DT - d_f) * (C.DT / (0.5 + C.DT))   # filtreli türev (0.5 s), D (türev): hatanın ne kadar hızlı değiştiği. Gürültü zıplatmasın diye yumuşatılıyor.
        e_prev = e

        m_hat = C.M0 - (m0 - m)                                # başlangıç kütle hatası bilinmez, Kontrolcünün tahmini kütle ve hava direnci. Gerçek değerleri bilmiyor, normal değerlerle tahmin ediyor.
        D_hat = 0.5 * rho * C.CD * C.AREA * v_olc ** 2

        a_cmd = aref + C.G0 * sg + D_hat / m_hat + Kp * e + Ki * I + Kd * d_f #İleri besleme + PID: istenen ivme = bilinen kısım (hedef ivme + yerçekimi + direnç) + düzeltme (P + I + D).

        u_raw = m_hat * a_cmd / (C.N_ENG * T_motor)
        u = np.clip(u_raw, C.THROTTLE_MIN, 1.0) #İstenen ivmeyi gaz koluna çeviriyor, sonra %40–100 arasına sınırlıyor.

        doyma = ((u_raw > 1.0) & (e > 0)) | ((u_raw < C.THROTTLE_MIN) & (e < 0))
        I += np.where(doyma, 0.0, e * C.DT) #Anti-windup: gaz sınırdaysa birikime ekleme yok; değilse I += e × Δt.

        n_aktif = C.N_ENG - (t >= sc["t_ariza"])[None, :]      # motor arızası: 33 -> 32, Motor arızası: arıza saniyesi geldiyse 32 motor. Sonra istenen toplam itki hesaplanıyor.
        T_cmd = u * n_aktif * T_motor * ft

        T += (T_cmd - T) * (C.DT / C.TAU_ENGINE) #Motor gecikmesi: itki istenen değere hemen değil, yavaş yavaş yaklaşıyor (τ = 0,2 s).

        D = 0.5 * rho * cdA * v ** 2
        a = (T - D) / m - C.G0 * sg
        q = 0.5 * rho * v ** 2 #Fizik: gerçek hava direnci, Newton’la ivme (itki − direnç, bölü kütle, eksi yerçekimi) ve dinamik basınç.

        iae += np.abs(v - vref) * C.DT
        tv += np.abs(u - u_prev) * (k > 0)
        qmax = np.maximum(qmax, q)
        amax = np.maximum(amax, (T - D) / m / C.G0)            # algılanan ivme (g)
        u_prev = u #Deftere yaz: sapma birikiyor, gaz oynaması birikiyor, en yüksek basınç ve ivme saklanıyor.

        v = v + a * C.DT
        h = h + v * sg * C.DT
        m = m - T / (isp * C.G0) * C.DT #Yeni durum: hız, yükseklik ve kütle (yakıt yandı) güncelleniyor. Sonra döngü başa dönüyor.

        if kayit and k % 5 == 0:
            for n_, x_ in zip(("t", "v", "vref", "u", "q", "h"), (t, v, vref, u, q, h)):
                log[n_].append(np.array(x_, copy=True)) #İstenirse (grafik çizmek için) her 0,5 saniyede bir değerleri kaydediyor.

    out = dict(iae=iae, tv=tv, qmax=qmax, amax=amax, yakit=m0 - m,
               v_son=v, v_son_ref=VREF[fl, -1][None, :] * np.ones((P, 1)),
               q_lim=C.Q_MARJ * QMAX_REF[fl][None, :] * np.ones((P, 1)), h_son=h)
    if kayit:
        out["log"] = {k_: np.array(v_) for k_, v_ in log.items()}
    return out #Defteri teslim ediyor: toplam sapma, gaz oynaması, en yüksek basınç ve ivme, yakılan yakıt, son hız, hedef son hız, max-Q sınırı, son yükseklik. Bunlar maliyet() fonksiyonunda J puanına çevriliyor.


def maliyet(r):
    """
    J = IAE/300 + ((v_son - v_ref_son)/10)^2 + 20*max(0, qmax/q_lim - 1)
        + 0.5*yakıt/2700 t + 0.003*TV(u)          (küçük = iyi), senaryo ortalaması
    uçuş defterini tek bir puana (J) çeviriyor. Puan ne kadar küçükse PID ayarı o kadar iyi."""
    J = (r["iae"] / 300.0                                                #Hedef hızdan ne kadar saptın?
         + ((r["v_son"] - r["v_son_ref"]) / 10.0) ** 2                   #Son saniyede hızın doğru mu?
         + 20.0 * np.maximum(0.0, r["qmax"] / r["q_lim"] - 1.0)          #Max-Q sınırını aştın mı?
         + 0.5 * r["yakit"] / 2700e3                                     #Ne kadar yakıt harcadın?
         + 0.003 * r["tv"])                                              #Gaz kolunu ne kadar oynattın?
    return J.mean(axis=1)                                                #9 testin her birinin puanı ayrı hesaplanıyor, sonra ortalaması alınıyor:




#Bu sınıf, GA ve Karınca’nın sahte rokete bağlandığı kapı. Algoritmalar içeride ne olduğunu bilmiyor; sadece 6 sayı veriyor, karşılığında bir puan (J) alıyor.
#Bu kod bir puan makinesi: 6 sayı (PID ayarı) gir → makine bir puan (J) versin.
class AmacFonksiyonu:
    def __init__(self, K=C.N_SCEN, seed=0):
        self.sc = senaryolar(K, seed)
        self.n_eval = 0

    def __call__(self, X):
        X = np.atleast_2d(X)
        self.n_eval += X.shape[0]
        return maliyet(simule(X, self.sc))


if __name__ == "__main__": #“Sadece bu dosya doğrudan çalıştırılırsa aşağıdakileri yap.” Başka dosya model.py’yi kullanınca bu kısım çalışmaz
    #Bu kısım, model.py doğrudan çalıştırıldığında 4 örnek PID ayarıyla puan makinesinin mantıklı sonuç verip vermediğini ve ne kadar hızlı çalıştığını kontrol etmek için var. Optimizasyonda kullanılmıyor.
    import time
    f = AmacFonksiyonu()
    X = np.array([[0.5, 0.05, 0.05, 0.5, 0.05, 0.05], [1, .2, .2, 1, .2, .2], [0, 0, 0, 0, 0, 0], [4, 1.5, 1.5, 4, 1.5, 1.5]])
    t0 = time.time(); J = f(X); print("J =", J.round(4), f"{time.time()-t0:.2f}s")
    r = simule(X, f.sc)
    for k in ["iae", "qmax", "q_lim", "yakit", "tv", "amax"]:
        print(k, np.round(r[k].mean(1), 1))
    np.set_printoptions(suppress=True, linewidth=150)   # bilimsel gösterim (e+01) kapalı
    print("v_son - ref (satır = PID, sütun = senaryo 1..9):")
    print(np.round(r["v_son"] - r["v_son_ref"], 1))
    t0 = time.time(); f(np.random.rand(30, 6) * C.UPPER); print(f"30 aday: {time.time()-t0:.2f}s")
