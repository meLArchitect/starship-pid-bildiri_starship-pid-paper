"""
Model parametreleri - Super Heavy + Starship kalkış (ascent) fazı
-----------------------------------------------------------------
Açık kaynaklardan alınan YAKLAŞIK değerler; kesin değerler yayımlanmadığı için varsayımdır.
Belirsizlik Monte Carlo senaryolarıyla modele katılır.
"""
import os

# --- Dosya yolları ---
KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# CSV dosyaları şu sırayla aranır: proje klasörü, src, Optimizasyon klasörü (iki üst)
_ADAYLAR = [KOK, os.path.join(KOK, "src"), os.path.join(KOK, "veri_data"), os.path.dirname(os.path.dirname(KOK))]
VERI_KLASORU = next((k for k in _ADAYLAR if os.path.exists(os.path.join(k, "ift8_results.csv"))), None)
if VERI_KLASORU is None:
    raise FileNotFoundError("ift6/7/8_results.csv bulunamadı. Şu klasörlerden birine koyun:\n  " + "\n  ".join(_ADAYLAR))
UCUSLAR = ["ift6", "ift7", "ift8"]                             # ift6_results.csv ...
SONUC_KLASORU = os.path.join(KOK, "sonuclar_results")
SEKIL_KLASORU = os.path.join(SONUC_KLASORU, "sekiller_figures")

# --- Fiziksel sabitler ---
G0 = 9.80665
RHO0 = 1.225
H_SCALE = 8500.0

# --- Araç (Super Heavy + Starship yığını) ---
M0 = 5000e3            # kalkış kütlesi, kg (booster ~3450 t + gemi ~1550 t, yaklaşık)
DIAMETER = 9.0
AREA = 3.1416 * (DIAMETER / 2) ** 2
CD = 0.5               # ortalama sürükleme katsayısı (burun önde)

# --- Raptor 2 (deniz seviyesi nozullu) x 33 ---
N_ENG = 33
MDOT_ENG = 705.0       # motor başına maksimum kütle debisi, kg/s (2.26 MN / (327 s * g0))
ISP_SL = 327.0         # deniz seviyesi özgül itki, s
ISP_VAC = 350.0        # vakumda (SL nozul) yaklaşık özgül itki, s
THROTTLE_MIN = 0.40
TAU_ENGINE = 0.20      # itki tepki zaman sabiti, s

# --- Uçuş fazları ---
T_END = 150.0          # MECO öncesi (sıcak ayrılma hazırlığından önce) analiz sonu, s
T_SWITCH = 75.0        # faz-1: kalkış + max-Q bölgesi  |  faz-2: üst atmosfer hızlanma

# --- Simülasyon ---
DT = 0.1
SENSOR_NOISE = 1.0     # m/s, hız ölçüm gürültüsü (1-sigma)
Q_MARJ = 1.05          # max-Q sınırı = 1.05 x referans uçuşun max-Q'su
P_MOTOR_ARIZA = 1 / 3  # bir senaryoda tek motor arızası olasılığı (33 -> 32)

# --- PID arama sınırları [Kp1, Ki1, Kd1, Kp2, Ki2, Kd2] ---
LOWER = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
UPPER = [2.0, 0.5, 1.0, 2.0, 0.5, 1.0]
PARAM_ADLARI = ["Kp1", "Ki1", "Kd1", "Kp2", "Ki2", "Kd2"]

# --- Optimizasyon bütçesi (adil karşılaştırma için hepsinde aynı) ---
POP = 30
ITER = 60              # 30 x 60 = 1800 değerlendirme / koşu
N_RUNS = 10
N_SCEN = 9             # eğitim: her uçuştan 3 senaryo
