"""
Model parametreleri / Model parameters
--------------------------------------
Super Heavy (Booster) için açık kaynaklardan alınan YAKLAŞIK değerler.
Kesin değerler SpaceX tarafından yayımlanmadığı için bunlar varsayımdır;
belirsizlik Monte Carlo senaryolarıyla (± aralıklar) modele katılır.
"""
import os

# --- Dosya yolları ---
KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ift8_results.csv şu sırayla aranır: proje klasörü, src, Optimizasyon klasörü (iki üst)
_ADAYLAR = [KOK, os.path.join(KOK, "src"), os.path.join(KOK, "veri_data"), os.path.dirname(os.path.dirname(KOK))]
_k = next((k for k in _ADAYLAR if os.path.exists(os.path.join(k, "ift8_results.csv"))), None)
if _k is None:
    raise FileNotFoundError("ift8_results.csv bulunamadı. Şu klasörlerden birine koyun:\n  " + "\n  ".join(_ADAYLAR))
VERI_DOSYASI = os.path.join(_k, "ift8_results.csv")
SONUC_KLASORU = os.path.join(KOK, "sonuclar_results")
SEKIL_KLASORU = os.path.join(SONUC_KLASORU, "sekiller_figures")

# --- Fiziksel sabitler ---
G0 = 9.80665            # m/s^2
RHO0 = 1.225            # deniz seviyesi hava yoğunluğu, kg/m^3
H_SCALE = 8500.0        # atmosfer ölçek yüksekliği, m

# --- Araç (Super Heavy) ---
M_DRY = 275e3           # kuru kütle, kg (kamuya açık tahminlerin üst ucu ~275 t)
M_PROP0 = 100e3         # iniş yanması başında kalan yakıt, kg (~%3 x 3400 t)
DIAMETER = 9.0          # m
AREA = 3.1416 * (DIAMETER / 2) ** 2   # ~63.6 m^2
CD = 1.0                # sürükleme katsayısı (motor tarafı önde, kör gövde)

# --- Raptor 2 motoru ---
T_MAX_ENGINE = 2.26e6   # deniz seviyesi max itki, N (~230 tf)
ISP_SL = 327.0          # s
THROTTLE_MIN = 0.35     # en düşük gaz kolu (varsayım; 3 motorla T/W<1 inebilmek için)
TAU_ENGINE = 0.20       # itki tepki zaman sabiti, s (birinci derece gecikme)

# Motor sayısı takvimi (IFT-8 verisindeki ivme değişiminden):
# iniş yanması 13 motorla başlar, ~6 s sonra 3 merkez motora düşülür.
N_ENGINES_HIGH = 13
N_ENGINES_LOW = 3
T_SWITCH = 6.0          # s, ateşlemeden sonra

# --- Simülasyon ---
DT = 0.05               # s
T_LIMIT = 60.0          # s
V_TOUCHDOWN_TARGET = 1.0   # m/s, hedef temas (yakalama) hızı
SENSOR_NOISE = 0.5      # m/s, hız ölçüm gürültüsü (1-sigma)

# --- PID arama sınırları [Kp1, Ki1, Kd1, Kp2, Ki2, Kd2] ---
LOWER = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
UPPER = [10.0, 5.0, 2.0, 10.0, 5.0, 2.0]
PARAM_ADLARI = ["Kp1", "Ki1", "Kd1", "Kp2", "Ki2", "Kd2"]

# --- Optimizasyon bütçesi (adil karşılaştırma için hepsinde aynı) ---
POP = 30
ITER = 60               # 30 x 60 = 1800 değerlendirme / koşu
N_RUNS = 10             # bağımsız koşu sayısı (farklı tohumlar)
N_SCEN = 8              # her değerlendirmede Monte Carlo senaryo sayısı
