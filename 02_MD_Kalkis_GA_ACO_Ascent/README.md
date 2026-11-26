# Kalkış Fazı – Genetik Algoritma ve Karınca Kolonisi ile PID Ayarı (Melih)

Super Heavy + Starship kalkış fazının ilk 150 saniyesi (IFT-6/7/8) için iki fazlı PID gaz kolu
denetleyicisinin 6 kazancı **Genetik Algoritma (GA)** ve **sürekli Karınca Kolonisi (ACO_R)** ile ayarlanır.

| Dosya | İçerik |
|---|---|
| `MD_Bildiri_Kalkis_GA_ACO.docx` | Bildiri (Türkçe, TNR 12, 1.5 satır, 9 sayfa, 16 kaynak) |
| `MD_Sunum_Kalkis_GA_ACO.pptx` | Sunum (9 slayt, konuşmacı notlu) |
| `src/referans.py` | 3 uçuşun verisi → v_ref(t), uçuş yolu açısı, max-Q |
| `src/model.py` | Uçuş yolu dinamiği, irtifaya bağlı Isp, motor arızası, max-Q kısıtı, PID |
| `src/optimizasyon.py` | GA, ACO_R, rastgele arama (sıfırdan, numpy) |
| `src/deneyler.py` | 10 koşu × 3 yöntem, 200 senaryoluk test → `sonuclar_results/*.csv` |
| `src/grafikler.py` | Bildiri şekilleri → `sonuclar_results/sekiller_figures/*.png` |

## Ana sonuçlar
- GA ilk aşamada daha hızlı (300 değerlendirmede J: 0,681 vs 0,692); ACO_R son çözümde çok daha tutarlı (std 3×10⁻⁷ vs 9×10⁻⁵, p < 0,001).
- Optimize PID, elle ayarlı PID'e göre gaz kolu titreşimini **%67**, maliyeti **%11** azaltır; takip hatası %7 artar, max-Q hiçbir senaryoda aşılmaz.

## Çalıştırma
```bash
pip install numpy pandas matplotlib scipy
cd src
python deneyler.py    # ~5 dk
python grafikler.py
```
Veri: `Optimizasyon/ift6_results.csv`, `ift7_results.csv`, `ift8_results.csv` (iki üst klasör).
