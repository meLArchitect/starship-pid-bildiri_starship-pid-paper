# İniş Fazı – PSO ve Gri Kurt Optimizasyonu ile PID Ayarı (İrem)

Super Heavy iniş yanması (IFT-8, 06:35–07:00) için iki fazlı PID gaz kolu denetleyicisinin
6 kazancı **Parçacık Sürü Optimizasyonu (PSO)** ve **Gri Kurt Optimizasyonu (GWO)** ile ayarlanır.

| Dosya | İçerik |
|---|---|
| `IS_Bildiri_Inis_PSO_GWO.docx` | Bildiri (Türkçe, TNR 12, 1.5 satır, 9 sayfa, 15 kaynak) |
| `IS_Sunum_Inis_PSO_GWO.pptx` | Sunum (9 slayt, konuşmacı notlu) |
| `src/referans.py` | IFT-8 verisi → hız–irtifa referansı v_ref(h) |
| `src/model.py` | 1-B iniş dinamiği, iki fazlı PID, Monte Carlo, amaç fonksiyonu |
| `src/optimizasyon.py` | PSO, GWO, rastgele arama (sıfırdan, numpy) |
| `src/deneyler.py` | 10 koşu × 3 yöntem, 200 senaryoluk test → `sonuclar_results/*.csv` |
| `src/grafikler.py` | Bildiri şekilleri → `sonuclar_results/sekiller_figures/*.png` |

## Ana sonuçlar
- GWO en iyi çözümün %1 yakınına **369**, PSO **900** değerlendirmede ulaşır; PSO 2/10 koşuda erken yakınsar (p = 0,026).
- Optimize PID, elle ayarlı PID'e göre maliyeti **%40**, takip hatasını **%48** azaltır; ort. temas hızı 1,81 → 1,18 m/s.

## Çalıştırma
```bash
pip install numpy pandas matplotlib scipy
cd src
python deneyler.py    # ~2 dk
python grafikler.py
```
Veri: `Optimizasyon/ift8_results.csv` (iki üst klasör).
