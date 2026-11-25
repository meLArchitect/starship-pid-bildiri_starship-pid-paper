# sonfinal — Optimizasyon Dersi Ortak Bildiri (İrem Sert · Melih Demir)

**Konu:** Super Heavy kalkış ve iniş fazlarında kazanç programlamalı PID denetleyicilerin
GA, ACO_R, PSO, GWO ve iki hibrit yöntemle (GA-ACO_R, PSO-GWO) ayarlanması.
**Son teslim:** 4 Aralık 2026 · Ders: Optimizasyon (KKÜ, EEM Yüksek Lisans)

## Klasör düzeni

| Klasör | İçerik |
|---|---|
| `00_Makale/` | **Teslim edilecek birleşik makale.** Türkçe ve İngilizce Word (.docx) + PDF; `makale_kaynak_kodu/` makaleyi yeniden üreten betik (`node build_iki_sutun.js tr` / `en` (tek sütunlu sürüm: `build_tek_sutun.js`)) |
| `01_Irem_Sert_Inis_PSO_GWO/` | İrem'in iniş çalışması: bildiri taslağı, sunumlar, `src/` kod, `sonuclar/`, `veri/`, görseller |
| `02_Melih_Demir_Kalkis_GA_ACO/` | Melih'in kalkış çalışması: bildiri taslakları (ablasyonlu dahil), sunumlar, `src/`, `sonuclar/`, `veri/`, görseller |
| `03_Ortak_Karsilastirma_ve_Hibrit/` | Dört algoritmanın iki probleme birlikte uygulanması + iki hibrit; `src/` ve `sonuclar/kalkis`, `sonuclar/inis` |
| `04_Sekiller/` | Makaledeki tüm şekiller, `tr/` ve `en/` olarak (300 dpi PNG) |

## Kodları yeniden çalıştırma

```bash
pip install numpy pandas scipy matplotlib
# Ortak deney (her problem için 7 yöntem x 10 koşu; kalkış ~10 dk, iniş ~5 dk)
cd 03_Ortak_Karsilastirma_ve_Hibrit/src
python ortak_deney.py kalkis
python ortak_deney.py inis
# Şekiller (tr / en)
python ortak_sekiller.py tr
python problem_sekilleri.py kalkis tr
python problem_sekilleri.py inis tr
```

Yazarların `src/` klasörlerindeki kod, tohum ve senaryolar değiştirilmeden kullanılır; bu yüzden
GA/ACO_R (kalkış) ve PSO/GWO (iniş) sonuçları önceki ayrı çalışmalarla birebir aynıdır.

## Ana sonuçlar (10 koşu, 1800 değerlendirme)

- Tüm yöntemler aynı optimuma ulaşıyor; fark güvenilirlikte (Friedman: kalkış p = 7,5e-8, iniş p = 0,018).
- GWO her iki problemde en hızlı başlangıç; PSO her ikisinde erken yakınsama (9/10, 8/10).
- GA-ACO_R kalkışta en iyi ortalama ve sıra (std GA'ya göre ~500x düşük).
- PSO-GWO inişte en iyi ortalama ve en düşük sapma (std PSO'ya göre 148x düşük, 10/10 başarı).
- 200 test senaryosunda elle ayarlı PID'e göre: kalkış maliyet -%11, salınım -%67; iniş maliyet -%40, takip hatası -%48, temas hızı 1,81 → 1,19 m/s.

## Teslimden önce doldurulacaklar

- Ortak GitHub deposunun bağlantısı (şu an `https://github.com/meLArchitect` + not)
- Yazar sırası şu an "İrem Sert, Melih Demir"; değiştirmek için `makale_kaynak_kodu/content_tr.js` ve `content_en.js` içindeki `authors` satırı
