"""
IFT-8 Booster iniş verisinden referans profil çıkarımı.
Reference profile extraction from IFT-8 booster landing telemetry.

Çıktı: t (s), v (m/s, aşağı yönlü pozitif), h (m), ve v_ref(h) fonksiyonu.
"""
import numpy as np
import pandas as pd
from config import VERI_DOSYASI, V_TOUCHDOWN_TARGET

T_BASLA, T_BITIS = "00:06:30", "00:07:00"
T_ATESLEME = "00:06:35"   # iniş yanması başlangıcı (ivmenin aniden arttığı an)


def _saniye(s):
    h, m, sec = map(int, s.split(":"))
    return h * 3600 + m * 60 + sec


def veri_yukle(dosya=VERI_DOSYASI):
    d = pd.read_csv(dosya, encoding="utf-8-sig")
    d = d.drop_duplicates(subset="time", keep="first")
    d["ts"] = d["time"].map(_saniye)
    d = d[(d.ts >= _saniye(T_BASLA)) & (d.ts <= _saniye(T_BITIS))].copy()
    d = d.sort_values("ts")
    v = d["booster_speed"].astype(float).values / 3.6          # km/h -> m/s
    t = (d["ts"] - _saniye(T_ATESLEME)).astype(float).values   # ateşleme t=0

    # OCR aykırı değeri: 06:46'da hız 0 okunmuş -> doğrusal interpolasyon
    hatali = v < 1e-6
    hatali[-1] = False
    v[hatali] = np.interp(t[hatali], t[~hatali], v[~hatali])

    # İrtifa: OCR tam sayı km verdiği için hızdan geriye integre edilir.
    # h(t) = ∫_t^{t_son} v dτ  (son noktada h = 0: yakalama noktası)
    # Varsayım: bu fazda hız vektörü büyük ölçüde düşeydir.
    seg = 0.5 * (v[1:] + v[:-1]) * np.diff(t)
    h = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
    return t, v, h, d


class Referans:
    """v_ref(h): irtifaya göre hedef iniş hızı (IFT-8'den)."""

    def __init__(self):
        self.t, self.v, self.h, _ = veri_yukle()
        # interp için artan h sırası
        hs, vs = self.h[::-1].copy(), self.v[::-1].copy()
        vs[0] = V_TOUCHDOWN_TARGET          # h = 0'da hedef temas hızı
        self._h, self._v = hs, vs
        # dv_ref/dh: ileri besleme (feedforward) ivmesi için eğim
        self._dvdh = np.gradient(vs, hs)
        i0 = np.argmin(np.abs(self.t - 0.0))
        self.h0, self.v0 = self.h[i0], self.v[i0]   # ateşleme anı koşulları

    def v_ref(self, h):
        return np.interp(h, self._h, self._v)

    def a_ff(self, h, v):
        """Profili izlemek için gereken yavaşlama (yukarı yönlü ivme) = v * dv_ref/dh."""
        return v * np.interp(h, self._h, self._dvdh)


if __name__ == "__main__":
    t, v, h, _ = veri_yukle()
    for a, b, c in zip(t, v, h):
        print(f"t={a:5.0f} s  v={b:7.1f} m/s  h={c:7.0f} m")
