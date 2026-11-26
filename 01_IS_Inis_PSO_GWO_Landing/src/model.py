"""
Super Heavy iniş yanması: 1 boyutlu (düşey) dinamik model + PID kontrolcü.
Vektörize: P aday PID x K senaryo aynı anda simüle edilir.

Durum (aşağı yön pozitif):
    dh/dt = -v
    dv/dt = g - (T + D) / m
    dm/dt = -T / (Isp * g0)
    dT/dt = (T_cmd - T) / tau              (motor tepki gecikmesi)
    D = 0.5 * rho(h) * Cd * A * v * |v|    (sürükleme, harekete zıt)

Kontrol (yerçekimi telafili, kazanç programlamalı PID, hız takibi):
    faz-1 (13 motor) ve faz-2 (3 motor) için ayrı [Kp, Ki, Kd]
    e      = v_ölçülen - v_ref(h)          (pozitif: gereğinden hızlı düşüyor)
    a_cmd  = g + a_ff(h) + Kp*e + Ki*∫e dt + Kd*de/dt     (a_ff = v_ref * dv_ref/dh)
    u      = sat( m_tahmin * a_cmd / (N * T_max), u_min, 1 )
"""
import numpy as np
import config as C
from referans import Referans

REF = Referans()


def senaryolar(K, seed):
    """Belirsizlik senaryoları (Monte Carlo)."""
    r = np.random.default_rng(seed)
    return dict(
        fm=r.uniform(0.90, 1.10, K),          # başlangıç kütlesi ±%10
        fcd=r.uniform(0.70, 1.30, K),         # sürükleme katsayısı ±%30
        ft=r.uniform(0.97, 1.03, K),          # motor itki verimi ±%3
        dv0=r.normal(0.0, 0.03, K),           # başlangıç hızı hatası (%3)
        dh0=r.normal(0.0, 50.0, K),           # başlangıç irtifa hatası (m)
        seed=int(seed),
    )


def simule(gains, sc, kayit=False):
    """
    gains : (P,6) [Kp1, Ki1, Kd1, Kp2, Ki2, Kd2]  (kazanç programlamalı PID)
            (P,3) verilirse iki fazda aynı kazanç kullanılır
    sc    : senaryolar() çıktısı (K senaryo)
    dönüş : metrik sözlüğü, her biri (P,K); kayit=True ise zaman serileri de
    """
    gains = np.atleast_2d(np.asarray(gains, float))
    if gains.shape[1] == 3:                    # tek PID -> iki faz için aynı kazanç
        gains = np.hstack([gains, gains])
    P, K = gains.shape[0], len(sc["fm"])
    G1 = [gains[:, i][:, None] for i in range(3)]      # faz-1: 13 motor, yüksek hız frenleme
    G2 = [gains[:, i][:, None] for i in range(3, 6)]   # faz-2: 3 motor, hassas iniş

    m0_nom = C.M_DRY + C.M_PROP0
    m0 = m0_nom * sc["fm"][None, :] * np.ones((P, 1))
    m_dry = C.M_DRY * sc["fm"][None, :] * np.ones((P, 1))
    cdA = C.CD * C.AREA * sc["fcd"][None, :]
    ft = sc["ft"][None, :]

    h = np.ones((P, 1)) * (REF.h0 + sc["dh0"])[None, :]
    v = np.ones((P, 1)) * (REF.v0 * (1 + sc["dv0"]))[None, :]
    m = m0.copy()
    T = np.zeros((P, K))                       # motorlar t=0'da ateşleniyor
    I = np.zeros((P, K))
    d_f = np.zeros((P, K))
    e_prev = None
    u_prev = np.zeros((P, K))

    n = int(C.T_LIMIT / C.DT)
    noise = np.random.default_rng(sc["seed"] + 999).normal(0, C.SENSOR_NOISE, (n, K))

    aktif = np.ones((P, K), bool)
    v_td = np.full((P, K), np.nan)
    t_td = np.full((P, K), np.nan)
    iae = np.zeros((P, K))
    tv = np.zeros((P, K))
    yakit_bitti = np.zeros((P, K), bool)
    log = {"t": [], "h": [], "v": [], "vref": [], "u": [], "m": []} if kayit else None

    for k in range(n):
        t = k * C.DT
        faz1 = t < C.T_SWITCH
        N = C.N_ENGINES_HIGH if faz1 else C.N_ENGINES_LOW
        Kp, Ki, Kd = G1 if faz1 else G2
        Tmax = N * C.T_MAX_ENGINE

        vref = REF.v_ref(np.maximum(h, 0.0))
        e = (v + noise[k][None, :]) - vref
        if e_prev is None:
            e_prev = e.copy()
        # filtrelenmiş türev (tau_f = 0.2 s), ölçüm gürültüsünü bastırmak için
        d_raw = (e - e_prev) / C.DT
        d_f += (d_raw - d_f) * (C.DT / (0.2 + C.DT))
        e_prev = e

        m_hat = m0_nom - (m0 - m)              # başlangıç kütlesi hatası bilinmiyor
        a_ff = REF.a_ff(np.maximum(h, 0.0), vref)      # referans ivme ileri beslemesi
        a_cmd = C.G0 + a_ff + Kp * e + Ki * I + Kd * d_f
        u_raw = m_hat * a_cmd / Tmax
        u = np.clip(u_raw, C.THROTTLE_MIN, 1.0)
        # anti-windup: doymadaysa ve doymayı artıracaksa integrali durdur
        doyma = ((u_raw > 1.0) & (e > 0)) | ((u_raw < C.THROTTLE_MIN) & (e < 0))
        I += np.where(doyma | ~aktif, 0.0, e * C.DT)

        yakit_var = m > m_dry
        yakit_bitti |= ~yakit_var & aktif
        T_cmd = np.where(yakit_var, u * Tmax * ft, 0.0)
        T += (T_cmd - T) * (C.DT / C.TAU_ENGINE)

        rho = C.RHO0 * np.exp(-np.maximum(h, 0) / C.H_SCALE)
        D = 0.5 * rho * cdA * v * np.abs(v)
        a = C.G0 - (T + D) / m

        # yalnızca aktif (henüz inmemiş) olanları ilerlet
        v_new = v + a * C.DT
        h_new = h - 0.5 * (v + v_new) * C.DT
        m_new = np.maximum(m - T / (C.ISP_SL * C.G0) * C.DT, m_dry * 0.999)
        iae += np.where(aktif, np.abs(v - vref) * C.DT, 0.0)
        tv += np.where(aktif, np.abs(u - u_prev) * (k > 0), 0.0)
        u_prev = u

        indi = aktif & (h_new <= 0.0)
        # temas anındaki hızı doğrusal interpolasyonla bul
        frac = np.where(indi, h / np.maximum(h - h_new, 1e-9), 0.0)
        v_td = np.where(indi, v + frac * (v_new - v), v_td)
        t_td = np.where(indi, t + frac * C.DT, t_td)

        h = np.where(aktif, h_new, h)
        v = np.where(aktif, v_new, v)
        m = np.where(aktif, m_new, m)
        aktif &= ~indi

        if kayit:
            log["t"].append(t); log["h"].append(h.copy()); log["v"].append(v.copy())
            log["vref"].append(vref.copy()); log["u"].append(u.copy()); log["m"].append(m.copy())
        if not aktif.any():
            break

    out = dict(v_td=v_td, t_td=t_td, iae=iae, tv=tv,
               yakit=(m0 - m), inmedi=aktif, h_son=h, yakit_bitti=yakit_bitti)
    if kayit:
        out["log"] = {k_: np.array(v_) for k_, v_ in log.items()}
    return out


def maliyet(r):
    """
    Amaç fonksiyonu J (küçük = iyi), (P,K) -> (P,) senaryo ortalaması.
      J = IAE/50 + ((v_td - v_hedef)/2)^2 + 0.5*yakıt/70t + 0.02*TV(u) + cezalar
    """
    v_td = np.nan_to_num(r["v_td"], nan=0.0)
    J = (r["iae"] / 50.0
         + ((v_td - C.V_TOUCHDOWN_TARGET) / 2.0) ** 2
         + 0.5 * r["yakit"] / 70e3
         + 0.02 * r["tv"])
    J += np.where(r["inmedi"], 100.0 + r["h_son"] / 10.0, 0.0)   # süre içinde inemedi
    J += np.where(r["yakit_bitti"], 50.0, 0.0)                    # yakıt bitti
    return J.mean(axis=1)


class AmacFonksiyonu:
    """Sabit eğitim senaryolarıyla deterministik amaç fonksiyonu + sayaç."""

    def __init__(self, K=C.N_SCEN, seed=0):
        self.sc = senaryolar(K, seed)
        self.n_eval = 0

    def __call__(self, X):
        X = np.atleast_2d(X)
        self.n_eval += X.shape[0]
        return maliyet(simule(X, self.sc))


if __name__ == "__main__":
    import time
    f = AmacFonksiyonu()
    X = np.array([[1.0, 0.1, 0.1], [2.0, 0.5, 0.2], [0.2, 0.0, 0.0], [4, 1.5, 0.8]])
    t0 = time.time(); J = f(X); print("J =", J, f"{time.time()-t0:.2f}s")
    r = simule(X, f.sc)
    print("v_td:", np.round(r["v_td"], 1))
    print("yakıt t:", np.round(r["yakit"] / 1e3, 1))
    print("t_td:", np.round(r["t_td"], 1))
    t0 = time.time(); f(np.random.rand(30, 3) * [5, 2, 1]); print(f"30 aday: {time.time()-t0:.2f}s")
