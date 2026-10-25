"""Multi-site data. Real use: a long CSV with columns timestamp, site, lat, lon, elevation,
wind_speed (+ optional met columns). Fallback: synthetic spatially-correlated sites."""
import numpy as np
import pandas as pd
from .data import synthetic_scada


def synthetic_sites(n_sites=6, n_hours=24 * 60, seed=0):
    rng = np.random.default_rng(seed)
    common = synthetic_scada(n_hours, seed)["wind_speed"].values
    lat, lon = rng.uniform(-34, 31, n_sites), rng.uniform(-17, 40, n_sites)
    elev = rng.uniform(0, 2500, n_sites)
    frames = []
    for s in range(n_sites):
        own = synthetic_scada(n_hours, seed + 1 + s)["wind_speed"].values
        lag = s % 4
        ws = np.clip(0.6 * np.roll(common, lag) + 0.4 * own + elev[s] / 1500, 0.2, None)
        frames.append(pd.DataFrame({"timestamp": pd.date_range("2018-01-01", periods=n_hours, freq="h"),
                                    "site": f"S{s:02d}", "lat": lat[s], "lon": lon[s],
                                    "elevation": elev[s], "wind_speed": ws}))
    return pd.concat(frames, ignore_index=True)


def load_sites_csv(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    need = {"timestamp", "site", "lat", "lon", "elevation", "wind_speed"}
    if not need <= set(df.columns):
        raise ValueError(f"multi-site CSV needs columns {sorted(need)}")
    return df


def to_tensor(df):
    """-> wind (T, S), static (S, 3)"""
    piv = df.pivot_table(index="timestamp", columns="site", values="wind_speed").interpolate().dropna()
    st = df.groupby("site")[["lat", "lon", "elevation"]].first().loc[piv.columns]
    return piv.values.astype("float32"), st.values.astype("float32"), list(piv.columns)


def windows(W, lookback, horizons):
    """Each sample: all sites' history (lookback, S) -> target site's future at each horizon.
    Returns X (N, lookback, S), Y (N, H), site_idx (N,)"""
    T, S = W.shape; H = max(horizons); X, Y, idx = [], [], []
    for t in range(lookback, T - H + 1):
        for s in range(S):
            X.append(np.roll(W[t - lookback:t], -s, axis=1))   # target site first column
            Y.append([W[t + h - 1, s] for h in horizons]); idx.append(s)
    return np.array(X, "float32"), np.array(Y, "float32"), np.array(idx)
