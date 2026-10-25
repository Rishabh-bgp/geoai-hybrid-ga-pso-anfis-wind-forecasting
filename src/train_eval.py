"""GA -> PSO -> ANFIS/CNN-LSTM quantile model. Reports MAE/RMSE/MAPE/R2 of the median,
pinball loss, CRPS (quantile approximation), 90% interval coverage and PIT histogram, per horizon."""
import argparse, json, os
import numpy as np, torch
from .sites import synthetic_sites, load_sites_csv, to_tensor, windows
from .model import GeoHybrid, pinball, QUANTILES
from .optimize import ga, pso
from .metrics import regression_metrics


def train(cfg, data, epochs, mc=False):
    torch.manual_seed(0)
    (Xtr, Str, Ytr), (Xva, Sva, Yva) = data
    m = GeoHybrid(Xtr.shape[2], Str.shape[1], range(Ytr.shape[1]), cfg["filters"], cfg["hidden"],
                  cfg["n_rules"], cfg.get("dropout", 0.1))
    opt = torch.optim.Adam(m.parameters(), lr=cfg.get("lr", 2e-3))
    for _ in range(epochs):
        m.train(); perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 256):
            b = perm[i:i + 256]; opt.zero_grad(); pinball(m(Xtr[b], Str[b]), Ytr[b]).backward(); opt.step()
    m.eval()
    with torch.no_grad(): v = float(pinball(m(Xva, Sva), Yva))
    return m, v


def mc_predict(m, X, S, n=20):
    m.train()   # Monte Carlo dropout
    with torch.no_grad(): q = torch.stack([m(X, S) for _ in range(n)]).mean(0)
    m.eval(); return q.numpy()


def crps_from_quantiles(q, y):
    t = np.array(QUANTILES); e = y[..., None] - q
    return float(2 * np.maximum(t * e, (t - 1) * e).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/synthetic/synthetic_sites.csv" if os.path.exists("data/synthetic/synthetic_sites.csv") else None, help="multi-site CSV (timestamp,site,lat,lon,elevation,wind_speed)")
    ap.add_argument("--horizons", default="1,6,24"); ap.add_argument("--lookback", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="results"); a = ap.parse_args()
    hz = [int(h) for h in a.horizons.split(",")]
    ga_kw, pso_kw, search_ep = dict(pop=6, gens=3), dict(particles=4, iters=3), 3
    if a.quick: ga_kw, pso_kw, search_ep, a.epochs, a.lookback = dict(pop=3, gens=1), dict(particles=2, iters=1), 1, 3, 24
    df = load_sites_csv(a.csv) if a.csv else synthetic_sites(n_hours=24 * (40 if a.quick else 90))
    W, static, names = to_tensor(df)
    T = len(W); a1, a2 = int(T * .7), int(T * .85)
    mu, sd = W[:a1].mean(), W[:a1].std(); Wn = (W - mu) / sd
    st = (static - static.mean(0)) / (static.std(0) + 1e-6)
    def part(lo, hi):
        X, Y, idx = windows(Wn[lo:hi], a.lookback, hz)
        last = X[:, -1, :1]   # target site's last reading: predict the change from it (residual on persistence)
        return torch.tensor(X), torch.tensor(st[idx]), torch.tensor(Y - last)
    tr, va, te = part(0, a1), part(a1 - a.lookback, a2), part(a2 - a.lookback, T)
    log = []; L = lambda s: (print(s), log.append(s))
    best, _ = ga(lambda c: train(c, (tr, va), search_ep)[1], log=L, **ga_kw)
    gb, _ = pso(lambda p: train({**best, "lr": 10 ** p[0], "dropout": p[1]}, (tr, va), search_ep)[1],
                [(-3.5, -2.0), (0.0, 0.3)], log=L, **pso_kw)
    cfg = {**best, "lr": float(10 ** gb[0]), "dropout": float(gb[1])}
    m, _ = train(cfg, (tr, va), a.epochs)
    last = te[0][:, -1, :1].numpy()
    q = (mc_predict(m, te[0], te[1]) + last[:, :, None]) * sd + mu; y = (te[2].numpy() + last) * sd + mu
    res = {"config": cfg, "search_log": log, "horizons": {}}
    for j, h in enumerate(hz):
        qj, yj = q[:, j], y[:, j]
        pit = (yj[:, None] > qj).mean(1)   # fraction of quantiles below obs (coarse PIT)
        res["horizons"][f"{h}h"] = {**regression_metrics(yj, qj[:, 2]),
            "CRPS": crps_from_quantiles(qj, yj), "coverage_90": float(((yj >= qj[:, 0]) & (yj <= qj[:, -1])).mean()),
            "PIT_hist": np.bincount((pit * 5).round().astype(int), minlength=6).tolist(),
            "persistence_RMSE": float(np.sqrt(((yj - (te[0][:, -1, 0].numpy() * sd + mu)) ** 2).mean()))}
    os.makedirs(a.out, exist_ok=True); json.dump(res, open(f"{a.out}/results.json", "w"), indent=2)
    print(json.dumps(res["horizons"], indent=2)); print("config:", cfg)


if __name__ == "__main__":
    main()
