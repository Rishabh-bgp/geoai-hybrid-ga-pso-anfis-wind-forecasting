"""Layer 1 (GA over discrete architecture genes) and Layer 2 (PSO over continuous genes)."""
import numpy as np

GENES = {"filters": [16, 32, 64], "hidden": [16, 32, 64], "n_rules": [4, 8, 16]}


def ga(fitness, pop=6, gens=3, pmut=0.3, seed=0, log=print):
    rng = np.random.default_rng(seed); keys = list(GENES)
    P = [{k: rng.choice(GENES[k]).item() for k in keys} for _ in range(pop)]
    cache = {}
    f = lambda c: cache.setdefault(tuple(c.values()), fitness(c))
    for g in range(gens):
        scored = sorted(P, key=f); log(f"GA gen {g}: best val loss {f(scored[0]):.4f} {scored[0]}")
        elite = scored[: max(2, pop // 2)]; P = elite[:]
        while len(P) < pop:
            a, b = rng.choice(len(elite), 2, replace=False)
            child = {k: (elite[a] if rng.random() < .5 else elite[b])[k] for k in keys}   # uniform crossover
            for k in keys:
                if rng.random() < pmut: child[k] = rng.choice(GENES[k]).item()
            P.append(child)
    best = min(P, key=f); return best, f(best)


def pso(fitness, bounds, particles=4, iters=3, w=0.7, c1=1.5, c2=1.5, seed=0, log=print):
    rng = np.random.default_rng(seed); lo, hi = np.array(bounds).T
    x = rng.uniform(lo, hi, (particles, len(lo))); v = np.zeros_like(x)
    pb, pbf = x.copy(), np.array([fitness(p) for p in x]); gb = pb[pbf.argmin()].copy()
    for it in range(iters):
        r1, r2 = rng.random(x.shape), rng.random(x.shape)
        v = w * v + c1 * r1 * (pb - x) + c2 * r2 * (gb - x); x = np.clip(x + v, lo, hi)
        fx = np.array([fitness(p) for p in x]); imp = fx < pbf
        pb[imp], pbf[imp] = x[imp], fx[imp]; gb = pb[pbf.argmin()].copy()
        log(f"PSO iter {it}: best val loss {pbf.min():.4f} at {np.round(gb, 5).tolist()}")
    return gb, pbf.min()
