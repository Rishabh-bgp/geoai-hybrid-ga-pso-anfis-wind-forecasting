"""Write a small synthetic multi-site CSV to data/synthetic/synthetic_sites.csv."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.sites import synthetic_sites
os.makedirs("data/synthetic", exist_ok=True); df = synthetic_sites(n_hours=24 * 60, seed=0); df.to_csv("data/synthetic/synthetic_sites.csv", index=False, float_format="%.3f")
print("wrote data/synthetic/synthetic_sites.csv", len(df))
