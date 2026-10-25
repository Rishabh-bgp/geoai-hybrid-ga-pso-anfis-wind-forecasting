"""Write a small synthetic multi-site CSV to data/synthetic_sites.csv."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.sites import synthetic_sites
os.makedirs("data", exist_ok=True); df = synthetic_sites(n_hours=24 * 60, seed=0); df.to_csv("data/synthetic_sites.csv", index=False)
print("wrote data/synthetic_sites.csv", len(df))
