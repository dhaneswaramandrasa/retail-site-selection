"""Build web/data/sites.json for the retail site-selection demo.

Pipeline, mirroring the case study:
  1. Input    a synthetic retailer with ~70 existing outlets and their monthly sales
  2. Describe every 400 m grid cell by demographic, POI and mobility attributes
  3. Learn    a ridge regression of outlet sales on the attributes of its cell
  4. Score    every cell with the fitted model
  5. Decide   (in the browser) pick the top cells that are far enough from existing outlets

Data:
  - POIs and transit stops: OpenStreetMap (ODbL), fetched once with Overpass into
    data/raw/osm_pois_south_central_jakarta.json (see README for the query).
  - Demographics (population density, household spend) and footfall: SYNTHETIC,
    generated below so their shape follows the real POI geography. No client data.

Only numpy is needed:  python scripts/build_data.py
"""
import json
import itertools
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "osm_pois_south_central_jakarta.json"
OUT = ROOT / "web" / "data" / "sites.json"

rng = np.random.default_rng(7)

# ---------- 1. grid ----------
S, W, N, E = -6.32, 106.74, -6.14, 106.92
STEP = 0.0036  # ~400 m
lats = np.arange(S, N, STEP)
lons = np.arange(W, E, STEP)
ny, nx = len(lats), len(lons)
KM_LAT, KM_LON = 110.6, 110.6 * np.cos(np.radians(-6.23))

# ---------- 2. POIs from OSM into per-cell counts ----------
GROUPS = {
    "food": {"restaurant", "cafe", "fast_food"},
    "shops": {"shop", "marketplace"},
    "finance": {"bank", "atm"},
    "health": {"clinic", "hospital", "pharmacy"},
    "education": {"school", "university"},
    "worship": {"place_of_worship"},
    "rail": {"station", "halt"},
    "bus": {"bus_stop", "bus_station"},
}
counts = {g: np.zeros((ny, nx)) for g in GROUPS}
pois = []
for e in json.loads(RAW.read_text())["elements"]:
    t = e.get("tags", {})
    kind = t.get("amenity") or ("shop" if "shop" in t else None) or t.get("railway") or t.get("highway")
    for g, kinds in GROUPS.items():
        if kind in kinds:
            i, j = int((e["lat"] - S) / STEP), int((e["lon"] - W) / STEP)
            if 0 <= i < ny and 0 <= j < nx:
                counts[g][i, j] += 1
                if g not in ("bus",):
                    pois.append([round(e["lon"], 5), round(e["lat"], 5), list(GROUPS).index(g)])
            break


def smooth(a, sigma_cells):
    """Gaussian catchment: what a shopper within walking/riding distance would see."""
    r = int(3 * sigma_cells)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_cells) ** 2)
    k /= k.sum()
    a = np.apply_along_axis(lambda v: np.convolve(np.pad(v, r, mode="edge"), k, "valid"), 0, a)
    return np.apply_along_axis(lambda v: np.convolve(np.pad(v, r, mode="edge"), k, "valid"), 1, a)


def field(sigma):
    """Smooth random field in [0, 1] for synthetic layers."""
    f = smooth(rng.normal(size=(ny, nx)), sigma)
    return (f - f.min()) / (f.max() - f.min())


catch = {g: smooth(c, 2.0) for g, c in counts.items()}  # ~800 m catchment
activity = sum(catch[g] for g in ("food", "shops", "finance"))
act = activity / activity.max()

# ---------- synthetic demographics and footfall ----------
yy, xx = np.meshgrid(lats, lons, indexing="ij")


def bump(lat, lon, km):
    d2 = ((yy - lat) * KM_LAT) ** 2 + ((xx - lon) * KM_LON) ** 2
    return np.exp(-d2 / (2 * km**2))


# Residential density peaks between commercial strips, not on top of them.
pop = 9000 + 16000 * field(5) + 9000 * smooth(counts["worship"] + counts["education"], 3) / 0.4 - 6000 * act
pop = np.clip(pop, 3000, 42000)  # people per km2, Jakarta range
# Household spend: higher along the Sudirman–Kuningan corridor, Kemang and Pondok Indah.
spend = 4.5 + 6 * (0.9 * bump(-6.215, 106.82, 2.2) + 0.7 * bump(-6.26, 106.81, 1.8)
                   + 0.6 * bump(-6.27, 106.78, 1.6)) + 2.0 * field(4)
spend = np.round(spend, 2)  # million IDR per household per month
# Footfall: people passing through, driven by transit and activity, plus noise.
footfall = 40 * catch["rail"] / max(catch["rail"].max(), 1e-9) + 30 * catch["bus"] / catch["bus"].max() + 30 * act
footfall = footfall * (0.8 + 0.4 * field(2))

FEATURES = [
    # key, label, family, array, unit
    ("pop", "Population density", "demographic", pop, "people / km²"),
    ("spend", "Household spend", "demographic", spend, "M IDR / month"),
    ("food", "Food & drink nearby", "poi", catch["food"], "places in ~800 m"),
    ("shops", "Shops nearby", "poi", catch["shops"], "places in ~800 m"),
    ("finance", "Banks & ATMs nearby", "poi", catch["finance"], "places in ~800 m"),
    ("anchors", "Schools, clinics, worship", "poi", catch["education"] + catch["health"] + catch["worship"], "places in ~800 m"),
    ("transit", "Transit access", "mobility", 3 * catch["rail"] + catch["bus"], "weighted stops"),
    ("footfall", "Footfall index", "mobility", footfall, "0–100"),
]
FAMILIES = ["demographic", "poi", "mobility"]
X = np.stack([f[3].ravel() for f in FEATURES], 1)
X = np.log1p(X - X.min(0))  # tame long tails (CBD cells)
mu, sd = X.mean(0), X.std(0)
Z = (X - mu) / sd

# ---------- 1. synthetic retailer: outlets and sales ----------
# "True" drivers of a convenience/coffee format. Saturating in density, with a
# mild penalty for heavy food competition. The model never sees these weights.
true_w = np.array([0.55, 0.45, -0.20, 0.30, 0.25, 0.20, 0.45, 0.55])
latent = Z @ true_w + 0.35 * np.tanh(Z[:, 0]) * np.tanh(Z[:, 7])
cells_live = (act.ravel() > 0.02) | (pop.ravel() > 12000)

# The retailer picked past sites by gut feel: partly good, partly random.
pick = 0.6 * latent + rng.normal(0, 1.0, latent.size)
pick[~cells_live] = -np.inf
outlets, taken = [], np.zeros(ny * nx, bool)
for c in np.argsort(-pick):
    if len(outlets) >= 72:
        break
    i, j = divmod(c, nx)
    if any((i - a) ** 2 + (j - b) ** 2 < 3.0**2 for a, b in ((o // nx, o % nx) for o in outlets)):
        continue
    outlets.append(c)
outlets = np.array(outlets)
sales = np.exp(5.6 + 0.32 * latent[outlets] + rng.normal(0, 0.18, len(outlets)))  # M IDR / month
y = np.log(sales)

# ---------- 3. learn: ridge on standardised features, CV for each signal-family combo ----------
def ridge(Zt, yt, lam=3.0):
    A = np.c_[np.ones(len(Zt)), Zt]
    R = lam * np.eye(A.shape[1])
    R[0, 0] = 0
    return np.linalg.solve(A.T @ A + R, A.T @ yt)


folds = np.arange(len(outlets)) % 5
rng.shuffle(folds)
models = {}
for r in range(1, 4):
    for fams in itertools.combinations(FAMILIES, r):
        cols = [k for k, f in enumerate(FEATURES) if f[2] in fams]
        Zo = Z[outlets][:, cols]
        pred = np.zeros(len(y))
        for f in range(5):
            tr, te = folds != f, folds == f
            b = ridge(Zo[tr], y[tr])
            pred[te] = b[0] + Zo[te] @ b[1:]
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        b = ridge(Zo, y)
        score = np.exp(b[0] + Z[:, cols] @ b[1:])
        w = np.zeros(len(FEATURES))
        w[cols] = b[1:]
        models["+".join(fams)] = {
            "r2": round(float(r2), 3),
            "w": [round(float(v), 3) for v in w],
            "score": [int(round(v)) for v in score],
            "cv": [[round(float(np.exp(a))), round(float(np.exp(p)))] for a, p in zip(y, pred)],
        }
        print(f"{'+'.join(fams):28s} CV R2 = {r2:.2f}")

# ---------- write ----------
out = {
    "meta": {
        "bbox": [W, S, E, N], "step": STEP, "nx": nx, "ny": ny,
        "features": [{"key": k, "label": l, "family": fam, "unit": u} for k, l, fam, _, u in FEATURES],
        "families": FAMILIES,
        "poiGroups": [g for g in GROUPS if g != "bus"],
        "nPoi": int(sum(c.sum() for c in counts.values())),
    },
    "live": [int(v) for v in cells_live],
    "feat": [[round(float(v), 1) for v in f[3].ravel()] for f in FEATURES],
    "outlets": [{"c": int(c), "sales": int(round(s))} for c, s in zip(outlets, sales)],
    "models": models,
    "pois": pois,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(out, separators=(",", ":")))
print(f"{nx}x{ny} cells, {len(outlets)} outlets, {len(pois)} POIs -> {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")
