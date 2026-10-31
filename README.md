# Retail site selection: choosing where to open the next store

**Live demo: https://dhaneswaramandrasa.github.io/retail-site-selection/**

An interactive, open-data version of a location-intelligence project I did at Bhumi Varta (2020–2021): learn what makes an existing store do well from the area around it, score every cell of the map, and shortlist where the next store should go. The real model guided a retail network from under 100 to 200+ outlets. Case study: [Choosing where to open the next store](https://dhaneswara-mandrasa.vercel.app/project-retail-sites).

The client, its stores and its sales are confidential, so this demo uses a **synthetic retailer on top of real OpenStreetMap places** in South and Central Jakarta. The method is the same; the numbers are not the client's.

## What it shows

Five steps, matching the case study:

| Step | What happens |
| --- | --- |
| 1 Input | 72 existing outlets of a synthetic retailer and their monthly sales |
| 2 Describe | Every ~400 m cell gets 8 attributes in 3 families, measured over a ~800 m catchment: **demographic** (population density, household spend), **places / POI** (food & drink, shops, banks & ATMs, schools/clinics/worship) and **mobility** (transit access, footfall) |
| 3 Learn | A ridge regression predicts outlet sales from its cell's attributes. Toggle signal families to see what each adds; fit is 5-fold cross-validated R² on held-out outlets (0.43 with demographics alone → 0.81 with all three) |
| 4 Score | The model predicts sales for all 2,601 cells, shown as a 3D surface |
| 5 Decide | Greedy shortlist of the top cells, each at least a chosen distance from existing outlets and from each other to avoid cannibalisation |

The production model used gradient boosting and richer data; this demo uses a linear model so the learned weights are readable.

## Data

| Layer | Source | Real or synthetic |
| --- | --- | --- |
| Restaurants, cafés, shops, banks, ATMs, clinics, schools, places of worship | OpenStreetMap via Overpass, `data/raw/` | Real (© OpenStreetMap contributors, ODbL) |
| Rail stations and bus stops (transit access) | OpenStreetMap | Real |
| Population density, household spend | Generated in `scripts/build_data.py`, shaped to Jakarta's geography | Synthetic |
| Footfall index | Built from real transit and POI density plus noise | Synthetic |
| Outlets and their sales | Generated from hidden "true" drivers plus noise; the model never sees them | Synthetic |

The Overpass query (bbox −6.32, 106.74, −6.14, 106.92), saved to `data/raw/osm_pois_south_central_jakarta.json`:

```
[out:json][timeout:60][bbox:-6.32,106.74,-6.14,106.92];
(node["amenity"~"^(cafe|restaurant|fast_food|school|university|bank|atm|hospital|clinic|place_of_worship|marketplace)$"];
 node["shop"]; node["railway"~"^(station|halt)$"]; node["highway"="bus_stop"];);
out;
```

## Run it

```bash
pip install -r requirements.txt     # numpy only
python scripts/build_data.py        # writes web/data/sites.json
python -m http.server -d web 8000   # open http://localhost:8000
```

`web/index.html` is a single static page (MapLibre GL + deck.gl from a CDN, basemap from OpenFreeMap). `?step=0`–`4` opens a given step and `?embed` hides the panels. GitHub Actions deploys `web/` to Pages on every push to `main`.

## Layout

```
data/raw/            OSM POIs (Overpass output)
scripts/build_data.py  grid, features, synthetic retailer, ridge models, JSON export
web/index.html       the demo
web/data/sites.json  model output read by the demo
```

MIT licensed. Map data © OpenStreetMap contributors.
