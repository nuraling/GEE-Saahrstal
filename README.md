# SOLAIN Stockyard Intelligence

Stockpile volume, material and heat monitoring for bulk-material yards: one drone
survey as the reference, then satellites between surveys.

**Portal:** the `docs/` folder is the password-protected WebGIS (GitHub Pages). All
site data in `docs/data/` is AES-256-GCM encrypted; the key is derived from the
login credentials (PBKDF2-SHA256, 250 000 rounds), which are not stored anywhere in
this repository. Client outlines and results are not included in plain form.

## What is new in this workflow
- **Foundation depth model for pile relief** — Depth Anything V2 turns a single
  satellite image into relief, used with image bands, shading and texture in a
  height model calibrated on the drone survey.
- **Erase test** — a satellite model must lose volume when a pile is digitally
  removed from the image; models that only learn the outline are rejected.
- **Heat probability** — warm-vs-surroundings counts per pile and building, by day
  and at night, tested against the sensor's noise rate.
- **Hot-spot plausibility** — pixel radiance inverted (Stefan–Boltzmann mixing) to the
  temperature a 5 × 5 m hot spot would need, compared with what the sensor resolves.
- **Pile detection by form** — angle-of-repose bulk vs walled/flat-topped buildings,
  thin wagons/conveyors and vegetation, confirmed on the orthophoto.

## Pipeline
```bash
pip install -r requirements.txt
python -m pytest -q                                   # synthetic data, no credentials
export GCP_CREDENTIALS_GEE=/path/to/service-account.json
python run_local.py --ee --stockpiles data/<site>_stockpiles.geojson \
    --buildings data/<site>_buildings.geojson --out outputs/<site> --cache outputs/cache/<site>.pkl
python port_spot.py outputs/<site>                    # SPOT/Pléiades port monitoring
python tools/export_webgis.py outputs/<site> outputs/webgis/data "<company>" "<place>"
SITE_USER=… SITE_PASS=… python webgis/build.py outputs/webgis/data docs
```
Volumes: height above the ground under each pile (own toe plane where the terrain
model is unreliable, the quay deck in the port), summed on a 1 m grid. Weights use
typical German bulk densities. Satellite volumes are reported as drone volume ×
the model's change for each pile.
