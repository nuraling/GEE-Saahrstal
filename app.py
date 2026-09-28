"""Cloud Run entrypoint — same contract shape as the Solain engine.

POST /          Pub/Sub push envelope; message data is the JSON payload below.
POST /analyze   the payload directly (synchronous; for demos and testing).
GET  /health

Payload:
  {"project_uuid": "saarlouis_2025_05",
   "stockpiles_geojson": {...},          # optional, lon/lat pile polygons (+ 'commodity')
   "buildings_geojson": {...},           # optional, lon/lat building polygons
   "aoi_geojson": {...},                 # optional, clips the UAV footprint
   "thermal_start": "2025-03-01", "thermal_end": "2025-08-31",   # optional
   "sensors": ["pleiades", "s2_10m", "s2_sr_3m"]}                # optional
"""

import base64
import json
import logging
import os
import tempfile

from flask import Flask, jsonify, request

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("stockpile.app")
app = Flask(__name__)


def _maybe_json(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return None
    return v


def validate_payload(payload):
    """(clean_payload, error) — error is a string when the payload is unusable."""
    if not isinstance(payload, dict):
        return None, "payload must be a JSON object"
    if not payload.get("project_uuid"):
        return None, "missing project_uuid"
    clean = {"project_uuid": str(payload["project_uuid"])}
    for k in ("stockpiles_geojson", "buildings_geojson", "aoi_geojson"):
        v = _maybe_json(payload.get(k))
        if payload.get(k) is not None and not isinstance(v, dict):
            return None, f"{k} is not valid GeoJSON"
        clean[k] = v
    for k in ("thermal_start", "thermal_end"):
        clean[k] = payload.get(k)
    if bool(clean["thermal_start"]) != bool(clean["thermal_end"]):
        return None, "give both thermal_start and thermal_end, or neither"
    sensors = payload.get("sensors")
    if sensors is not None and not (isinstance(sensors, list) and all(isinstance(s, str) for s in sensors)):
        return None, "sensors must be a list of names"
    clean["sensors"] = sensors
    return clean, None


def execute(p):
    from stockpile_engine import cloud
    from stockpile_engine.auth import gcp_clients, initialize_earth_engine
    from stockpile_engine.pipeline import run_pipeline
    from stockpile_engine.sources_ee import load_site

    storage_client, bq, pub = gcp_clients(log.info)
    uuid = p["project_uuid"]
    cloud.publish_status(pub, uuid, "PROCESSING")
    try:
        initialize_earth_engine(log.info)
        aoi = p.get("aoi_geojson")
        if aoi and aoi.get("type") == "Feature":
            aoi = aoi["geometry"]
        site = load_site(aoi_geojson=aoi, thermal_start=p.get("thermal_start"),
                         thermal_end=p.get("thermal_end"), sensors=p.get("sensors"), log=log.info)
        outdir = os.path.join(tempfile.gettempdir(), "stockpile", uuid)
        report = run_pipeline(site, uuid, outdir, p.get("stockpiles_geojson"), p.get("buildings_geojson"),
                              p.get("sensors"), log=log.info,
                              upload=lambda d, u: cloud.upload_dir(storage_client, d, u, log=log.info))
        cloud.export_bigquery(bq, report, os.path.join(outdir, "piles.geojson"), log=log.info)
        cloud.publish_status(pub, uuid, "SUCCESS", summary=report["summary"])
        return report
    except Exception as exc:
        log.exception("run failed")
        cloud.publish_status(pub, uuid, "FAILED", error=str(exc)[:500])
        raise


@app.get("/health")
def health():
    return {"status": "ok", "service": "stockpile-intelligence-engine"}, 200


@app.post("/analyze")
def analyze():
    p, err = validate_payload(request.get_json(silent=True))
    if err:
        return jsonify({"error": err}), 400
    try:
        report = execute(p)
    except Exception as exc:
        return jsonify({"status": "FAILED", "project_uuid": p["project_uuid"], "error": str(exc)}), 500
    return jsonify({"status": "SUCCESS", "project_uuid": p["project_uuid"], "summary": report["summary"],
                    "piles": report["piles"], "sensor_study": report["sensor_study"],
                    "files": report.get("uploaded") or report["files"]}), 200


@app.post("/")
def pubsub_push():
    env = request.get_json(silent=True)
    if not env or "message" not in env:
        return "Bad Request: missing message", 400
    try:
        payload = json.loads(base64.b64decode(env["message"]["data"]).decode("utf-8"))
    except Exception as exc:
        log.error("undecodable Pub/Sub message: %s", exc)
        return "", 204          # ACK: a retry cannot fix a broken message
    p, err = validate_payload(payload)
    if err:
        log.error("rejected payload: %s", err)
        return "", 204
    try:
        execute(p)
    except Exception:
        return "", 204          # already reported FAILED; do not redeliver a failing run forever
    return "", 204


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
