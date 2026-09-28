"""Optional cloud sinks: GCS upload, BigQuery rows, Pub/Sub status.

None of these exist yet for the stockpile product. Each is switched on by an
environment flag (see config) once ``infra/setup_gcp.sh`` has created it, and
each failure is logged, never fatal — the local outputs are the product.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from typing import Dict, List, Optional

from .config import StockpileConfig


def upload_dir(storage_client, outdir: str, project_uuid: str, cfg=StockpileConfig, log=print) -> List[str]:
    if storage_client is None:
        return []
    bucket = storage_client.bucket(cfg.GCS_BUCKET_NAME)
    date = dt.date.today().isoformat()
    prefix = f"{cfg.GCS_BASE_PREFIX}{project_uuid}/{date}/"
    uris = []
    for root, _, files in os.walk(outdir):
        for f in files:
            local = os.path.join(root, f)
            rel = os.path.relpath(local, outdir).replace(os.sep, "/")
            try:
                bucket.blob(prefix + rel).upload_from_filename(local)
                uris.append(f"gs://{cfg.GCS_BUCKET_NAME}/{prefix}{rel}")
            except Exception as exc:
                log(f"GCS upload failed for {rel}: {exc}")
    log(f"Uploaded {len(uris)} files to gs://{cfg.GCS_BUCKET_NAME}/{prefix}")
    return uris


PILE_SCHEMA = [
    ("run_ts", "TIMESTAMP"), ("project_uuid", "STRING"), ("pile_id", "STRING"),
    ("commodity", "STRING"), ("commodity_source", "STRING"),
    ("occupied_area_m2", "FLOAT64"), ("volume_m3", "FLOAT64"), ("volume_sigma_m3", "FLOAT64"),
    ("tonnage_t", "FLOAT64"), ("max_height_m", "FLOAT64"), ("alert", "STRING"),
    ("max_delta_t_c", "FLOAT64"), ("persistence", "FLOAT64"), ("swir_hot_scenes", "INT64"),
    ("geometry_geojson", "STRING"),
]
SENSOR_SCHEMA = [
    ("run_ts", "TIMESTAMP"), ("project_uuid", "STRING"), ("sensor", "STRING"), ("method", "STRING"),
    ("grid_m", "FLOAT64"), ("height_rmse_m", "FLOAT64"), ("volume_wape_pct", "FLOAT64"),
    ("volume_bias_pct", "FLOAT64"), ("footprint_iou", "FLOAT64"),
]


def export_bigquery(bq, report: Dict, geojson_path: Optional[str] = None, cfg=StockpileConfig, log=print) -> bool:
    if bq is None:
        return False
    from google.cloud import bigquery
    ds_ref = f"{bq.project}.{cfg.BQ_DATASET_ID}"
    try:
        bq.get_dataset(ds_ref)
    except Exception:
        ds = bigquery.Dataset(ds_ref)
        ds.location = "EU"
        bq.create_dataset(ds, exists_ok=True)
    tables = {}
    for name, schema in (("piles", PILE_SCHEMA), ("sensor_study", SENSOR_SCHEMA)):
        t = bigquery.Table(f"{ds_ref}.{name}", schema=[bigquery.SchemaField(n, ty) for n, ty in schema])
        t.time_partitioning = bigquery.TimePartitioning(field="run_ts")
        tables[name] = bq.create_table(t, exists_ok=True)
    ts = report["generated_utc"]
    geoms = {}
    if geojson_path and os.path.exists(geojson_path):
        for f in json.load(open(geojson_path))["features"]:
            geoms[f["properties"]["pile_id"]] = json.dumps(f["geometry"])
    rows = [{"run_ts": ts, "project_uuid": report["project_uuid"], "pile_id": p["pile_id"],
             "commodity": p["commodity"], "commodity_source": p["commodity_source"],
             "occupied_area_m2": p["occupied_area_m2"], "volume_m3": p["volume_m3"],
             "volume_sigma_m3": p["volume_sigma_m3"], "tonnage_t": p["tonnage_t"],
             "max_height_m": p["max_height_m"], "alert": p.get("alert"),
             "max_delta_t_c": p.get("max_delta_t_c"), "persistence": p.get("persistence"),
             "swir_hot_scenes": p.get("swir_hot_scenes"), "geometry_geojson": geoms.get(p["pile_id"])}
            for p in report["piles"]]
    srows = []
    for s, sr in report["sensor_study"].items():
        for m, mr in sr["methods"].items():
            if mr.get("available"):
                srows.append({"run_ts": ts, "project_uuid": report["project_uuid"], "sensor": s, "method": m,
                              "grid_m": sr["grid_m"],
                              "height_rmse_m": mr["pixel_height_oof"].get("rmse_m"),
                              "volume_wape_pct": mr["pile_volume_known_footprint"].get("wape_pct"),
                              "volume_bias_pct": mr["pile_volume_known_footprint"].get("bias_pct"),
                              "footprint_iou": mr.get("footprint_iou")})
    ok = True
    for name, r in (("piles", rows), ("sensor_study", srows)):
        if r:
            errs = bq.insert_rows_json(tables[name], r)
            if errs:
                ok = False
                log(f"BigQuery insert errors ({name}): {errs[:3]}")
    return ok


def publish_status(publisher, project_uuid: str, status: str, cfg=StockpileConfig, **extra) -> None:
    if publisher is None:
        return
    topic = publisher.topic_path(cfg.GCP_PROJECT_ID, cfg.PUBSUB_TOPIC_ID)
    msg = {"project_uuid": project_uuid, "status": status, **extra}
    publisher.publish(topic, json.dumps(msg).encode()).result(timeout=30)
