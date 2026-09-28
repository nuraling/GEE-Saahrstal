"""Credentials — the same service accounts the Solain Thermal Engine uses.

Resolution order, first hit wins:

1. ``GCP_CREDENTIALS_GEE`` / ``GCP_CREDENTIALS_GCS`` — paths to service-account
   JSON files. Same variable names as Solain, so one Cloud Run secret mount
   serves both engines. This is the production path.
2. The Solain engine's own fallback accounts, read from its source file
   (``SOLAIN_ENGINE_PATH``, default: a sibling ``solain-thermal-engine``
   checkout). Read with ``ast`` so Solain's heavy imports are not pulled in.
   No key material is copied into this repository.
3. Application Default Credentials (``gcloud auth application-default login``)
   — the convenient path on a laptop.
"""

from __future__ import annotations

import ast
import json
import os
from typing import Optional, Tuple

from .config import StockpileConfig

EE_SCOPES = ["https://www.googleapis.com/auth/earthengine",
             "https://www.googleapis.com/auth/cloud-platform"]
GCP_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

_SOLAIN_FILE = "solain_pipeline_v7_2.py"


def _json_from_env(var: str) -> Optional[dict]:
    path = os.environ.get(var)
    if path and os.path.isfile(path):
        with open(path) as fh:
            return json.load(fh)
    return None


def _default_solain_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "solain-thermal-engine"))


def solain_legacy_credentials(solain_dir: Optional[str] = None) -> Tuple[Optional[dict], Optional[dict]]:
    """(gee, gcs) service-account dicts from SolainAuthManager, or (None, None).

    Parses the class body statically; nothing in the Solain module executes.
    """
    solain_dir = solain_dir or os.environ.get("SOLAIN_ENGINE_PATH") or _default_solain_path()
    src_path = os.path.join(solain_dir, _SOLAIN_FILE)
    if not os.path.isfile(src_path):
        return None, None
    with open(src_path) as fh:
        tree = ast.parse(fh.read(), filename=src_path)
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "SolainAuthManager":
            for stmt in node.body:
                if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                        and isinstance(stmt.targets[0], ast.Name)
                        and stmt.targets[0].id in ("_LEGACY_GEE", "_LEGACY_GCS")):
                    try:
                        found[stmt.targets[0].id] = ast.literal_eval(stmt.value)
                    except ValueError:
                        pass
    return found.get("_LEGACY_GEE"), found.get("_LEGACY_GCS")


def resolve_service_accounts() -> Tuple[Optional[dict], Optional[dict], str]:
    """(gee_sa, gcs_sa, source) — either SA may be None (→ ADC)."""
    gee = _json_from_env("GCP_CREDENTIALS_GEE")
    gcs = _json_from_env("GCP_CREDENTIALS_GCS")
    if gee:
        return gee, gcs or gee, "env"
    s_gee, s_gcs = solain_legacy_credentials()
    if s_gee:
        return s_gee, gcs or s_gcs, "solain"
    return None, gcs, "adc"


def initialize_earth_engine(log=print) -> str:
    """Initialise the ee client. Returns which credential source was used."""
    import ee

    gee_sa, _, source = resolve_service_accounts()
    if gee_sa:
        from google.oauth2 import service_account
        creds = service_account.Credentials.from_service_account_info(gee_sa, scopes=EE_SCOPES)
        ee.Initialize(creds, project=gee_sa.get("project_id", StockpileConfig.EE_PROJECT_ID))
        log(f"Earth Engine initialised with service account {gee_sa.get('client_email')} ({source}).")
    else:
        ee.Initialize(project=StockpileConfig.EE_PROJECT_ID)
        log("Earth Engine initialised with application-default credentials.")
    return source


def gcp_clients(log=print):
    """(storage_client, bigquery_client, publisher) — each None if disabled/unavailable."""
    cfg = StockpileConfig
    _, gcs_sa, _ = resolve_service_accounts()
    creds, project = None, cfg.GCP_PROJECT_ID
    if gcs_sa:
        from google.oauth2 import service_account
        creds = service_account.Credentials.from_service_account_info(gcs_sa, scopes=GCP_SCOPES)
        project = gcs_sa.get("project_id", project)
    storage_client = bq_client = publisher = None
    if cfg.ENABLE_GCS:
        try:
            from google.cloud import storage
            storage_client = storage.Client(credentials=creds, project=project)
        except Exception as exc:  # pragma: no cover - depends on cloud
            log(f"GCS unavailable: {exc}")
    if cfg.ENABLE_BQ:
        try:
            from google.cloud import bigquery
            bq_client = bigquery.Client(credentials=creds, project=project)
        except Exception as exc:  # pragma: no cover
            log(f"BigQuery unavailable: {exc}")
    if cfg.ENABLE_PUBSUB:
        try:
            from google.cloud import pubsub_v1
            publisher = pubsub_v1.PublisherClient(credentials=creds)
        except Exception as exc:  # pragma: no cover
            log(f"Pub/Sub unavailable: {exc}")
    return storage_client, bq_client, publisher
