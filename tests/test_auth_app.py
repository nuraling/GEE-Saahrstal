import json
import os

from stockpile_engine import auth


def test_solain_fallback_is_read_statically(tmp_path):
    (tmp_path / "solain_pipeline_v7_2.py").write_text(
        "import this_module_does_not_exist\n"
        "class SolainAuthManager:\n"
        "    _LEGACY_GEE = {'type': 'service_account', 'project_id': 'p-gee', 'client_email': 'a@b'}\n"
        "    _LEGACY_GCS = {'type': 'service_account', 'project_id': 'p-gcs'}\n")
    gee, gcs = auth.solain_legacy_credentials(str(tmp_path))
    assert gee["project_id"] == "p-gee" and gcs["project_id"] == "p-gcs"


def test_env_credentials_win(tmp_path, monkeypatch):
    f = tmp_path / "sa.json"
    f.write_text(json.dumps({"type": "service_account", "project_id": "from-env"}))
    monkeypatch.setenv("GCP_CREDENTIALS_GEE", str(f))
    gee, gcs, src = auth.resolve_service_accounts()
    assert src == "env" and gee["project_id"] == "from-env"


def test_missing_solain_checkout_is_not_an_error(tmp_path):
    assert auth.solain_legacy_credentials(str(tmp_path)) == (None, None)


def test_payload_validation():
    import app
    ok, err = app.validate_payload({"project_uuid": "x", "stockpiles_geojson": json.dumps(
        {"type": "FeatureCollection", "features": []})})
    assert err is None and isinstance(ok["stockpiles_geojson"], dict)
    assert app.validate_payload({})[1] == "missing project_uuid"
    assert app.validate_payload({"project_uuid": "x", "thermal_start": "2025-01-01"})[1]
    assert app.validate_payload({"project_uuid": "x", "stockpiles_geojson": "not json"})[1]


def test_health():
    import app
    r = app.app.test_client().get("/health")
    assert r.status_code == 200
