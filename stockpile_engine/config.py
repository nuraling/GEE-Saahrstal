"""All tunables in one place. Nothing here touches the network."""

from __future__ import annotations

import os


def _env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


class StockpileConfig:
    # ── Cloud (same GCP project as Solain; bucket / dataset are new) ─────────
    GCP_PROJECT_ID   = os.environ.get("STOCKPILE_GCP_PROJECT", "level-sol-480011-r1")
    EE_PROJECT_ID    = os.environ.get("STOCKPILE_EE_PROJECT", "shaped-producer-482312-m0")
    GCS_BUCKET_NAME  = os.environ.get("STOCKPILE_GCS_BUCKET", "stockpile-intelligence-engine")
    GCS_BASE_PREFIX  = "stockpile/"
    PUBSUB_TOPIC_ID  = os.environ.get("STOCKPILE_PUBSUB_TOPIC", "stockpile-processing-status")
    BQ_DATASET_ID    = os.environ.get("STOCKPILE_BQ_DATASET", "stockpile_engine")
    # BigQuery and GCS do not exist yet for this product. Both are opt-in so a
    # run never fails for want of infrastructure; outputs always land locally.
    ENABLE_GCS       = _env_flag("STOCKPILE_ENABLE_GCS", False)
    ENABLE_BQ        = _env_flag("STOCKPILE_ENABLE_BQ", False)
    ENABLE_PUBSUB    = _env_flag("STOCKPILE_ENABLE_PUBSUB", False)

    # ── Client assets (Saarlouis harbour, UAV flight 01.05.2025) ─────────────
    ASSET_ROOT     = "projects/shaped-producer-482312-m0/assets"
    ASSET_ORTHO    = f"{ASSET_ROOT}/Saarlouis/Ortho_010525"
    ASSET_DSMS     = [f"{ASSET_ROOT}/Saarlouis/DSM_1_010525",
                      f"{ASSET_ROOT}/Saarlouis/DSM_2_010525",
                      f"{ASSET_ROOT}/Saarlouis/DSM_3_010525"]
    ASSET_DTM      = f"{ASSET_ROOT}/Saarlouis/DTM_1_010525"
    ASSET_PLEIADES = f"{ASSET_ROOT}/pleiades_saarfactory"
    # Not uploaded yet — set when a SPOT 6/7 scene is ingested.
    ASSET_SPOT     = os.environ.get("STOCKPILE_ASSET_SPOT", "")
    UAV_DATE       = "2025-05-01"
    # DSM_3 covers the port on the Saar: the hold-out block for validation.
    PORT_SURVEY_BLOCK = 3
    # Ground where a block has no DTM. The port is flat: one level for the
    # whole block — its low land level (percentile below, water excluded via
    # Sentinel-2 NDWI; the absolute minimum is the Saar surface, 4 m lower).
    # "morphological" = the DSM ground filter instead.
    NO_DTM_GROUND        = os.environ.get("STOCKPILE_NO_DTM_GROUND", "flat_min")
    FLAT_GROUND_PERCENTILE = 5.0

    # Building footprints for the thermal product when the client supplies
    # none: VIDA combined Google + Microsoft footprints (GEE community catalog).
    BUILDINGS_FC        = "projects/sat-io/open-datasets/VIDA_COMBINED/DEU"
    MIN_BUILDING_AREA_M2 = 30.0

    # Working CRS: ETRS89 / UTM 32N. Every source is pulled onto the same
    # metric grid so pixel (i, j) is the same patch of ground in every array.
    CRS = "EPSG:25832"

    # ── Optical sources compared in the sensor study ─────────────────────────
    # grid_m is the processing grid, native_m the sensor's own resolution.
    # A source on a finer grid than its native resolution carries no more
    # information than native — the study reports both so nobody mistakes
    # resampling for resolution.
    SENSORS = {
        "pleiades":     {"grid_m": 1.0,  "native_m": 0.3,  "kind": "asset"},
        "spot":         {"grid_m": 1.5,  "native_m": 1.5,  "kind": "asset"},
        "s2_sr_1m":     {"grid_m": 1.0,  "native_m": 10.0, "kind": "s2_sr"},
        "s2_sr_3m":     {"grid_m": 3.0,  "native_m": 10.0, "kind": "s2_sr"},
        "s2_10m":       {"grid_m": 10.0, "native_m": 10.0, "kind": "s2"},
    }
    # Band order every optical source is harmonised to.
    OPTICAL_BANDS = ["blue", "green", "red", "nir"]
    # Pléiades / SPOT DIMAP band order is B0=blue, B1=green, B2=red, B3=nir.
    # Override if the ingested asset differs (check with bandNames()).
    # pleiades_saarfactory is Pléiades Neo, 6 bands at 0.3 m, delivered as
    # b1=R b2=G b3=B b4=NIR b5=RedEdge b6=DeepBlue (checked: NDVI p99 0.68
    # with b4/b1; water rises b1→b6; orange wood chips R>G>B).
    PLEIADES_BAND_MAP = {"blue": 2, "green": 1, "red": 0, "nir": 3}
    SPOT_BAND_MAP     = {"blue": 0, "green": 1, "red": 2, "nir": 3}
    # Newer Pléiades scenes the validated Pléiades model is applied to
    # (date → asset). Override with STOCKPILE_PLEIADES_RECENT="date=asset,…".
    PLEIADES_RECENT = dict(
        kv.split("=", 1) for kv in os.environ.get(
            "STOCKPILE_PLEIADES_RECENT",
            f"2026-08-03={ASSET_ROOT}/Saarlouis/pleiades_20260803").split(",") if "=" in kv)
    # UNVERIFIED: assumes pleiades_20260803 has the same band layout as
    # pleiades_saarfactory. fetch_pleiades_recent checks NDVI and warns.
    PLEIADES_RECENT_BAND_MAP = PLEIADES_BAND_MAP
    S2_COLLECTION     = "COPERNICUS/S2_SR_HARMONIZED"
    S2_BAND_MAP       = {"blue": "B2", "green": "B3", "red": "B4", "nir": "B8",
                         "swir1": "B11", "swir2": "B12"}
    S2_MAX_CLOUD_PCT  = 20
    # Search window around the UAV flight for the comparison imagery.
    S2_WINDOW_DAYS    = 30
    # Recent scenes the validated Sentinel-2 model is applied to (best scene
    # within ± window of each date). Override with STOCKPILE_S2_RECENT.
    S2_RECENT_DATES   = [d for d in os.environ.get("STOCKPILE_S2_RECENT", "2026-08-13,2026-09-22").split(",") if d]
    S2_RECENT_WINDOW_DAYS = 10

    # Super-resolution for Sentinel-2: "geoai" (pretrained SR network, as in
    # Solain), "reference" (trained on this site against Pléiades), "bicubic".
    S2_SR_METHOD      = os.environ.get("STOCKPILE_S2_SR", "reference")

    # ── Stockpile extraction from the UAV surfaces (the ground truth) ────────
    PROC_RES_M          = 1.0     # UAV processing grid
    MIN_PILE_HEIGHT_M   = 0.5
    MAX_PILE_HEIGHT_M   = 40.0    # was a 3 m cap — which dropped every real coal pile
    MIN_PILE_AREA_M2    = 25.0
    # Buildings and rail cars have vertical walls and flat tops; bulk material
    # rests at its angle of repose (≈ 30–40°). These separate the two.
    WALL_SLOPE_DEG      = 65.0
    MAX_WALL_FRACTION   = 0.12
    FLAT_SLOPE_DEG      = 5.0
    MAX_FLAT_TOP_FRAC   = 0.55
    MAX_PILE_NDVI       = 0.30    # trees are tall and rough too
    # mean/max height: a cone is 1/3, a ridge ≈ 1/2, a wagon or roof ≈ 1.
    MAX_FILL_RATIO      = 0.75
    # Volume base: "dtm" (UAV DTM under the pile) or "toe" (plane fitted to the
    # surface along the pile boundary — standard survey practice on a pad).
    VOLUME_BASE         = "dtm"
    # Stackers, conveyor booms and cranes stand inside pile polygons. They are
    # narrow; a grey opening with a disc this wide removes them and only
    # shaves a negligible cap off a pile's crest.
    BOOM_REMOVAL_RADIUS_M = 3.0
    # A pile needs this share of its polygon covered by UAV data to be measured.
    MIN_UAV_COVERAGE    = 0.8
    # Vertical uncertainty of each surface, used for the volume error band.
    SIGMA_Z_UAV_M       = 0.05

    # Scrap metal vs bulk: UAV DSM roughness (m, detrended over 3 m) and the
    # spread of Sentinel-2 brightness inside the pile. Both must be exceeded.
    SCRAP_MIN_ROUGHNESS_M = 0.6
    SCRAP_MIN_PATCHINESS  = 0.015

    # Bulk density t/m³ (loose, stockpiled). Indicative — replace with the
    # operator's own weighbridge-calibrated factors.
    # SWIR combustion hits are pooled across piles closer than this (two S2 SWIR pixels)
    SWIR_POOL_M = 20.0

    BULK_DENSITY_T_M3 = {
        "coal": 0.85, "coke": 0.50, "iron_ore": 2.40, "limestone": 1.55,
        "sand_gravel": 1.65, "wood_chips": 0.30, "scrap_metal": 0.90,
        "slag": 1.80, "unknown": 1.00,
    }

    # ── Height model (optical → height) ──────────────────────────────────────
    CV_BLOCK_M          = 150.0   # spatial block size for cross-validation
    CV_FOLDS            = 5
    N_TRAIN_PX          = 200_000
    RAND_SEED           = 42
    GBM_MAX_ITER        = 300
    # Depth Anything V2 — the relative-depth ("-hf") checkpoints are the ones
    # transformers' pipeline can load. "Depth-Anything-V2-small" (no -hf, lower
    # case) is the original PyTorch release and fails in pipeline().
    HF_DEPTH_MODEL      = os.environ.get("STOCKPILE_DEPTH_MODEL",
                                         "depth-anything/Depth-Anything-V2-Small-hf")
    DEPTH_TILE_PX       = 518     # the model's native input size
    # Depth Anything → metres: relative depth vs height is not linear
    # (quadratic seen on the S2 SR 3 m scatter). Polynomial degree of the
    # calibration, fitted on training folds, mean matched to the UAV.
    DEPTH_CALIB_DEGREE  = 2
    DEPTH_TILE_OVERLAP  = 96
    ENABLE_DEPTH        = _env_flag("STOCKPILE_ENABLE_DEPTH", True)
    # ONNX export of Depth-Anything-V2-Small (GitHub release of
    # fabio-sim/Depth-Anything-ONNX) — used first: no torch and no
    # huggingface.co needed. Falls back to the transformers checkpoint.
    DEPTH_ONNX_URL      = ("https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/"
                           "v2.0.0/depth_anything_v2_vits.onnx")
    DEPTH_ONNX_PATH     = os.environ.get("STOCKPILE_DEPTH_ONNX", os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models",
        "depth_anything_v2_vits.onnx"))

    # ── Thermal ──────────────────────────────────────────────────────────────
    LANDSAT_COLLECTIONS = ["LANDSAT/LC08/C02/T1_L2", "LANDSAT/LC09/C02/T1_L2"]
    # QA_PIXEL: 1 dilated cloud, 2 cirrus, 3 cloud, 4 shadow, 5 snow.
    QA_REJECT_BITS      = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5)
    ECOSTRESS_COLLECTION = "NASA/ECOSTRESS/L2T_LSTE/V2"
    THERMAL_WINDOW_DAYS = 120     # scenes either side of the UAV date
    LANDSAT_TIRS_NATIVE_M = 100.0 # ST_B10 is delivered at 30 m, measured at 100 m
    LANDSAT_BAND10_UM   = 10.9
    ECOSTRESS_NATIVE_M  = 70.0
    # Background ring for per-pile temperature contrast.
    BG_RING_INNER_M     = 30.0
    BG_RING_OUTER_M     = 150.0
    # A pile is "anomalous" in a scene when its contrast exceeds k·σ of the
    # background and at least this many degrees (≈ Landsat ST uncertainty).
    # Not higher: a pile filling 30% of a 100 m pixel with its whole surface
    # 6 °C warm reads only +1.6 °C.
    ANOMALY_K_SIGMA     = 2.0
    ANOMALY_MIN_DT_C    = 1.0
    PERSISTENCE_FLAG    = 0.5     # share of scenes anomalous → persistent flag
    # A 2σ one-sided test fires by chance in ~2.5% of scenes: with 48 scenes
    # one or two "anomalies" are expected from noise alone. A pile is only
    # "watch" when its count is unlikely under that rate.
    NOISE_ANOMALY_RATE  = 0.025
    NOISE_P_VALUE       = 0.01
    # Hypothetical hot-spot size for the sub-pixel "what would it take" estimate.
    HOTSPOT_SIDE_M      = 5.0
    # SWIR high-temperature detection (Marchese et al. 2019, NHI).
    NHI_SWIR_THRESHOLD  = 0.0
    # Dark materials (coal, scrap) sit at NHI ≥ 0 all the time: fire is a
    # departure from a pixel's OWN normal, not a sign test.
    NHI_ANOMALY         = 0.10
    B12_ANOMALY_RATIO   = 1.3
    LST_CLAMP_MIN       = -40.0
    LST_CLAMP_MAX       = 80.0
