# What data raises accuracy

The showcase runs on what exists today: one UAV survey (DSM ×3, DTM, ortho), one
Pléiades scene, and the free archives (Sentinel-2, Landsat 8/9, ECOSTRESS).
This page covers what each extra dataset would add, so the post-approval data
request can be specific.

## Volume: what each source can measure

| Source | Pixel | Height? | Realistic use for stockpiles | Expected volume quality* |
|---|---|---|---|---|
| UAV photogrammetry (+ GCPs) | 2–5 cm | **Measured** (σz ≈ 2–5 cm) | Reference, audit-grade | ±1–3 % |
| **Pléiades / Pléiades Neo tri-stereo** | 0.5 / 0.3 m | **Measured** (DSM ≈ 1 m, σz ≈ 0.5–1.5 m) | Monthly satellite volumes without flying | ±5–10 % on large piles (> 5 m), worse on low piles |
| SPOT 6/7 stereo | 1.5 m | Measured (σz ≈ 2–3 m) | Big piles, trends | ±10–25 % |
| Pléiades mono (what we have) | 0.5–2 m | Modelled from shading and texture | Footprint and area (good), height (site-trained) | Measured by the sensor study, not assumed |
| Depth Anything on Pléiades | same | Relative depth, calibrated to UAV | Adds a shape prior to the model | Measured by the sensor study |
| Sentinel-2 SR 1–3 m | 10 m native | Modelled | Footprint of piles ≳ 30 × 30 m | Measured by the sensor study. Cannot resolve gaps < 10 m |
| Sentinel-2 10 m | 10 m | Modelled | Change detection, large piles, weekly revisit | Measured by the sensor study |

\* Typical ranges from the photogrammetry literature and survey practice, given
to set expectations. The sensor study replaces them with measured numbers for
this yard.

Single-image methods (spectral RF, gradient boosting, Depth Anything) learn how
piles *look* on this yard. They can do well on piles that resemble the training
piles, and they fail on new shapes, new materials, or different sun angles. The
spatial cross-validation shows how well they generalise within the yard.
Generalising across dates needs more than one UAV date.

## Data request, in priority order

1. **Pléiades (Neo) tri-stereo over the yard**, as close as possible to a UAV
   flight date. This is the single biggest step. Satellite volumes become
   photogrammetric measurements, and the UAV validates them instead of
   training them.
2. **Repeat UAV flights (≥ 3 dates)** with ground control, at different stock
   levels. This lets the height model and the stereo DSM be checked across
   dates, not on one snapshot.
3. **Bare-pad DTM**: a flight when a pad is empty, or the as-built survey.
   Volumes are then surface − pad, not surface − interpolated ground.
4. **Operator stock book**: per-pile tonnage by date, plus measured bulk
   densities per commodity. Tonnage stops being volume × a textbook density,
   and volume can be validated against reality.
5. **Pile polygons with commodity labels.** These replace the colour rule, and
   also train it for any unlabelled piles.
6. **SPOT 6/7** only if cost rules out Pléiades stereo, or for a denser time
   series between Pléiades acquisitions.

## Thermal: what each source can detect

A self-heating spot of 5 × 5 m at 90 °C inside a Landsat pixel raises the pixel
by about **0.2 °C**. That is below the noise. The engine reports, per pile,
the **minimum hot-spot temperature each sensor can flag**. Values at a 20 °C
background with 0.8 °C scatter and a 2σ threshold (quieter backgrounds lower them):

| Sensor | Footprint | Min. detectable 5 × 5 m hot-spot | Revisit / time |
|---|---|---|---|
| Landsat 8/9 TIRS | 100 m | ≈ 320 °C | 8 days combined, ~10:30 local |
| ECOSTRESS | 70 m | ≈ 200 °C | irregular, **includes night** |
| Commercial HR thermal / UAV radiometric thermal | ≤ 5 m | ≈ 22 °C (the spot fills the pixel) | on demand |
| Sentinel-2 SWIR (NHI) | 20 m | surface combustion only (≳ 300 °C) | 5 days |

The engine computes these numbers with `thermal.min_detectable_hotspot`. The
per-pile values are in the report.

What this means:

- With the free data, the engine can reliably see **broad** warming of a pile's
  surface, and **open burning** (SWIR). It cannot see an early, small smoulder.
- Early warning of spontaneous combustion needs **night-time** and
  **pile-scale** thermal. The practical options:
  - **UAV radiometric thermal flights** at dawn. This is cheapest, and the
    best fit for the showcase.
  - **Commercial high-resolution thermal satellites.** Check current
    availability and resolution with the providers.
  - **In-pile probe temperatures** from the operator's own checks. These
    calibrate what a surface anomaly means.

## Ground truth to collect with any new data

- GCPs / checkpoints on the pads (vertical accuracy of every surface)
- Acquisition time and sun angle for every optical scene (the shading features depend on them)
- For thermal: pile-surface IR gun readings at overpass time on 3–5 piles
