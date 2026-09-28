/*
 * ====================================================================
 * STOCKPILE INTELLIGENCE PLATFORM — v2 (Code Editor showcase)
 * Saarlouis harbour · UAV 01.05.2025 · Pléiades · Sentinel-2 · Landsat 8/9
 *
 * What changed from v1 (and why):
 *  1. Piles are no longer "objects with mean height <= 3 m" — that removed
 *     every real coal/ore pile and kept wagons. Objects are separated by
 *     FORM: piles rest at their angle of repose; roofs and wagons have
 *     vertical walls and are box-shaped (mean/max height near 1).
 *  2. Volume = Σ max(h,0) · pixel area over each drawn pile polygon
 *     (geometry2 = the client's pile inventory).
 *  3. Satellite height is scored with SPATIAL cross-validation: the yard is
 *     cut into 150 m blocks, 5 folds, every pile predicted by a model that
 *     never saw its block. v1 scored the model on its own training site.
 *  4. Error is reported as computed (WAPE, bias, RMSE). v1 divided MAPE by 10.
 *  5. Sensors side by side: Pléiades vs Sentinel-2 (10 m) vs SPOT (if an
 *     asset is set). Same folds, same piles, same metrics.
 *  6. Thermal: Landsat band 10 is MEASURED at 100 m (delivered at 30 m).
 *     Per pile and per scene: pile vs background-ring contrast, z-score,
 *     persistence across scenes. Hot-spot "equivalent" temperature uses
 *     Planck radiance in the 100 m footprint (v1: T^4 in 30 m).
 *  7. SWIR fire detector: Sentinel-2 NHI = (B12-B11)/(B12+B11) > 0 flags
 *     surface burning at 20 m — much sharper than any free thermal band.
 *
 * Imports expected (as in v1): geometry (buildings), geometry2 (piles).
 * ====================================================================
 */

// ─────────────────────────── 0. INPUTS ───────────────────────────────
var geometry = typeof geometry !== 'undefined' ? geometry :
  ee.Geometry.Polygon([[[6.75, 49.35], [6.76, 49.35], [6.76, 49.36], [6.75, 49.36]]]);
var geometry2 = typeof geometry2 !== 'undefined' ? geometry2 :
  ee.Geometry.Polygon([[[6.755, 49.355], [6.758, 49.355], [6.758, 49.358], [6.755, 49.358]]]);

var ROOT = 'projects/shaped-producer-482312-m0/assets/';
var ortho = ee.Image(ROOT + 'Saarlouis/Ortho_010525');
var DSM_1 = ee.Image(ROOT + 'Saarlouis/DSM_1_010525');
var DSM_2 = ee.Image(ROOT + 'Saarlouis/DSM_2_010525');
var DSM_3 = ee.Image(ROOT + 'Saarlouis/DSM_3_010525');
var DTM_1 = ee.Image(ROOT + 'Saarlouis/DTM_1_010525');
var PLEIADES = ee.Image(ROOT + 'pleiades_saarfactory');
var SPOT_ASSET = '';                      // set when a SPOT 6/7 scene is ingested
var UAV_DATE = ee.Date('2025-05-01');
var THERMAL_START = '2025-02-01', THERMAL_END = '2025-09-30';

var UTM = ee.Projection('EPSG:25832');
var BLOCK_M = 150, K_FOLDS = 5, SEED = 42;
var MIN_H = 0.5, WALL_SLOPE = 65, MAX_WALL_FRAC = 0.12, MAX_FILL = 0.75;
var LANDSAT_NATIVE_M = 100, HOTSPOT_M = 5, K_SIGMA = 2, MIN_DT = 1.0;
var DENSITY = {coal: 0.85, coke: 0.50, iron_ore: 2.40, limestone: 1.55, sand_gravel: 1.65,
               wood_chips: 0.30, scrap_metal: 0.90, unknown: 1.00};

function splitGeometryToFeatures(g) {
  return ee.FeatureCollection(ee.Geometry(g).geometries().map(function(x) {
    return ee.Feature(ee.Geometry(x));
  }));
}
var buildings = splitGeometryToFeatures(geometry).map(function(f) {
  return f.set('Bldg_ID', ee.String('Bldg_').cat(f.id()));
});
var piles = splitGeometryToFeatures(geometry2).map(function(f) {
  return f.set('Pile_ID', ee.String('Pile_').cat(f.id()));
});

// ─────────────────────────── 1. UI ───────────────────────────────────
ui.root.clear();
var infoPanel = ui.Panel({style: {width: '40%', padding: '15px', backgroundColor: '#F9F9F9'}});
var mapPanel = ui.Map();
mapPanel.setOptions('SATELLITE');
mapPanel.setControlVisibility({layerList: true, zoomControl: true, scaleControl: true, mapTypeControl: false});
ui.root.add(ui.SplitPanel(infoPanel, mapPanel));

var header = ui.Panel({style: {backgroundColor: '#1E293B', padding: '15px', margin: '-15px -15px 15px -15px'}});
header.add(ui.Label('STOCKPILE INTELLIGENCE', {fontWeight: 'bold', fontSize: '24px', color: '#38BDF8', backgroundColor: '#1E293B'}));
header.add(ui.Label('Volumetric & Thermal Monitoring · UAV-calibrated satellite', {fontSize: '14px', color: '#E2E8F0', backgroundColor: '#1E293B'}));
header.add(ui.Label('Client: Saarlouis Operations', {fontWeight: 'bold', fontSize: '16px', color: '#FFFFFF', backgroundColor: '#1E293B'}));
infoPanel.add(header);

function section(title) {
  infoPanel.add(ui.Label(title, {fontWeight: 'bold', fontSize: '18px', color: '#0F172A', margin: '20px 0 5px 0'}));
  var p = ui.Panel();
  p.add(ui.Label('Computing…', {color: 'gray'}));
  infoPanel.add(p);
  return p;
}
var invPanel = section('1. Stockpile inventory (UAV reference)');
var sensorPanel = section('2. Satellite vs UAV — held-out piles');
var thermalPanel = section('3. Thermal monitoring (piles)');
var bldgPanel = section('4. Building thermal');
var toolsPanel = section('5. Interactive tools');
toolsPanel.clear();

function card(txt, border, bg) {
  return ui.Label(txt, {fontSize: '12px', whiteSpace: 'pre-wrap', border: '1px solid ' + border,
                        backgroundColor: bg, padding: '6px', margin: '3px 0'});
}
function fmt(x, d) { return (x === null || x === undefined || isNaN(x)) ? '–' : Number(x).toFixed(d); }

// ─────────────────────────── 2. UAV REFERENCE ────────────────────────
var bounds = DTM_1.geometry();
mapPanel.centerObject(piles, 16);
var dsm = ee.ImageCollection([DSM_1, DSM_2, DSM_3]).mosaic().select(0);
var dtm = DTM_1.select(0);
var nDSM = dsm.subtract(dtm).rename('h');
var hPos = nDSM.max(0);
var slope = ee.Terrain.slope(dsm.setDefaultProjection(DTM_1.projection()).reproject(UTM, null, 1)).rename('slope');

// 2A. pile inventory = client polygons. Volume = Σ h⁺ · pixel area (1 m grid).
var volImg = ee.Image.cat([
  hPos.multiply(ee.Image.pixelArea()).rename('vol'),
  nDSM.gt(MIN_H).multiply(ee.Image.pixelArea()).rename('occ_area'),
  nDSM.rename('hmax')]);
var refPiles = volImg.reduceRegions({
  collection: piles,
  reducer: ee.Reducer.sum().forEachBand(volImg.select(['vol', 'occ_area']))
             .combine(ee.Reducer.max().setOutputs(['hmax']), null, false),
  scale: 1, crs: UTM, tileScale: 16
});

// 2B. automatic object detection, classified by form
var cand = nDSM.gt(MIN_H).selfMask();
var objects = cand.addBands(nDSM).reduceToVectors({
  geometry: bounds, scale: 1, crs: UTM, geometryType: 'polygon', eightConnected: true,
  labelProperty: 'label', reducer: ee.Reducer.count(), maxPixels: 1e10, tileScale: 16
}).filter(ee.Filter.gt('count', 25));   // ≥ 25 m² at 1 m
var formImg = ee.Image.cat([nDSM.rename('h'), slope.gt(WALL_SLOPE).rename('wall'), nDSM.rename('h2')]);
var objectsForm = formImg.reduceRegions({
  collection: objects,
  reducer: ee.Reducer.mean().forEachBand(formImg.select(['h', 'wall']))
             .combine(ee.Reducer.max().setOutputs(['h_max']), null, false),
  scale: 1, crs: UTM, tileScale: 16
}).map(function(f) {
  var fill = ee.Number(f.get('h')).divide(ee.Number(f.get('h_max')).max(0.01));
  var isStruct = ee.Number(f.get('wall')).gt(MAX_WALL_FRAC).or(fill.gt(MAX_FILL));
  return f.set({fill_ratio: fill, cls: ee.Algorithms.If(isStruct, 'structure', 'stockpile')});
});
var autoPiles = objectsForm.filter(ee.Filter.eq('cls', 'stockpile'));
var autoStruct = objectsForm.filter(ee.Filter.eq('cls', 'structure'));

// ─────────────────────────── 3. SENSOR STUDY ─────────────────────────
var pileMask = ee.Image(0).paint(piles, 1).And(nDSM.gt(MIN_H)).rename('pile');
var xy = ee.Image.pixelCoordinates(UTM);
var fold = xy.select('x').divide(BLOCK_M).floor().multiply(7)
  .add(xy.select('y').divide(BLOCK_M).floor().multiply(13)).mod(K_FOLDS).abs().int().rename('fold');

function pleiadesBands(img) {
  return img.select([0, 1, 2, 3], ['blue', 'green', 'red', 'nir']);
}
function features(img, scale) {
  // bands + indices + shading gradients + texture: what a single image has to say about height
  var b = img.select(['blue', 'green', 'red', 'nir']);
  var bright = b.select(['blue', 'green', 'red']).reduce(ee.Reducer.mean()).rename('bright');
  var ndvi = b.normalizedDifference(['nir', 'red']).rename('ndvi');
  var red = b.select('red').subtract(b.select('blue')).divide(b.select('red').add(b.select('blue'))).rename('redness');
  var grad = bright.gradient().rename(['gx', 'gy']);
  var k5 = ee.Kernel.circle(Math.max(5 / scale, 1.5), 'pixels');
  var k15 = ee.Kernel.circle(Math.max(15 / scale, 1.5), 'pixels');
  var k40 = ee.Kernel.circle(Math.max(40 / scale, 1.5), 'pixels');
  var tex5 = bright.reduceNeighborhood(ee.Reducer.stdDev(), k5).rename('tex5');
  var tex15 = bright.reduceNeighborhood(ee.Reducer.stdDev(), k15).rename('tex15');
  var dog15 = bright.subtract(bright.reduceNeighborhood(ee.Reducer.mean(), k15)).rename('dog15');
  var dog40 = bright.subtract(bright.reduceNeighborhood(ee.Reducer.mean(), k40)).rename('dog40');
  return ee.Image.cat([b, bright, ndvi, red, grad, tex5, tex15, dog15, dog40]).float();
}

var s2scene = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(bounds).filterDate(UAV_DATE.advance(-30, 'day'), UAV_DATE.advance(30, 'day'))
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .map(function(i) { return i.set('dt', ee.Number(i.date().difference(UAV_DATE, 'day')).abs()); })
  .sort('dt').first();
var s2 = ee.Image(s2scene).select(['B2', 'B3', 'B4', 'B8'], ['blue', 'green', 'red', 'nir']).divide(10000);

var SENSORS = [
  {name: 'Pléiades', img: pleiadesBands(PLEIADES), scale: 2},
  {name: 'Sentinel-2 10 m', img: s2, scale: 10}
];
if (SPOT_ASSET) SENSORS.push({name: 'SPOT 6/7', img: pleiadesBands(ee.Image(SPOT_ASSET)), scale: 3});

function oofHeight(sensor) {
  // 5 models; model f never sees fold f; each fold predicted by its own model
  var feat = features(sensor.img, sensor.scale);
  var names = feat.bandNames();
  var samples = feat.addBands(nDSM.max(0).rename('h')).addBands(fold).addBands(pileMask.unmask(0))
    .stratifiedSample({numPoints: 1500, classBand: 'pile', region: bounds, scale: sensor.scale,
                       seed: SEED, tileScale: 8, geometries: false});
  var preds = ee.List.sequence(0, K_FOLDS - 1).map(function(f) {
    var model = ee.Classifier.smileGradientTreeBoost(150).setOutputMode('REGRESSION')
      .train(samples.filter(ee.Filter.neq('fold', f)), 'h', names);
    return feat.classify(model).max(0).updateMask(fold.eq(ee.Number(f))).rename('h_pred');
  });
  return ee.ImageCollection.fromImages(preds).mosaic()
    .setDefaultProjection(UTM.atScale(sensor.scale)).resample('bilinear');
}

var oof = SENSORS.map(function(s) { return {name: s.name, img: oofHeight(s)}; });

var predVolImg = ee.Image.cat(oof.map(function(o, i) {
  return o.img.multiply(ee.Image.pixelArea()).rename('v' + i);
}));
var studyFc = predVolImg.reduceRegions({
  collection: refPiles, reducer: ee.Reducer.sum(), scale: 1, crs: UTM, tileScale: 16
});

function metricsFor(fc, idx) {
  var e = fc.map(function(f) {
    var a = ee.Number(f.get('vol')), p = ee.Number(f.get('v' + idx));
    return f.set({ae: p.subtract(a).abs(), se: p.subtract(a).pow(2), err: p.subtract(a)});
  });
  var s = e.reduceColumns(ee.Reducer.sum().repeat(4), ['ae', 'se', 'err', 'vol']).get('sum');
  return ee.Dictionary({sums: s, n: e.size()});
}

ee.Dictionary({
  n: refPiles.size(),
  total: refPiles.aggregate_sum('vol'),
  occ: refPiles.aggregate_sum('occ_area'),
  top: refPiles.sort('vol', false).limit(8),
  nAuto: autoPiles.size(), nStruct: autoStruct.size()
}).evaluate(function(r) {
  invPanel.clear();
  if (!r) { invPanel.add(ui.Label('No result', {color: 'red'})); return; }
  invPanel.add(card(r.n + ' piles (client polygons) · ' + Math.round(r.total).toLocaleString() + ' m³ · ' +
                    Math.round(r.occ).toLocaleString() + ' m² under material\n' +
                    'Auto-detection: ' + r.nAuto + ' pile-shaped objects, ' + r.nStruct + ' structures (walls / box-shaped)',
                    '#BAE6FD', '#F0F9FF'));
  r.top.features.forEach(function(f, i) {
    var p = f.properties;
    invPanel.add(card('#' + (i + 1) + ' ' + p.Pile_ID + ' | ' + Math.round(p.occ_area) + ' m² | max ' +
                      fmt(p.hmax, 1) + ' m | ' + Math.round(p.vol).toLocaleString() + ' m³ (±' +
                      Math.round(p.occ_area * 0.05) + ' m³ at σz 5 cm)', '#E2E8F0', '#FFFFFF'));
  });
});

ee.List(oof.map(function(o, i) { return metricsFor(studyFc, i); })).evaluate(function(ms) {
  sensorPanel.clear();
  if (!ms) { sensorPanel.add(ui.Label('Sensor study failed', {color: 'red'})); return; }
  sensorPanel.add(ui.Label('Out-of-fold volume per pile, 150 m spatial blocks, 5 folds. Lower WAPE is better.',
                           {fontSize: '11px', color: '#475569'}));
  var rows = [['Sensor', 'WAPE %', {role: 'annotation'}]];
  ms.forEach(function(m, i) {
    var s = m.sums;  // [Σ|e|, Σe², Σe, Σactual]
    var wape = 100 * s[0] / s[3], bias = 100 * s[2] / s[3], rmse = Math.sqrt(s[1] / m.n);
    rows.push([oof[i].name, wape, wape.toFixed(0) + '%']);
    sensorPanel.add(card(oof[i].name + ' → WAPE ' + wape.toFixed(1) + ' %  |  bias ' + (bias >= 0 ? '+' : '') +
                         bias.toFixed(1) + ' %  |  RMSE ' + Math.round(rmse) + ' m³  (n=' + m.n + ')',
                         '#CBD5E1', '#F8FAFC'));
  });
  sensorPanel.add(ui.Chart(rows, 'ColumnChart', {
    title: 'Pile volume error vs UAV (held-out)', legend: {position: 'none'},
    vAxis: {title: 'WAPE %', minValue: 0}, colors: ['#2a78d6'], height: 220}));
  sensorPanel.add(ui.Label('Single-image satellite height is modelled, not measured. For survey-grade ' +
                           'volumes: Pléiades tri-stereo DSM (≈1 m vertical).', {fontSize: '11px', color: '#B45309'}));
});

// ─────────────────────────── 4. THERMAL (PILES) ───────────────────────
function landsatLST(img) {
  var qa = img.select('QA_PIXEL');
  var ok = qa.bitwiseAnd((1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5)).eq(0);
  return img.select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15).rename('lst')
    .updateMask(ok).copyProperties(img, ['system:time_start']);
}
var region = piles.geometry().buffer(300);
var lstCol = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'))
  .filterBounds(region).filterDate(THERMAL_START, THERMAL_END).map(landsatLST)
  .map(function(i) {
    var clear = i.mask().reduceRegion(ee.Reducer.mean(), region, 30).values().get(0);
    return i.set('clear', clear);
  }).filter(ee.Filter.gt('clear', 0.6));

// background: not piles, not buildings, not water, not trees
var s2comp = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(region)
  .filterDate(THERMAL_START, THERMAL_END).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)).median();
var water = s2comp.normalizedDifference(['B3', 'B8']).gt(0.1);
var veg = s2comp.normalizedDifference(['B8', 'B4']).gt(0.4);
var exclude = ee.Image(0).paint(piles.map(function(f) { return f.buffer(15); }), 1)
  .paint(buildings, 1).Or(water).Or(veg);
var rings = piles.map(function(f) {
  return ee.Feature(f.geometry().buffer(150).difference(f.geometry().buffer(30)), {Pile_ID: f.get('Pile_ID')});
});

var perScene = lstCol.map(function(img) {
  var t = img.select('lst');
  var obj = t.reduceRegions({collection: piles, reducer: ee.Reducer.max().setOutputs(['t_obj']), scale: 30, crs: UTM});
  var bg = t.updateMask(exclude.not()).reduceRegions({
    collection: rings, reducer: ee.Reducer.median().setOutputs(['t_bg'])
      .combine(ee.Reducer.stdDev().setOutputs(['s_bg']), null, true), scale: 30, crs: UTM});
  var joined = ee.Join.inner().apply(obj, bg, ee.Filter.equals({leftField: 'Pile_ID', rightField: 'Pile_ID'}));
  return joined.map(function(j) {
    var a = ee.Feature(j.get('primary')), b = ee.Feature(j.get('secondary'));
    var dT = ee.Number(a.get('t_obj')).subtract(b.get('t_bg'));
    var s = ee.Number(b.get('s_bg')).max(0.1);
    return ee.Feature(null, {Pile_ID: a.get('Pile_ID'), date: img.date().format('YYYY-MM-dd'),
      t: img.date().millis(), dT: dT, z: dT.divide(s), t_bg: b.get('t_bg'), s_bg: s,
      anom: dT.gt(MIN_DT).and(dT.divide(s).gt(K_SIGMA))});
  });
}).flatten().filter(ee.Filter.notNull(['dT']));

// SWIR high-temperature detector (Sentinel-2, 20 m)
var nhiCount = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(region)
  .filterDate(THERMAL_START, THERMAL_END).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
  .map(function(i) {
    var b11 = i.select('B11').divide(10000), b12 = i.select('B12').divide(10000);
    return b12.subtract(b11).divide(b12.add(b11)).gt(0).And(b12.gt(0.02)).rename('hot');
  }).sum().rename('hot_scenes');
var swirPerPile = nhiCount.reduceRegions({collection: piles, reducer: ee.Reducer.max().setOutputs(['swir_hot']), scale: 20, crs: UTM});

// Planck mixing, 10.9 µm — the temperature a 5×5 m spot needs to explain dT
function planck(tK) { return 1.191042e8 / (Math.pow(10.9, 5) * (Math.exp(1.4387752e4 / (10.9 * tK)) - 1)); }
function invPlanck(L) { return 1.4387752e4 / (10.9 * Math.log(1.191042e8 / (Math.pow(10.9, 5) * L) + 1)); }
function median(a) {
  var s = a.slice().sort(function(x, y) { return x - y; });
  return s[Math.floor(s.length / 2)];
}
function hotspotC(tObs, tBg) {
  var phi = HOTSPOT_M * HOTSPOT_M / (LANDSAT_NATIVE_M * LANDSAT_NATIVE_M);
  var L = (planck(tObs + 273.15) - (1 - phi) * planck(tBg + 273.15)) / phi;
  return L > 0 ? invPlanck(L) - 273.15 : NaN;
}

ee.Dictionary({
  rows: perScene.reduceColumns(ee.Reducer.toList(5), ['Pile_ID', 'dT', 'z', 'anom', 't_bg']).get('list'),
  swir: swirPerPile.reduceColumns(ee.Reducer.toList(2), ['Pile_ID', 'swir_hot']).get('list'),
  nScenes: lstCol.size()
}).evaluate(function(r) {
  thermalPanel.clear();
  if (!r) { thermalPanel.add(ui.Label('Thermal failed', {color: 'red'})); return; }
  var by = {};
  r.rows.forEach(function(x) {
    var o = by[x[0]] = by[x[0]] || {n: 0, anom: 0, maxdT: -99, maxz: -99, dTs: [], bg: []};
    o.n++; o.anom += x[3] ? 1 : 0; o.maxdT = Math.max(o.maxdT, x[1]); o.maxz = Math.max(o.maxz, x[2]);
    o.dTs.push(x[1]); o.bg.push(x[4]);
  });
  var sw = {};
  r.swir.forEach(function(x) { sw[x[0]] = x[1] || 0; });
  thermalPanel.add(ui.Label(r.nScenes + ' clear Landsat 8/9 scenes · contrast vs 30–150 m ring · ' +
                            'anomalous = ΔT > ' + MIN_DT + ' °C and > ' + K_SIGMA + 'σ', {fontSize: '11px', color: '#475569'}));
  var list = Object.keys(by).map(function(k) {
    var o = by[k], persist = o.anom / o.n, s = sw[k] || 0;
    var level = (s >= 2 || (s >= 1 && persist >= 0.5)) ? 'CRITICAL' : (persist >= 0.5 && o.anom >= 2) ? 'WARNING' :
                (s === 1 || o.anom >= 1) ? 'WATCH' : 'none';
    return {id: k, o: o, persist: persist, swir: s, level: level};
  }).sort(function(a, b) { return b.o.maxz - a.o.maxz; });
  var colors = {CRITICAL: ['#d03b3b', '#FEF2F2'], WARNING: ['#ec835a', '#FFF7ED'], WATCH: ['#fab219', '#FEFCE8'], none: ['#CBD5E1', '#FFFFFF']};
  list.slice(0, 10).forEach(function(x) {
    var med = median(x.o.bg);
    var hs = x.o.maxdT > MIN_DT ? hotspotC(med + x.o.maxdT, med) : NaN;
    thermalPanel.add(card('● ' + x.level + '  ' + x.id + '\n' +
      'max ΔT ' + fmt(x.o.maxdT, 1) + ' °C (z ' + fmt(x.o.maxz, 1) + ') · anomalous in ' + x.o.anom + '/' + x.o.n +
      ' scenes · SWIR fire flags: ' + x.swir + '\n' +
      (isNaN(hs) ? '' : 'equivalent 5×5 m hot-spot: ' + fmt(hs, 0) + ' °C (model, 100 m footprint)'),
      colors[x.level][0], colors[x.level][1]));
  });
  var bgAll = [].concat.apply([], list.map(function(x) { return x.o.bg; }));
  var med = bgAll.length ? median(bgAll) : 20;
  thermalPanel.add(ui.Label('Landsat can flag a 5×5 m hot-spot only above ≈ ' + fmt(hotspotC(med + 2 * 0.8, med), 0) +
    ' °C. Pile-scale early warning needs night ECOSTRESS or commercial 3–10 m thermal.', {fontSize: '11px', color: '#B45309'}));
});

var topPile = perScene.sort('z', false).first();
ee.Feature(topPile).get('Pile_ID').evaluate(function(id) {
  if (!id) return;
  thermalPanel.add(ui.Chart.feature.byFeature(perScene.filter(ee.Filter.eq('Pile_ID', id)).sort('t'), 'date', ['dT'])
    .setChartType('LineChart').setOptions({title: 'ΔT vs background — ' + id, vAxis: {title: '°C'},
      colors: ['#eb6834'], lineWidth: 2, pointSize: 5, height: 200, legend: {position: 'none'}}));
});

// ─────────────────────────── 5. BUILDINGS ─────────────────────────────
var lstMedian = lstCol.select('lst').median().rename('LST');
// modelled fine-scale LST (context layer): optical → LST at 30 m, residual added back
var feat30 = features(pleiadesBands(PLEIADES), 2);
var lstTrain = feat30.reduceResolution(ee.Reducer.mean(), true, 1024).reproject(UTM, null, 30)
  .addBands(lstMedian).sample({region: region, scale: 30, numPixels: 3000, seed: SEED, tileScale: 8});
var lstModel = ee.Classifier.smileRandomForest(60).setOutputMode('REGRESSION')
  .train(lstTrain, 'LST', feat30.bandNames());
var lstPredFine = feat30.classify(lstModel).rename('LST');
var resid = lstMedian.subtract(lstPredFine.reduceResolution(ee.Reducer.mean(), true, 1024).reproject(UTM, null, 30));
var lstDown = lstPredFine.add(resid.resample('bilinear')).rename('LST_down');

var bldgT = ee.Image.cat([lstMedian, lstDown]).reduceRegions({
  collection: buildings, reducer: ee.Reducer.max(), scale: 3, crs: UTM, tileScale: 16
}).filter(ee.Filter.notNull(['LST']));
var p90b = ee.Number(bldgT.reduceColumns(ee.Reducer.percentile([90]), ['LST']).get('p90'));
var top10b = bldgT.filter(ee.Filter.gte('LST', p90b));
ee.Dictionary({top: bldgT.sort('LST', false).limit(10), p90: p90b}).evaluate(function(r) {
  bldgPanel.clear();
  if (!r || r.p90 === null) { bldgPanel.add(ui.Label('No building thermal data', {color: 'red'})); return; }
  bldgPanel.add(ui.Label('Top 10% threshold (Landsat composite max): ' + fmt(r.p90, 1) + ' °C', {fontSize: '12px'}));
  r.top.features.forEach(function(f, i) {
    var p = f.properties;
    bldgPanel.add(card('#' + (i + 1) + ' ' + p.Bldg_ID + ' | measured (30 m) ' + fmt(p.LST, 1) + ' °C | modelled 3 m ' +
                       fmt(p.LST_down, 1) + ' °C', '#FECACA', '#FEF2F2'));
  });
});

// ─────────────────────────── 6. MAP LAYERS ────────────────────────────
var thermalPal = ['#000080', '#0000FF', '#00FFFF', '#80FF00', '#FFFF00', '#FF8000', '#FF0000', '#800000'];
var heightPal = ['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d3268'];
mapPanel.addLayer(ortho, {min: 0, max: 255}, 'UAV ortho', true);
var layers = {
  'UAV nDSM (height)': ui.Map.Layer(nDSM.updateMask(nDSM.gt(MIN_H)), {min: 0, max: 15, palette: heightPal}, 'UAV nDSM', true),
  'Pile volume (client piles)': ui.Map.Layer(ee.Image().float().paint(refPiles, 'vol'), {min: 0, max: 20000, palette: heightPal, opacity: 0.7}, 'Pile volume', false),
  'Satellite height — Pléiades (held-out)': ui.Map.Layer(oof[0].img.updateMask(pileMask), {min: 0, max: 15, palette: heightPal}, 'Pléiades height', false),
  'Satellite height — Sentinel-2 (held-out)': ui.Map.Layer(oof[1].img.updateMask(pileMask), {min: 0, max: 15, palette: heightPal}, 'S2 height', false),
  'Auto-detected piles vs structures': ui.Map.Layer(autoPiles.style({color: '1baf7a', fillColor: '1baf7a44'}).blend(autoStruct.style({color: 'e34948', fillColor: 'e3494844'})), {}, 'Auto objects', false),
  'Landsat LST median (30 m, measured)': ui.Map.Layer(lstMedian, {min: 10, max: 35, palette: thermalPal, opacity: 0.7}, 'LST 30 m', false),
  'LST modelled 3 m (context)': ui.Map.Layer(lstDown, {min: 10, max: 35, palette: thermalPal, opacity: 0.7}, 'LST 3 m', false),
  'SWIR fire flags (scenes)': ui.Map.Layer(nhiCount.selfMask(), {min: 1, max: 5, palette: ['#fab219', '#d03b3b']}, 'SWIR hot', false),
  'Top 10% hottest buildings': ui.Map.Layer(top10b.style({color: 'FF0000', fillColor: 'FF000055', width: 2}), {}, 'Top 10% bldgs', false)
};
Object.keys(layers).forEach(function(k) { mapPanel.add(layers[k]); });
mapPanel.addLayer(piles.style({color: 'FFFFFF', fillColor: '00000000', width: 1}), {}, 'Pile polygons', true);
mapPanel.addLayer(buildings.style({color: '000000', fillColor: '00000000', width: 1}), {}, 'Buildings', true);

var ctrl = ui.Panel({style: {position: 'bottom-right', padding: '8px', width: '300px'}});
ctrl.add(ui.Label('Analysis overlay', {fontWeight: 'bold'}));
ctrl.add(ui.Select({items: Object.keys(layers), value: 'UAV nDSM (height)', style: {width: '100%'},
  onChange: function(sel) { Object.keys(layers).forEach(function(k) { layers[k].setShown(k === sel); }); }}));
mapPanel.add(ctrl);

// ─────────────────────────── 7. TOOLS ─────────────────────────────────
var btns = ui.Panel({layout: ui.Panel.Layout.flow('horizontal')});
var out = ui.Panel({style: {height: '260px', border: '1px solid #ddd', padding: '5px'}});
out.add(ui.Label('Select a tool…'));
toolsPanel.add(btns); toolsPanel.add(out);
var dt = mapPanel.drawingTools();
dt.setShown(false);
while (dt.layers().length() > 0) dt.layers().remove(dt.layers().get(0));
dt.layers().add(ui.Map.GeometryLayer({geometries: null, name: 'tool', color: 'cyan'}));
var mode = null;

function analyse() {
  var g = dt.layers().get(0).getEeObject();
  out.clear();
  if (mode === 'line') {
    var prof = nDSM.rename('UAV').addBands(oof[0].img.rename('Pleiades')).addBands(oof[1].img.rename('Sentinel2'));
    var pts = prof.sample({region: g, scale: 1, geometries: true, dropNulls: true});
    out.add(ui.Chart.feature.byFeature(pts, 'system:index', ['UAV', 'Pleiades', 'Sentinel2'])
      .setChartType('LineChart').setOptions({title: 'Height profile', vAxis: {title: 'm'},
        colors: ['#0b0b0b', '#2a78d6', '#eb6834'], lineWidth: 2, height: 230}));
  } else if (mode === 'vol') {
    var v = ee.Image.cat([hPos, oof[0].img, oof[1].img]).multiply(ee.Image.pixelArea())
      .rename(['uav', 'ple', 's2']).reduceRegion({reducer: ee.Reducer.sum(), geometry: g, scale: 1, crs: UTM, maxPixels: 1e9});
    ee.Dictionary(v).set('area', g.area(1)).evaluate(function(r) {
      out.clear();
      out.add(ui.Label('Area ' + Math.round(r.area).toLocaleString() + ' m²', {fontWeight: 'bold'}));
      out.add(ui.Label('UAV volume: ' + Math.round(r.uav).toLocaleString() + ' m³', {fontSize: '15px'}));
      out.add(ui.Label('Pléiades (held-out model): ' + Math.round(r.ple).toLocaleString() + ' m³', {color: '#2a78d6'}));
      out.add(ui.Label('Sentinel-2 (held-out model): ' + Math.round(r.s2).toLocaleString() + ' m³', {color: '#eb6834'}));
      Object.keys(DENSITY).forEach(function(k) {
        if (k !== 'unknown') out.add(ui.Label('  if ' + k + ': ' + Math.round(r.uav * DENSITY[k]).toLocaleString() + ' t', {fontSize: '11px', color: '#475569'}));
      });
    });
  }
}
dt.onDraw(ui.util.debounce(analyse, 500));
dt.onEdit(ui.util.debounce(analyse, 500));
function reset() {
  dt.stop();
  var l = dt.layers().get(0);
  while (l.geometries().length() > 0) l.geometries().remove(l.geometries().get(0));
  out.clear();
}
btns.add(ui.Button('📉 Profile', function() { reset(); mode = 'line'; dt.setShape('line'); dt.draw(); }));
btns.add(ui.Button('⛰️ Volume', function() { reset(); mode = 'vol'; dt.setShape('polygon'); dt.draw(); }));
btns.add(ui.Button('❌ Clear', function() { reset(); mode = null; out.add(ui.Label('Cleared.')); }));

// ─────────────────────────── 8. EXPORTS (optional) ───────────────────
// Export.table.toDrive({collection: studyFc, description: 'stockpile_volumes_sensor_study', fileFormat: 'CSV'});
// Export.table.toDrive({collection: perScene, description: 'stockpile_thermal_per_scene', fileFormat: 'CSV'});
