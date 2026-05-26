<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { navigate } from '../../lib/stores';
  import MapCanvas from './MapCanvas.svelte';
  import FitControls from './FitControls.svelte';
  import MapControls from './MapControls.svelte';
  import MapProgressPanel from './MapProgressPanel.svelte';
  import ClassificationEditor from './ClassificationEditor.svelte';
  import MapSpectrumPanel from './MapSpectrumPanel.svelte';
  import { MapWebSocket } from '../../lib/mapWebSocket';
  import {
    mapPointSet,
    mapLayers,
    mapFitJob,
    mapLogEntries,
    mapDisplayMode,
    mapGeometryMode,
    mapOverlayOpacity,
    mapShowPointPositions,
    mapSavedTransform,
    mapFittedResultsCache,
    mapActiveProfile,
    mapScanId,
    resetMapState,
  } from '../../lib/stores/mapStore';
  import type {
    PointSet,
    ScalarLayer,
    LayerValue,
    WSPointFitted,
    ClassificationRule,
    ClassificationProfile,
  } from '../../lib/types/map';
  import {
    AuthRequiredError,
    fetchAciImage,
    getMapData,
    getMapLayers,
    getScan,
    startMapFit,
  } from '../../lib/api';

  export let scanId: string;

  let mapCanvas: MapCanvas;
  let ws: MapWebSocket | null = null;
  let aciImage: HTMLImageElement | null = null;
  let colorizedAciImage: HTMLImageElement | null = null;
  let mapLoadGeneration = 0; // stale-load guard for rapid scan switches
  let mapAuthRequired = false;
  let aciLoading = false;
  let brightness = 1.0;
  let contrast = 1.0;
  let showClassEditor = false;
  let scanLabel = '';
  let loadError = '';

  // Deferred layer loading: store available_layers metadata from initial fetch,
  // but only load actual scalar values when the user selects a display mode.
  let availableLayers: Record<string, Record<string, { n_detections: number; classes: string[] }>> | null = null;
  let layerDataLoaded = false;

  // --- Layer ownership model (spec §2.1) ---
  type LayerSource = 'init' | 'live_fit' | 'profile';
  let layerSource: LayerSource = 'init';
  let sourceGeneration = 0;

  // --- Fitting result caches (synced with stores for persistence) ---
  let fittedResultsCache: WSPointFitted[] = [];
  let pendingBuffer: WSPointFitted[] = [];
  let flushScheduled = false;
  let pointIndexMap: Map<number, number> = new Map();

  // --- Active classification profile (synced with store) ---
  let activeProfile: ClassificationProfile | null = null;

  // Sync local vars from stores on init
  function restoreFromStores(): void {
    fittedResultsCache = [...$mapFittedResultsCache];
    activeProfile = $mapActiveProfile;
  }

  // Persist local vars to stores (called after mutations)
  function persistToStores(): void {
    mapFittedResultsCache.set(fittedResultsCache);
    mapActiveProfile.set(activeProfile);
  }

  // --- Inline spectrum viewer state ---
  let spectrumMode: 'empty' | 'class_average' | 'single_point' = 'empty';
  let spectrumClassInfo: { domain: string; class_id: string; label: string } | null = null;
  let spectrumPointIndex: number | null = null;

  // Domain → class_id mapping (fitting engine canonical IDs)
  const DOMAIN_CLASSES: Record<string, string[]> = {
    minerals: ['olivine', 'phosphate', 'pyroxene', 'sulf1_v1', 'sulf2_v1',
               '1050', 'lo-carb', 'hi-carb', 'sulf_v3'],
    organics: ['D_band', 'G_band'],
    hydration: ['OH_stretch'],
    fluorescence: ['group3', 'group1a', 'group1b', 'group2'],
  };

  // Store subscriptions
  let pointSet: PointSet | null = null;
  let layers: ScalarLayer[] = [];

  const unsubPointSet = mapPointSet.subscribe((v) => (pointSet = v));
  const unsubLayers = mapLayers.subscribe((v) => (layers = v));

  // Watch display mode changes: load layer data on first non-default selection
  // Also drives inline spectrum viewer
  const unsubDisplayMode = mapDisplayMode.subscribe((mode) => {
    // Deferred DB layer loading (only when layerSource is 'init')
    if (availableLayers && !layerDataLoaded && layerSource === 'init' && mode.type !== 'all_domains') {
      layerDataLoaded = true;
      loadLayerData(availableLayers);
    }

    // Drive spectrum viewer panel
    if (mode.type === 'class' && 'class_id' in mode) {
      const label = classLabel(mode.class_id);
      spectrumClassInfo = { domain: mode.domain, class_id: mode.class_id, label };
      spectrumMode = 'class_average';
      spectrumPointIndex = null;
    } else {
      spectrumClassInfo = null;
      if (spectrumMode === 'class_average') {
        spectrumMode = 'empty';
      }
      // Keep single_point mode if user clicked a point
    }
  });

  onMount(async () => {
    const isReturning = $mapScanId === scanId && $mapPointSet !== null;

    if (!isReturning) {
      resetMapState();
      mapScanId.set(scanId);
    } else {
      // Restore local state from stores
      restoreFromStores();
      // Rebuild pointIndexMap from existing pointSet
      if ($mapPointSet) {
        pointIndexMap = new Map($mapPointSet.points.map((p, i) => [p.index, i]));
      }
      // Restore layerDataLoaded if layers have values
      if ($mapLayers.some((l) => l.values.length > 0)) {
        layerDataLoaded = true;
      }
    }

    // Authenticated ACI image fetch (Session 93 design memo §2.3).
    // Stale-load guard via mapLoadGeneration; rapid scan switches discard
    // older decoded images before they reach `aciImage`.
    // Clear prior-scan refs BEFORE awaiting the new fetch so MapCanvas
    // doesn't keep rendering scan-A's image while scan-B is loading or
    // auth-error-stating (Codex PR9 R1 F1).
    aciImage = null;
    colorizedAciImage = null;
    const gen = ++mapLoadGeneration;
    mapAuthRequired = false;
    aciLoading = true;
    try {
      const img = await fetchAciImage(scanId, {});
      if (gen === mapLoadGeneration) aciImage = img;
    } catch (e) {
      if (gen !== mapLoadGeneration) return;
      if (e instanceof AuthRequiredError) {
        mapAuthRequired = true;
        loadError = '';
        aciLoading = false;
        return;
      }
      // Non-auth fetch error: surface as load error but allow map-layers
      // metadata fetch below to proceed (it may have its own success path).
      console.error('Failed to fetch ACI image:', e);
      loadError = 'Failed to load ACI image';
    } finally {
      if (gen === mapLoadGeneration) aciLoading = false;
    }

    // Fetch scan info for label. Auth-attaching via getScan() helper —
    // raw `fetch('/api/scans/...')` would skip Bearer and 401 under Auth0
    // (Codex PR9 R3 F5).
    try {
      const data = await getScan(scanId);
      const scan = data.scan;
      scanLabel = `${scan.target ?? 'Unknown'} Sol ${scan.sol_number} ${scan.scan_name}`;
    } catch {
      // non-critical (incl. AuthRequiredError — the auth-required state is
      // already surfaced via the ACI fetch failure above)
    }

    // Skip layer fetch on return — stores already have everything
    if (isReturning) return;

    // Fetch available map layers (authenticated via getMapLayers wrapper).
    try {
      const data = await getMapLayers(scanId);
      if (gen !== mapLoadGeneration) return; // stale (scan switched mid-flight)
      // Always accept geometry (source-agnostic)
      mapPointSet.set({
        scan_id: scanId,
        source: 'sherloc',
        coordinate_source: data.coordinate_source ?? 'aci_pixel',
        points: data.point_set?.points ?? [],
        voronoi: data.point_set?.voronoi ?? null,
      });
      const colorizedEntry = (data.base_images ?? []).find(
        (b) => b.type === 'aci_colorized',
      );
      if (colorizedEntry) {
        // Lazy-fetch colorized variant via authenticated path.
        try {
          const cimg = await fetchAciImage(scanId, { colorized: true });
          if (gen === mapLoadGeneration) colorizedAciImage = cimg;
        } catch (e) {
          // Colorized fetch failure is non-fatal; grayscale still works.
          if (!(e instanceof AuthRequiredError)) {
            console.warn('Colorized ACI fetch failed:', e);
          }
        }
      }
      // Only build layers from DB if layerSource is still 'init'
      // (user may have started fitting before this fetch returned)
      if (data.available_layers && layerSource === 'init') {
        availableLayers = data.available_layers;
        buildLayersFromAvailable(data.available_layers);
        // Do NOT auto-load layer data here — show clean ACI first.
        // Data loads on demand when user selects a display mode.
      }
    } catch (e) {
      if (gen !== mapLoadGeneration) return;
      if (e instanceof AuthRequiredError) {
        mapAuthRequired = true;
        loadError = '';
      } else if ((e as { status?: number })?.status === 404) {
        // No map data yet -- just show ACI
        loadError = '';
      } else {
        console.error('Failed to load map layers:', e);
        loadError = 'Failed to load map layers';
      }
    }
  });

  onDestroy(() => {
    ws?.close();
    // Persist local state to stores (don't reset — enables return)
    persistToStores();
    if (mapCanvas) {
      mapSavedTransform.set(mapCanvas.getTransform?.() ?? null);
    }
    unsubPointSet();
    unsubLayers();
    unsubDisplayMode();
  });

  // Center values for known class_ids (used for display labels).
  // Raman in cm-1, fluorescence in nm.
  const CLASS_CENTERS: Record<string, { center: number; unit: string }> = {
    olivine: { center: 836, unit: 'cm\u207b\u00b9' },
    phosphate: { center: 962, unit: 'cm\u207b\u00b9' },
    pyroxene: { center: 998, unit: 'cm\u207b\u00b9' },
    sulf1_v1: { center: 1016, unit: 'cm\u207b\u00b9' },
    sulf2_v1: { center: 1026, unit: 'cm\u207b\u00b9' },
    '1050': { center: 1050, unit: 'cm\u207b\u00b9' },
    'lo-carb': { center: 1067, unit: 'cm\u207b\u00b9' },
    'hi-carb': { center: 1088, unit: 'cm\u207b\u00b9' },
    sulf_v3: { center: 1132, unit: 'cm\u207b\u00b9' },
    D_band: { center: 1350, unit: 'cm\u207b\u00b9' },
    G_band: { center: 1598, unit: 'cm\u207b\u00b9' },
    OH_stretch: { center: 3434, unit: 'cm\u207b\u00b9' },
    group3: { center: 285, unit: 'nm' },
    group1a: { center: 304, unit: 'nm' },
    group1b: { center: 326, unit: 'nm' },
    g1_doublet: { center: 315, unit: 'nm' },
    group2: { center: 341, unit: 'nm' },
    unidentified: { center: 0, unit: '' },
  };

  // Domain sort order: minerals, organics, hydration, fluorescence last
  const DOMAIN_ORDER: Record<string, number> = {
    minerals: 0, organics: 1, hydration: 2, fluorescence: 3,
  };

  function classLabel(cls: string): string {
    const info = CLASS_CENTERS[cls];
    if (info && info.center > 0) {
      return `${cls} ~${info.center} ${info.unit}`;
    }
    return cls;
  }

  function buildLayersFromAvailable(
    available: Record<string, Record<string, { n_detections: number; classes: string[] }>>,
  ): void {
    // Build layers from DB-available classes only.
    // Each class gets a formatted label like "sulf1_v1 ~1016 cm⁻¹".
    const newLayers: ScalarLayer[] = [];
    let idx = 0;

    // Collect all (domain, class) pairs then sort
    const allEntries: { domain: string; cls: string }[] = [];
    for (const [_source, domains] of Object.entries(available)) {
      for (const [domain, info] of Object.entries(domains)) {
        for (const cls of info.classes) {
          if (cls === 'unidentified') continue; // skip unidentified
          allEntries.push({ domain, cls });
        }
      }
    }

    // Sort: by domain order, then by center within domain
    allEntries.sort((a, b) => {
      const da = DOMAIN_ORDER[a.domain] ?? 99;
      const db = DOMAIN_ORDER[b.domain] ?? 99;
      if (da !== db) return da - db;
      const ca = CLASS_CENTERS[a.cls]?.center ?? 0;
      const cb = CLASS_CENTERS[b.cls]?.center ?? 0;
      return ca - cb;
    });

    for (const entry of allEntries) {
      newLayers.push({
        id: `${entry.domain}:${entry.cls}:snr:${idx}`,
        point_set_id: scanId,
        label: classLabel(entry.cls),
        domain: entry.domain,
        class_id: entry.cls,
        value_type: 'snr',
        values: [],
        colormap: { type: 'sequential', name: 'viridis', range: [0, 20] },
        opacity: 0.7,
        visible: true,
      });
      idx++;
    }

    // Add domain-level "all" layers
    const seenDomains = new Set(allEntries.map((e) => e.domain));
    for (const domain of seenDomains) {
      newLayers.push({
        id: `${domain}:all:snr:${idx}`,
        point_set_id: scanId,
        label: `All ${domain}`,
        domain,
        class_id: null,
        value_type: 'snr',
        values: [],
        colormap: { type: 'sequential', name: 'viridis', range: [0, 20] },
        opacity: 0.7,
        visible: false,
      });
      idx++;
    }

    mapLayers.set(newLayers);
  }

  async function loadLayerData(
    available: Record<string, Record<string, { n_detections: number; classes: string[] }>>,
  ): Promise<void> {
    const capturedGen = sourceGeneration;
    type ApiPoint = {
      index: number;
      value: number | null;
      status: string;
      assignment?: string | null;
      center_cm1?: number | null;
      center_nm?: number | null;
    };

    // Fetch data per class (not just per domain) so each class gets its own
    // values independent of which class has the highest SNR at a given point.
    for (const [_source, domains] of Object.entries(available)) {
      for (const [domain, info] of Object.entries(domains)) {
        if (info.n_detections === 0) continue;

        // Fetch domain-level "all" (no class_id filter — returns max SNR peak per point).
        // Auth-attaching via getMapData() — raw fetch skipped Bearer (Codex PR9 R3 F5).
        try {
          const data = await getMapData(scanId, domain, 'snr');
          if (sourceGeneration !== capturedGen) return; // stale
          const apiPoints: ApiPoint[] = (data.points ?? []) as unknown as ApiPoint[];
          updateLayerValues(domain, null, apiPoints);
        } catch (e) {
          if (e instanceof AuthRequiredError) {
            // The auth-required state was surfaced earlier in onMount() via
            // mapAuthRequired; no separate UI hook needed here.
            return;
          }
          console.warn(`Failed to load domain data for ${domain}:`, e);
        }

        // Fetch per-class data
        for (const cls of info.classes) {
          if (sourceGeneration !== capturedGen) return; // stale
          try {
            const data = await getMapData(scanId, domain, 'snr', cls);
            if (sourceGeneration !== capturedGen) return; // stale
            const apiPoints: ApiPoint[] = (data.points ?? []) as unknown as ApiPoint[];
            updateLayerValues(domain, cls, apiPoints);
          } catch (e) {
            if (e instanceof AuthRequiredError) return;
            console.warn(`Failed to load class data for ${domain}/${cls}:`, e);
          }
        }
      }
    }
  }

  function updateLayerValues(
    domain: string,
    classId: string | null,
    apiPoints: Array<{ index: number; value: number | null; status: string; assignment?: string | null; center_cm1?: number | null; center_nm?: number | null }>,
  ): void {
    mapLayers.update((currentLayers) => {
      return currentLayers.map((layer) => {
        if (layer.domain !== domain) return layer;
        if (layer.class_id !== classId) return layer;

        const measuredValues = apiPoints
          .filter((p) => p.status === 'measured' && p.value != null)
          .map((p) => p.value as number);
        const vmin = measuredValues.length > 0 ? Math.min(...measuredValues) : 0;
        const vmax = measuredValues.length > 0 ? Math.max(...measuredValues) : 1;

        const values = apiPoints.map((p) => ({
          value: p.value,
          status: p.status as 'measured' | 'below_threshold' | 'missing' | 'masked',
          uncertainty: null,
          metadata: {
            assignment: p.assignment,
            center_cm1: p.center_cm1,
            center_nm: p.center_nm,
          },
        }));

        return {
          ...layer,
          values,
          colormap: { ...layer.colormap, range: [vmin, vmax] as [number, number] },
        };
      });
    });
  }

  // --- Layer initialization for live fitting (spec §3.1.1) ---

  function makeEmptyValue(): LayerValue {
    return { value: null, status: 'missing', uncertainty: null, metadata: {} };
  }

  function ensureFittingLayers(domains: string[]): void {
    if (!pointSet) return;
    const nPoints = pointSet.points.length;

    // Transition layer ownership to live_fit
    layerSource = 'live_fit';
    sourceGeneration++;
    layerDataLoaded = true; // block deferred DB fetch

    // Build point index map
    pointIndexMap = new Map(pointSet.points.map((p, i) => [p.index, i]));

    // Reset caches
    fittedResultsCache = [];
    pendingBuffer = [];
    flushScheduled = false;

    // Build layers for requested domains, preserve others
    const existingNonFitted = layers.filter((l) => !domains.includes(l.domain));
    const newLayers: ScalarLayer[] = [...existingNonFitted];
    let idx = existingNonFitted.length;

    // Collect entries, sort, then build
    const entries: { domain: string; cls: string }[] = [];
    for (const domain of domains) {
      const classes = DOMAIN_CLASSES[domain] ?? [];
      for (const cls of classes) {
        entries.push({ domain, cls });
      }
    }
    entries.sort((a, b) => {
      const da = DOMAIN_ORDER[a.domain] ?? 99;
      const db = DOMAIN_ORDER[b.domain] ?? 99;
      if (da !== db) return da - db;
      const ca = CLASS_CENTERS[a.cls]?.center ?? 0;
      const cb = CLASS_CENTERS[b.cls]?.center ?? 0;
      return ca - cb;
    });

    for (const entry of entries) {
      newLayers.push({
        id: `${entry.domain}:${entry.cls}:snr:${idx}`,
        point_set_id: scanId,
        label: classLabel(entry.cls),
        domain: entry.domain,
        class_id: entry.cls,
        value_type: 'snr',
        values: Array.from({ length: nPoints }, makeEmptyValue),
        colormap: { type: 'sequential', name: 'viridis', range: [0, 30] },
        opacity: 0.7,
        visible: true,
      });
      idx++;
    }

    // Add domain-level "all" layers
    for (const domain of domains) {
      newLayers.push({
        id: `${domain}:all:snr:${idx}`,
        point_set_id: scanId,
        label: `All ${domain}`,
        domain,
        class_id: null,
        value_type: 'snr',
        values: Array.from({ length: nPoints }, makeEmptyValue),
        colormap: { type: 'sequential', name: 'viridis', range: [0, 30] },
        opacity: 0.7,
        visible: false,
      });
      idx++;
    }

    mapLayers.set(newLayers);

    // Auto-switch display mode so results are visible
    if ($mapDisplayMode.type === 'all_domains') {
      mapDisplayMode.set({ type: 'domain', domain: domains[0] });
    }
  }

  // --- Profile-aware ingest (spec §3.1.2) ---

  function reclassifyWSResult(
    msg: WSPointFitted,
    rules: ClassificationRule[],
  ): WSPointFitted {
    const newResults: Record<string, { status: string; peaks: typeof msg.results[string]['peaks'] }> = {};

    for (const [domain, domainResult] of Object.entries(msg.results)) {
      const domainRules = rules.filter((r) => r.domain === domain);
      const newPeaks = domainResult.peaks.map((peak) => {
        const center = peak.center_cm1 ?? peak.center_nm ?? 0;
        const matchingRule = domainRules.find(
          (r) => !r.disabled && Math.abs(center - r.center) <= r.range,
        );
        return { ...peak, assignment: matchingRule ? matchingRule.class_id : 'unidentified' };
      });

      // Recompute domain status (spec §3.3.4)
      const identified = newPeaks.filter((p) => {
        if (p.assignment === 'unidentified') return false;
        const rule = domainRules.find((r) => r.class_id === p.assignment);
        return rule ? p.snr >= rule.snr_threshold : p.snr >= 3.0;
      });
      const subThreshold = newPeaks.filter((p) => {
        if (p.assignment === 'unidentified') return false;
        const rule = domainRules.find((r) => r.class_id === p.assignment);
        return rule ? p.snr < rule.snr_threshold : p.snr < 3.0;
      });

      let status: string;
      if (identified.length > 0) status = 'measured';
      else if (subThreshold.length > 0) status = 'below_threshold';
      else status = 'missing';

      newResults[domain] = { status, peaks: newPeaks };
    }

    return { ...msg, results: newResults };
  }

  function ingestPointFitted(msg: WSPointFitted): void {
    // Cache raw message (immutable) — store persisted on destroy
    fittedResultsCache.push(msg);

    // Apply active profile if set
    const effective = activeProfile
      ? reclassifyWSResult(msg, activeProfile.rules)
      : msg;

    pendingBuffer.push(effective);

    // Schedule rAF-based flush
    if (!flushScheduled) {
      flushScheduled = true;
      requestAnimationFrame(flushPending);
    }
  }

  function flushPending(): void {
    flushScheduled = false;
    if (pendingBuffer.length === 0) return;

    const batch = pendingBuffer;
    pendingBuffer = [];

    mapLayers.update((currentLayers) => {
      // Collect domains present in this batch
      const batchDomains = new Set<string>();
      for (const msg of batch) {
        for (const domain of Object.keys(msg.results)) {
          batchDomains.add(domain);
        }
      }

      return currentLayers.map((layer) => {
        if (!batchDomains.has(layer.domain)) return layer;

        // Clone values array
        const values = [...layer.values];

        for (const msg of batch) {
          const domainResult = msg.results[layer.domain];
          if (!domainResult) continue;

          const arrayIndex = pointIndexMap.get(msg.point_index);
          if (arrayIndex === undefined) continue;

          if (layer.class_id === null) {
            // Domain "all" layer: best SNR among identified peaks
            const identifiedPeaks = domainResult.peaks.filter(
              (p) => p.assignment !== 'unidentified',
            );
            const bestPeak = identifiedPeaks.reduce<typeof domainResult.peaks[0] | null>(
              (best, p) => (!best || p.snr > best.snr ? p : best),
              null,
            );
            values[arrayIndex] = {
              value: bestPeak ? bestPeak.snr : null,
              status: (domainResult.status as LayerValue['status']) || 'missing',
              uncertainty: null,
              metadata: bestPeak
                ? { assignment: bestPeak.assignment, center_cm1: bestPeak.center_cm1, center_nm: bestPeak.center_nm }
                : {},
            };
          } else {
            // Class-specific layer
            const peak = domainResult.peaks.find((p) => p.assignment === layer.class_id);
            if (peak) {
              values[arrayIndex] = {
                value: peak.snr,
                status: 'measured',
                uncertainty: null,
                metadata: { assignment: peak.assignment, center_cm1: peak.center_cm1, center_nm: peak.center_nm },
              };
            } else if (domainResult.status === 'below_threshold') {
              values[arrayIndex] = {
                value: null,
                status: 'below_threshold',
                uncertainty: null,
                metadata: {},
              };
            }
            // else: leave as 'missing' (initial value)
          }
        }

        return { ...layer, values };
      });
    });
  }

  function recalcColormapRanges(): void {
    mapLayers.update((currentLayers) =>
      currentLayers.map((layer) => {
        const measured = layer.values.filter(
          (v) => v.status === 'measured' && v.value != null,
        );
        if (measured.length === 0) return layer;
        const vals = measured.map((v) => v.value as number);
        const vmin = Math.min(...vals);
        const vmax = Math.max(...vals);
        return {
          ...layer,
          colormap: { ...layer.colormap, range: [vmin, Math.max(vmax, vmin + 1)] as [number, number] },
        };
      }),
    );
  }

  function buildLayersFromProfile(rules: ClassificationRule[]): void {
    if (!pointSet) return;
    const nPoints = pointSet.points.length;
    const enabledRules = rules.filter((r) => !r.disabled);

    const newLayers: ScalarLayer[] = [];
    let idx = 0;

    // Group by domain, sort
    const entries = enabledRules.map((r) => ({ domain: r.domain, cls: r.class_id, label: r.label }));
    entries.sort((a, b) => {
      const da = DOMAIN_ORDER[a.domain] ?? 99;
      const db = DOMAIN_ORDER[b.domain] ?? 99;
      if (da !== db) return da - db;
      const ca = CLASS_CENTERS[a.cls]?.center ?? 0;
      const cb = CLASS_CENTERS[b.cls]?.center ?? 0;
      return ca - cb;
    });

    for (const entry of entries) {
      newLayers.push({
        id: `${entry.domain}:${entry.cls}:snr:${idx}`,
        point_set_id: scanId,
        label: classLabel(entry.cls),
        domain: entry.domain,
        class_id: entry.cls,
        value_type: 'snr',
        values: Array.from({ length: nPoints }, makeEmptyValue),
        colormap: { type: 'sequential', name: 'viridis', range: [0, 30] },
        opacity: 0.7,
        visible: true,
      });
      idx++;
    }

    // Add domain-level "all" layers
    const seenDomains = new Set(enabledRules.map((r) => r.domain));
    for (const domain of seenDomains) {
      newLayers.push({
        id: `${domain}:all:snr:${idx}`,
        point_set_id: scanId,
        label: `All ${domain}`,
        domain,
        class_id: null,
        value_type: 'snr',
        values: Array.from({ length: nPoints }, makeEmptyValue),
        colormap: { type: 'sequential', name: 'viridis', range: [0, 30] },
        opacity: 0.7,
        visible: false,
      });
      idx++;
    }

    mapLayers.set(newLayers);
  }

  function handleProfileApply(e: CustomEvent<{ profile: ClassificationProfile }>) {
    const { profile } = e.detail;
    if (fittedResultsCache.length === 0) return;

    activeProfile = profile;
    layerSource = 'profile';
    sourceGeneration++;

    // Build layers from profile rules
    buildLayersFromProfile(profile.rules);

    // Re-ingest all cached results through the profile
    for (const msg of fittedResultsCache) {
      const effective = reclassifyWSResult(msg, profile.rules);
      pendingBuffer.push(effective);
    }
    flushPending();
    recalcColormapRanges();
  }

  async function handleStartFit(e: CustomEvent<{ domains: string[] }>) {
    const { domains } = e.detail;
    try {
      // Auth-attaching POST via startMapFit() — raw fetch skipped Bearer
      // and produced 401 under Auth0 (Codex PR9 R3 F5).
      let data: import('../../lib/api').MapFitResponse;
      try {
        data = await startMapFit(scanId, domains);
      } catch (err) {
        if (err instanceof AuthRequiredError) {
          mapAuthRequired = true;
          return;
        }
        throw err;
      }

      // Initialize layers for live fitting BEFORE WS connects
      ensureFittingLayers(domains);

      mapFitJob.set({
        jobId: data.job_id,
        status: 'queued',
        fitted: 0,
        total: data.n_points,
        etaSeconds: 0,
      });
      mapLogEntries.set([]);

      // Connect WebSocket
      ws = new MapWebSocket(data.ws_url, {
          onJobStarted: (msg) => {
            mapFitJob.update((j) =>
              j ? { ...j, status: 'running' } : j,
            );
            if (msg.voronoi) {
              mapPointSet.update((ps) =>
                ps ? { ...ps, voronoi: msg.voronoi } : ps,
              );
            }
          },
          onPointFitted: (msg: WSPointFitted) => {
            mapFitJob.update((j) =>
              j ? { ...j, fitted: (j.fitted || 0) + 1 } : j,
            );
            // Feed results into mapLayers via the ingest pipeline
            ingestPointFitted(msg);
          },
          onProgress: (msg) => {
            mapFitJob.update((j) =>
              j
                ? { ...j, fitted: msg.fitted, total: msg.total, etaSeconds: msg.eta_s }
                : j,
            );
          },
          onLog: (msg) => {
            mapLogEntries.update((logs) => [...logs.slice(-499), msg.message]);
          },
          onComplete: (msg) => {
            // Flush any remaining pending results
            flushPending();
            recalcColormapRanges();
            mapFitJob.update((j) =>
              j
                ? {
                    ...j,
                    status: 'complete',
                    fitted: msg.summary.total_points,
                  }
                : j,
            );
          },
          onFailed: (msg) => {
            flushPending();
            recalcColormapRanges();
            mapFitJob.update((j) =>
              j ? { ...j, status: 'failed' } : j,
            );
            mapLogEntries.update((logs) => [...logs, `ERROR: ${msg.error}`]);
          },
          onCancelled: () => {
            flushPending();
            recalcColormapRanges();
            mapFitJob.update((j) =>
              j ? { ...j, status: 'cancelled' } : j,
            );
          },
        onDisconnect: () => {
          // Connection closed -- status remains as-is
        },
      });
    } catch (e) {
      // ApiError from startMapFit (non-2xx) or any other failure during
      // job-create / WS init. AuthRequiredError was already caught inline.
      const detail = (e as { detail?: string; message?: string })?.detail
        ?? (e as Error)?.message
        ?? String(e);
      console.error('Failed to start fitting:', e);
      mapFitJob.set({ jobId: '', status: 'failed', fitted: 0, total: 0, etaSeconds: 0 });
      mapLogEntries.update((logs) => [...logs, `Fitting error: ${detail}`]);
    }
  }

  function handleCancelFit() {
    ws?.sendCancel();
  }

  async function refreshLayers() {
    // Transition back to DB-owned layers
    layerSource = 'init';
    sourceGeneration++;
    activeProfile = null;
    const capturedGen = sourceGeneration;
    // Note: fittedResultsCache is preserved (editor can re-apply profile)

    try {
      const data = await getMapLayers(scanId);
      // Guard against stale response
      if (sourceGeneration !== capturedGen) return;

      if (data.point_set) {
        mapPointSet.set({
          scan_id: scanId,
          source: 'sherloc',
          coordinate_source: data.coordinate_source ?? 'aci_pixel',
          points: data.point_set.points,
          voronoi: data.point_set.voronoi,
        });
      }
      if (data.available_layers && layerSource === 'init') {
        availableLayers = data.available_layers;
        buildLayersFromAvailable(data.available_layers);
        layerDataLoaded = true;
        await loadLayerData(data.available_layers);
      }
    } catch (e) {
      // AuthRequiredError or any other fetch failure: log and skip the
      // refresh. The map-layers panel will surface the error via its
      // existing UI state if the call site cares.
      console.error('Failed to refresh layers:', e);
    }
  }

  function handlePointClick(e: CustomEvent<{ pointIndex: number }>) {
    spectrumMode = 'single_point';
    spectrumPointIndex = e.detail.pointIndex;
  }

  function handleBackToAverage() {
    if (spectrumClassInfo) {
      spectrumMode = 'class_average';
      spectrumPointIndex = null;
    } else {
      spectrumMode = 'empty';
      spectrumPointIndex = null;
    }
  }
</script>

<div class="page-container map-page">
  <!-- Breadcrumb -->
  <div class="breadcrumb">
    <button class="btn-link" on:click={() => navigate('#/')}>Scans</button>
    <span class="breadcrumb-sep">/</span>
    <button class="btn-link" on:click={() => navigate(`#/scan/${scanId}`)}>
      {scanLabel || scanId}
    </button>
    <span class="breadcrumb-sep">/</span>
    <span>Map Mode</span>
  </div>

  {#if mapAuthRequired}
    <div class="error-banner">Log in to view map data and ACI image.</div>
  {:else if loadError}
    <div class="error-banner">{loadError}</div>
  {/if}

  <div class="map-layout">
    <!-- Sidebar -->
    <div class="map-sidebar">
      <!-- Scan info card -->
      <div class="sidebar-card">
        <div class="card-header">Scan</div>
        <div class="card-body">
          <div class="scan-info mono">{scanLabel || scanId}</div>
          {#if pointSet}
            <div class="scan-stat">{pointSet.points.length} points</div>
          {/if}
        </div>
      </div>

      <!-- Display controls -->
      <div class="sidebar-card">
        <div class="card-header">Display</div>
        <div class="card-body">
          <MapControls />

          <div class="brightness-controls">
            <label class="slider-label">
              Brightness: {brightness.toFixed(1)}
              <input
                type="range"
                min="0.2"
                max="3.0"
                step="0.1"
                bind:value={brightness}
              />
            </label>
            <label class="slider-label">
              Contrast: {contrast.toFixed(1)}
              <input
                type="range"
                min="0.2"
                max="3.0"
                step="0.1"
                bind:value={contrast}
              />
            </label>
          </div>
        </div>
      </div>

      <!-- Fit controls -->
      <div class="sidebar-card">
        <div class="card-header">Fitting</div>
        <div class="card-body">
          <FitControls
            {scanId}
            on:startFit={handleStartFit}
            on:cancelFit={handleCancelFit}
          />
        </div>
      </div>

      <!-- Classification editor toggle -->
      <div class="sidebar-card">
        <button
          class="card-header toggle-header"
          on:click={() => (showClassEditor = !showClassEditor)}
        >
          <span>Classification</span>
          <span class="toggle-icon">{showClassEditor ? '-' : '+'}</span>
        </button>
        {#if showClassEditor}
          <div class="card-body">
            <ClassificationEditor
              hasFittedData={fittedResultsCache.length > 0}
              on:profileApply={handleProfileApply}
              on:profileReset={() => {
                if (fittedResultsCache.length === 0) return;
                activeProfile = null;
                layerSource = 'live_fit';
                sourceGeneration++;
                // Rebuild default layers and re-ingest from cache
                const domains = [...new Set(fittedResultsCache.flatMap((m) => Object.keys(m.results)))];
                ensureFittingLayers(domains);
                for (const msg of fittedResultsCache) {
                  pendingBuffer.push(msg);
                }
                flushPending();
                recalcColormapRanges();
              }}
            />
          </div>
        {/if}
      </div>
    </div>

    <!-- Main area -->
    <div class="map-main">
      <div class="canvas-area">
        <MapCanvas
          bind:this={mapCanvas}
          {scanId}
          {pointSet}
          {layers}
          geometryMode={$mapGeometryMode}
          displayMode={$mapDisplayMode}
          overlayOpacity={$mapOverlayOpacity}
          showPointPositions={$mapShowPointPositions}
          initialTransform={$mapSavedTransform}
          {brightness}
          {contrast}
          {aciImage}
          {colorizedAciImage}
          {aciLoading}
          on:pointClick={handlePointClick}
        />
      </div>

      <div class="bottom-panel">
        <div class="progress-col">
          <MapProgressPanel />
        </div>
        <div class="spectrum-col">
          <MapSpectrumPanel
            {scanId}
            mode={spectrumMode}
            classInfo={spectrumClassInfo}
            pointIndex={spectrumPointIndex}
            on:backToAverage={handleBackToAverage}
            on:openWorkbench={() => {
              if (spectrumPointIndex != null) {
                navigate(`#/scan/${scanId}/workbench?point=${spectrumPointIndex}`);
              } else {
                navigate(`#/scan/${scanId}/workbench`);
              }
            }}
          />
        </div>
      </div>
    </div>
  </div>
</div>

<style>
  .map-page {
    display: flex;
    flex-direction: column;
    height: calc(100vh - 48px);
    overflow: hidden;
  }

  .breadcrumb {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 0;
    font-size: 0.85rem;
    color: var(--color-text-secondary);
    flex-shrink: 0;
  }

  .breadcrumb-sep {
    color: var(--color-text-tertiary);
  }

  .btn-link {
    background: none;
    border: none;
    color: var(--color-primary);
    padding: 0;
    cursor: pointer;
    font-size: inherit;
  }

  .btn-link:hover {
    text-decoration: underline;
  }

  .error-banner {
    background: var(--color-error-light, #fef2f2);
    color: var(--color-error, #dc2626);
    padding: 8px 12px;
    border-radius: var(--radius-md);
    font-size: 0.85rem;
    margin-bottom: 8px;
    flex-shrink: 0;
  }

  .map-layout {
    display: grid;
    grid-template-columns: 320px 1fr;
    gap: 12px;
    flex: 1;
    min-height: 0;
    overflow: hidden;
  }

  @media (max-width: 900px) {
    .map-layout {
      grid-template-columns: 1fr;
      grid-template-rows: auto 1fr;
    }
  }

  /* Sidebar */
  .map-sidebar {
    display: flex;
    flex-direction: column;
    gap: 8px;
    overflow-y: auto;
    padding-bottom: 16px;
  }

  .sidebar-card {
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    overflow: visible;
  }

  .card-header {
    padding: 8px 12px;
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--color-text);
    background: var(--color-background);
    border-bottom: 1px solid var(--color-border);
  }

  .toggle-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    width: 100%;
    cursor: pointer;
    border: none;
    border-radius: 0;
  }

  .toggle-header:hover {
    background: var(--color-primary-light, rgba(59, 130, 246, 0.06));
  }

  .toggle-icon {
    font-family: var(--font-mono);
    font-size: 1rem;
    color: var(--color-text-tertiary);
  }

  .card-body {
    padding: 10px 12px;
  }

  .scan-info {
    font-size: 0.82rem;
    color: var(--color-text);
    word-break: break-all;
  }

  .scan-stat {
    font-size: 0.78rem;
    color: var(--color-text-secondary);
    margin-top: 2px;
  }

  .brightness-controls {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid var(--color-border);
  }


  .slider-label {
    display: flex;
    flex-direction: column;
    gap: 2px;
    font-size: 0.78rem;
    color: var(--color-text-secondary);
    margin-bottom: 0;
  }

  .slider-label input[type='range'] {
    width: 100%;
  }

  /* Main area — fills the grid cell, same height as sidebar */
  .map-main {
    display: flex;
    flex-direction: column;
    min-height: 0;
    height: 100%;
    gap: 8px;
  }

  .canvas-area {
    flex: 1;
    min-height: 200px;
    border-radius: var(--radius-md);
    overflow: hidden;
    border: 1px solid var(--color-border);
  }

  .bottom-panel {
    height: 340px;
    display: flex;
    flex-direction: column;
    gap: 4px;
    flex-shrink: 0;
  }

  .progress-col {
    flex-shrink: 0;
  }

  .spectrum-col {
    flex: 1;
    min-height: 0;
  }
</style>
