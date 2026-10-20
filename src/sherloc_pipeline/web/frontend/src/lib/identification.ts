import type { EvidenceTrace } from './evidenceTypes';
import { evidenceTrace } from './evidenceStore';

export interface ReferenceBinding {
  reference_release_id: string;
  reference_subset_id: string;
  method_release_id: string;
  template_ids: string[];
}

export interface ReferenceRecord {
  trace_revision_id: string;
  release_id: string;
  axis_sha256: string;
  intensity_sha256: string;
  point_count: number;
  provenance: {
    source_label: string;
    species_name: string | null;
    x_quantity: string;
    x_unit: string;
    y_unit: string | null;
    citation_text: string | null;
    [key: string]: unknown;
  };
}

export interface ReferencePage {
  items: ReferenceRecord[];
  next_cursor: string | null;
  binding: ReferenceBinding | null;
}

export interface IdentificationEvidence {
  result_digest: string;
  source_digest: string;
  status: 'candidate' | 'abstained';
  reasons: string[];
  method: { release_id: string; support_min: number; status: 'provisional'; [key: string]: unknown };
  reference_release_id: string;
  reference_subset_id: string;
  ranking: {
    template_id: string;
    species_name: string;
    support: number | null;
    retrieval_score?: number | null;
    trace_revision_ids: string[];
    citations: { trace_revision_id: string; citation_text: string | null; [key: string]: unknown }[];
  }[];
  mixture?: {
    support: number | null;
    components: { template_id: string; species_name: string; coefficient: number; weight: number }[];
    diagnostics: { converged: boolean; finite_solution: boolean; template_ids?: string[] };
  } | null;
  evidence?: {
    grid: number[];
    mask: boolean[];
    prepared_trace: number[] | null;
    inspection?: FitEvidence | null;
    mixture_fit?: FitEvidence | null;
  };
  [key: string]: unknown;
}

interface FitEvidence {
  mask: boolean[];
  prepared_trace?: (number | null)[];
  fit: (number | null)[];
  residual: (number | null)[];
  converged?: boolean;
}

export function identificationPlot(result: IdentificationEvidence | null) {
  const evidence = result?.evidence;
  const mixture = result?.method.method_id === 'raman-mixture';
  const fit = mixture ? evidence?.mixture_fit : evidence?.inspection;
  if (!evidence || !fit || !evidence.prepared_trace) return null;
  const size = evidence.grid.length;
  if (![fit.mask, fit.fit, fit.residual, evidence.prepared_trace].every(values => values.length === size)) return null;
  const masked = (values: (number | null)[]) => values.map((value, i) => fit.mask[i] && Number.isFinite(value) ? value : null);
  const label = mixture ? 'Mixture fit' : 'Pearson inspection fit';
  return { label, converged: mixture ? fit.converged === true : true, grid: evidence.grid,
    traces: [
      { name: 'Prepared source', y: masked(evidence.prepared_trace) },
      { name: label, y: masked(fit.fit) },
      { name: `${label} residual`, y: masked(fit.residual) },
    ] };
}

export function evidenceCitation(citation: Record<string, unknown> | undefined): string {
  return String(citation?.citation_text || citation?.citation_doi || citation?.citation_source || 'missing');
}

export function referenceEvidence(record: ReferenceRecord & { x: number[]; y: number[] }): EvidenceTrace | null {
  const p = record.provenance;
  if (p.x_quantity !== 'raman_shift' && p.x_quantity !== 'wavelength') return null;
  const raman = p.x_quantity === 'raman_shift';
  const x = raman ? record.x : [];
  const trace = evidenceTrace({ stage: 'raw', params: {}, processingChain: [],
    raman: { wavenumber: x, intensity: record.y },
    trace: { region: raman ? 'R1' : 'R2', source: { reference: record, citation: evidenceCitation(p) },
      selection: { trace_revision_id: record.trace_revision_id }, channel_indices: [],
      wavenumber: x, wavelength: raman ? null : record.x, intensity: record.y,
      provenance: { calibration_version: null, data_origin: 'reference', source: { citation: evidenceCitation(p), reference: record },
        wavenumber_unit: raman ? p.x_unit : 'unknown', wavelength_unit: raman ? 'unknown' : p.x_unit, intensity_unit: p.y_unit } },
  });
  return trace ? { ...trace, label: p.species_name ?? p.source_label, color: '#059669' } : null;
}
