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
    support: number;
    trace_revision_ids: string[];
    citations: { trace_revision_id: string; citation_text: string | null; [key: string]: unknown }[];
  }[];
  [key: string]: unknown;
}

export function referenceEvidence(record: ReferenceRecord & { x: number[]; y: number[] }): EvidenceTrace | null {
  const p = record.provenance;
  if (p.x_quantity !== 'raman_shift' && p.x_quantity !== 'wavelength') return null;
  const raman = p.x_quantity === 'raman_shift';
  const x = raman ? record.x : [];
  const trace = evidenceTrace({ stage: 'raw', params: {}, processingChain: [],
    raman: { wavenumber: x, intensity: record.y },
    trace: { region: raman ? 'R1' : 'R2', source: { reference: record, citation: p.citation_text },
      selection: { trace_revision_id: record.trace_revision_id }, channel_indices: [],
      wavenumber: x, wavelength: raman ? null : record.x, intensity: record.y,
      provenance: { calibration_version: null, data_origin: 'reference', source: { citation: p.citation_text, reference: record },
        wavenumber_unit: raman ? p.x_unit : 'unknown', wavelength_unit: raman ? 'unknown' : p.x_unit, intensity_unit: p.y_unit } },
  });
  return trace ? { ...trace, label: p.species_name ?? p.source_label, color: '#059669' } : null;
}
