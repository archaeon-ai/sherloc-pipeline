import type { ExportTrace, ProcessingSnapshot } from './types';

export type EvidenceDomain = 'raman_shift' | 'emission_wavelength';
export type EvidenceTheme = 'system' | 'light' | 'dark';

export interface EvidenceTrace {
  id: string;
  label: string;
  snapshot: ProcessingSnapshot & { trace: ExportTrace };
  visible: boolean;
  color: string;
  breaks: boolean[];
  annotations: Record<string, unknown>;
}

export interface EvidenceDisplay {
  trace: EvidenceTrace;
  domain: EvidenceDomain;
  x: number[];
  original: number[];
  breaks: boolean[];
  displayed: number[];
  divisor: number | null;
  offset: number;
  scalable: boolean;
  normalization: 'native' | 'fixed_window_maximum';
}
