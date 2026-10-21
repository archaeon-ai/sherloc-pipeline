import type { ExportTrace, ProcessingSnapshot } from './types';

export type EvidenceDomain = 'raman_shift' | 'emission_wavelength';
export type EvidenceTheme = 'system' | 'light' | 'dark';

export interface EvidenceTrace {
  id: string;
  label: string;
  nativeDomain: EvidenceDomain | null;
  snapshot: ProcessingSnapshot & { trace: ExportTrace };
  visible: boolean;
  color: string;
  breaks: boolean[];
  annotations: Record<string, unknown>;
  // Set only on a shared-chain output (see lib/sharedChain.ts).
  derived?: {
    derived_from: string;
    shared_chain_id: string;
    original_processing_chain: Record<string, unknown>[];
    steps: unknown[];
    scale_used: number | null;
    reference_processing_state?: unknown;
  };
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
