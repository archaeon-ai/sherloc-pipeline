// Scan fit status shown in the Scan Browser and the Processing Workbench.
// The API derives `has_fitted_peaks` from fitted_peaks rows at read time;
// the stored `processing_status` column has no writer, so it is not shown.

export function fitStatusLabel(hasFittedPeaks: boolean | undefined): string {
  return hasFittedPeaks ? 'peaks fitted' : 'no fitted peaks';
}

export function fitStatusBadge(hasFittedPeaks: boolean | undefined): string {
  return hasFittedPeaks ? 'badge-success' : 'badge-neutral';
}
