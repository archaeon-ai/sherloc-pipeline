// ============================================================
// Processing chain state management
// ============================================================

import { writable, get } from 'svelte/store';
import type { ProcessingSnapshot, PointSelectionState } from './types';

const MAX_UNDO_DEPTH = 20;

// --- Processing state (current snapshot) ---
export const processingState = writable<ProcessingSnapshot | null>(null);

// --- Undo stack ---
export const undoStack = writable<ProcessingSnapshot[]>([]);

// --- Point selection mode ---
export const pointSelection = writable<PointSelectionState>({ mode: 'average' });

// --- Actions ---

/**
 * Push the current state onto the undo stack before transitioning to a new state.
 * Maintains a max stack depth of 20.
 */
export function pushUndo(snapshot: ProcessingSnapshot): void {
  undoStack.update((stack) => {
    const next = [...stack, snapshot];
    if (next.length > MAX_UNDO_DEPTH) {
      return next.slice(next.length - MAX_UNDO_DEPTH);
    }
    return next;
  });
}

/**
 * Pop the most recent snapshot from the undo stack and return it.
 * Also sets it as the current processing state.
 * Returns null if the stack is empty.
 */
export function undo(): ProcessingSnapshot | null {
  const stack = get(undoStack);
  if (stack.length === 0) return null;

  const previous = stack[stack.length - 1];
  undoStack.update((s) => s.slice(0, -1));
  processingState.set(previous);
  return previous;
}

/**
 * Reset processing back to raw state.
 * Clears the undo stack and sets processingState to the raw snapshot
 * stored at the bottom of the undo stack (or null if empty).
 */
export function resetProcessing(rawSnapshot?: ProcessingSnapshot): void {
  undoStack.set([]);
  if (rawSnapshot) {
    processingState.set(rawSnapshot);
  } else {
    // Try to recover the raw state from the current undo stack
    const current = get(processingState);
    if (current) {
      processingState.set({
        stage: 'raw',
        raman: current.raman,
        params: {},
        artifacts: undefined,
      });
    } else {
      processingState.set(null);
    }
  }
}
