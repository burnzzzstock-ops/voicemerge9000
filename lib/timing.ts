/** Original detection bounds stay fixed even after edits and splits. Times are seconds. */
export type SpeechWindow = {
  start: number;
  end: number;
  detectedStart: number;
  detectedEnd: number;
  minStart: number;
  maxEnd: number;
};

export function isValidTimingEdit(
  region: SpeechWindow,
  start: number,
  end: number,
  others: ReadonlyArray<{ start: number; end: number }>,
): boolean {
  // Match the engine's sub-sample tolerance for seconds/ms serialization noise.
  const epsilon = 1e-10;
  return Number.isFinite(start) && Number.isFinite(end) && end - start >= 0.04 - epsilon &&
    start >= region.minStart - epsilon && end <= region.maxEnd + epsilon &&
    start < region.detectedEnd && end > region.detectedStart &&
    !others.some((other) => start < other.end && end > other.start);
}
