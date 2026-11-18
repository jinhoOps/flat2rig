/**
 * T11a: progress model for long `task(label, run)` jobs.
 *
 * Two sources combine into one bar:
 * - `outer`: steps the front end knows about (pipeline stages run in sequence, parts filled in a loop).
 * - `inner`: stage-boundary `task-progress` events from a Rust command (desktop only).
 *
 * When neither gives a real fraction the bar is indeterminate. A single front-end step with no Rust
 * events is also indeterminate: "0 of 1" says nothing about how far along the work is.
 */
export interface StepProgress {
  done: number;
  total: number;
  step?: string;
}

export interface CombinedProgress {
  /** 0..1 when known, `null` for indeterminate. */
  fraction: number | null;
  /** Name of the step now running, if any source named one. */
  step?: string;
}

const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);

function stepFraction(p: StepProgress): number {
  if (!(p.total > 0)) return 0;
  return clamp01(p.done / p.total);
}

export function combineProgress(
  outer: StepProgress | null,
  inner: StepProgress | null,
): CombinedProgress {
  const innerOk = !!inner && inner.total > 0;
  const outerOk = !!outer && outer.total > 0;
  if (outerOk && outer!.total > 1) {
    const base = clamp01(outer!.done / outer!.total);
    const slice = 1 / outer!.total;
    const sub = innerOk ? stepFraction(inner!) * slice : 0;
    const step = [outer!.step, innerOk ? inner!.step : undefined].filter(Boolean).join(" · ");
    return { fraction: clamp01(base + sub), step: step || undefined };
  }
  if (innerOk) {
    const step = [outerOk ? outer!.step : undefined, inner!.step].filter(Boolean).join(" · ");
    return { fraction: stepFraction(inner!), step: step || undefined };
  }
  return { fraction: null, step: outerOk ? outer!.step : undefined };
}

/** Whole percent for text / `aria-valuenow`; `null` when indeterminate. */
export function progressPercent(fraction: number | null): number | null {
  return fraction === null ? null : Math.round(clamp01(fraction) * 100);
}
