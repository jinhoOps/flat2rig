import { useEffect, useState } from "react";
import { CAT_BOB, CAT_FRAMES, CAT_H, CAT_W, frameRuns } from "./catSprite";
import { progressPercent } from "./progress";

const REDUCED_QUERY = "(prefers-reduced-motion: reduce)";

/** Follows the OS "reduce motion" setting; false where `matchMedia` is unavailable. */
export function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(() => {
    try {
      return typeof window !== "undefined" && !!window.matchMedia?.(REDUCED_QUERY).matches;
    } catch {
      return false;
    }
  });
  useEffect(() => {
    if (typeof window === "undefined" || !window.matchMedia) return;
    const mq = window.matchMedia(REDUCED_QUERY);
    const on = () => setReduced(mq.matches);
    on();
    mq.addEventListener?.("change", on);
    return () => mq.removeEventListener?.("change", on);
  }, []);
  return reduced;
}

const VIEW_H = CAT_H + 1; // room for the one-pixel bob

/** All four frames side by side; CSS `steps(4)` slides the strip one frame at a time. */
function CatStrip({ still }: { still: boolean }) {
  const frames = still ? [CAT_FRAMES[0]] : CAT_FRAMES;
  return (
    <svg
      className="cat-strip"
      viewBox={`0 0 ${CAT_W * frames.length} ${VIEW_H}`}
      shapeRendering="crispEdges"
      aria-hidden="true"
      focusable="false"
    >
      {frames.map((frame, f) =>
        frameRuns(frame, still ? 0 : CAT_BOB[f]).map((r, i) => (
          <rect
            key={`${f}-${i}`}
            x={f * CAT_W + r.x}
            y={r.y}
            width={r.w}
            height={1}
            fill={r.color}
          />
        )),
      )}
    </svg>
  );
}

export interface CatProgressProps {
  /** What is running (also the accessible name). */
  label: string;
  /** 0..1 when known; `null` = indeterminate (the cat laps the track). */
  fraction: number | null;
  /** Step now running, shown under the bar. */
  step?: string;
  /** Overrides the OS setting (tests, previews). */
  reducedMotion?: boolean;
  /** Status-bar size. */
  compact?: boolean;
  /** Second copy of a bar already announced elsewhere (status bar): hidden from assistive tech. */
  decorative?: boolean;
}

/**
 * T11a: pixel-cat progress bar. Determinate: the bar fills and the cat runs to the current
 * position. Indeterminate: the cat laps the track. Reduced motion: a still cat plus percent or
 * step text, no animation.
 */
export default function CatProgress({
  label,
  fraction,
  step,
  reducedMotion,
  compact,
  decorative,
}: CatProgressProps) {
  const osReduced = usePrefersReducedMotion();
  const still = reducedMotion ?? osReduced;
  const pct = progressPercent(fraction);
  const determinate = pct !== null;
  const mode = determinate ? "determinate" : "indeterminate";
  const text = determinate ? `${pct}%` : still ? "진행 중" : "";
  const valueText = [determinate ? `${pct}%` : "진행 중", step].filter(Boolean).join(" · ");
  return (
    <div
      className={`cat-progress is-${mode}${still ? " is-still" : ""}${compact ? " is-compact" : ""}`}
      {...(decorative
        ? { "aria-hidden": true as const }
        : {
            role: "progressbar",
            "aria-label": label,
            "aria-valuemin": determinate ? 0 : undefined,
            "aria-valuemax": determinate ? 100 : undefined,
            "aria-valuenow": determinate ? pct : undefined,
            "aria-valuetext": valueText,
            "aria-busy": true,
          })}
      title={decorative ? `${label} · ${valueText}` : undefined}
      data-mode={mode}
      data-still={still ? "true" : "false"}
      style={{ "--cat-p": determinate ? pct / 100 : 0 } as React.CSSProperties}
    >
      <div className="cat-track">
        {determinate && <div className="cat-fill" />}
        <div className="cat-runner">
          <div className="cat-window">
            <CatStrip still={still} />
          </div>
        </div>
      </div>
      {!compact && (step || text) && (
        <div className="cat-caption">
          <span className="cat-step">{step}</span>
          {text && <span className="cat-pct">{text}</span>}
        </div>
      )}
      {compact && text && <span className="cat-pct">{text}</span>}
    </div>
  );
}
