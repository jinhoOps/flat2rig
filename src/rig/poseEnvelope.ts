import envelopeDoc from "./pose-envelope.json";
import { REST, type Parameters } from "../types";
import type { ShapeDeformInput } from "../engine";

/**
 * T13a deterministic pose envelope shared with Rust (`src-tauri/src/rig/seams.rs` embeds the same
 * `pose-envelope.json` with `include_str!`). Each sparse entry is merged over `REST`, exactly as
 * Rust merges it over `Pose::default()`. The ranges must equal `PARAM_DEFS` (never shrunk) and
 * every range extreme appears as an `axis` pose; both are asserted in `tests/pipeline.test.ts`.
 */
export interface EnvelopePose {
  name: string;
  group: string;
  params: Parameters;
  /**
   * T10 `eyeSeal` expression poses: the eye `ShapeDeformInput` the app evaluates for that LD
   * expression (asserted against `evaluateRigFrame` in tests). Rust carries it as `Pose::shape`.
   */
  shape?: ShapeDeformInput;
}

type ParamKey = keyof Parameters;

export const POSE_ENVELOPE_VERSION: number = envelopeDoc.version;

export const POSE_ENVELOPE_RANGES: Readonly<Record<ParamKey, readonly [number, number]>> =
  Object.fromEntries(
    Object.entries(envelopeDoc.ranges).map(([k, v]) => [k, [v[0], v[1]] as const]),
  ) as Record<ParamKey, readonly [number, number]>;

export const POSE_ENVELOPE: readonly EnvelopePose[] = envelopeDoc.poses.map((p) => {
  const shape = (p as { shape?: ShapeDeformInput }).shape;
  return {
    name: p.name,
    group: p.group,
    params: { ...REST, ...(p.params as Partial<Parameters>) },
    ...(shape ? { shape } : {}),
  };
});
