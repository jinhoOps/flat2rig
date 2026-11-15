import { clamp, type RegionDeformParams, type ShapeDeformInput } from "../engine";
import {
  clampParameters,
  createDefaultShapePresets,
  DEFAULT_EYE_HANDLES,
  DEFAULT_MOUTH_HANDLES,
  REST,
  type ExpressionPreset,
  type Parameters,
  type Project,
  type ShapeHandleValues,
  type ShapePreset,
  type ShapeRegion,
  type ShapeSubstituteAsset,
} from "../types";

export interface HairPhysicsState {
  position: number;
  velocity: number;
  previousTime: number;
}

export function createPhysicsState(): HairPhysicsState {
  return { position: 0, velocity: 0, previousTime: 0 };
}

export function stepHairPhysics(
  prev: HairPhysicsState,
  timeSec: number,
  targetHeadX: number,
  hairScale: number,
): HairPhysicsState {
  if (!Number.isFinite(timeSec) || timeSec <= 0 || timeSec < prev.previousTime) {
    return { position: 0, velocity: 0, previousTime: Math.max(0, timeSec || 0) };
  }
  const dt = clamp(timeSec - prev.previousTime, 0, 0.05);
  if (dt <= 0) return prev;
  const target =
    (Math.sin(timeSec * 0.9) * 0.55 + clamp(targetHeadX / 25, -1, 1) * 0.35) *
    hairScale;
  const force = (target - prev.position) * 35 - prev.velocity * 7;
  const velocity = clamp(prev.velocity + force * dt, -6, 6);
  const position = clamp(prev.position + velocity * dt, -1.2, 1.2);
  return {
    position,
    velocity,
    previousTime: timeSec,
  };
}

export interface EvaluateRigInput {
  project: Project;
  timeSec?: number;
  playing?: boolean;
  activeExpressionId?: string;
  previousExpressionId?: string | null;
  transitionElapsedMs?: number;
  userParameters?: Parameters;
  userOffsets?: Partial<Parameters>;
  overrideShapeIds?: Partial<Record<ShapeRegion, string>>;
  overrideHandles?: Partial<Record<ShapeRegion, Partial<ShapeHandleValues>>>;
  physicsState?: HairPhysicsState;
  deterministic?: boolean;
}

export interface EvaluatedRigFrame {
  parameters: Parameters;
  activeShapeIds: Record<ShapeRegion, string>;
  activeShapes: Record<ShapeRegion, ShapePreset>;
  activeHandles: Record<ShapeRegion, ShapeHandleValues>;
  shapeDeform: ShapeDeformInput;
  hiddenBasePartIds: Set<string>;
  mouthInteriorAlpha: number;
  eyeIrisAlpha: { L: number; R: number };
  eyeWhiteAlpha: { L: number; R: number };
  eyeLowerAlpha: { L: number; R: number };
  eyeClosedOrSmiling: { L: boolean; R: boolean };
  nextPhysicsState: HairPhysicsState;
  isRest: boolean;
}

function getShapeByIdOrFallback(
  project: Project,
  region: ShapeRegion,
  shapeId: string | undefined,
): ShapePreset {
  const presets = project.shapePresets;
  if (shapeId) {
    const exact = presets.find((s) => s.id === shapeId && s.region === region);
    if (exact) return exact;
  }
  const defId = project.rigProfile.defaultShapeIds[region];
  const byDef = presets.find((s) => s.id === defId && s.region === region);
  if (byDef) return byDef;
  const firstOfRegion = presets.find((s) => s.region === region);
  if (firstOfRegion) return firstOfRegion;
  const fallbackHandles =
    region === "mouth" ? { ...DEFAULT_MOUTH_HANDLES } : { ...DEFAULT_EYE_HANDLES };
  return {
    id: `${region}_original`,
    region,
    label: "원본",
    kind: "original",
    recipeVersion: 1,
    builtIn: true,
    handles: fallbackHandles,
    parameterOverrides: {},
  };
}

function originalHandlesForRegion(
  project: Project,
  region: ShapeRegion,
): ShapeHandleValues {
  const orig = project.shapePresets.find(
    (s) => s.region === region && s.kind === "original",
  );
  if (orig) return orig.handles;
  return region === "mouth" ? DEFAULT_MOUTH_HANDLES : DEFAULT_EYE_HANDLES;
}

function bakedBaselineHandles(
  project: Project,
  region: ShapeRegion,
  preset: ShapePreset,
  defaultPresets: ShapePreset[],
  shapeAssets: Record<string, ShapeSubstituteAsset> | undefined,
): ShapeHandleValues {
  if (preset.assetId && shapeAssets?.[preset.assetId]) {
    const defPreset = defaultPresets.find(
      (s) => s.id === preset.id && s.region === region,
    );
    if (defPreset) return defPreset.handles;
  }
  return originalHandlesForRegion(project, region);
}

function toRegionDeform(
  handles: ShapeHandleValues,
  orig: ShapeHandleValues,
  weight: number,
): RegionDeformParams | undefined {
  if (weight <= 1e-4) return undefined;
  const dw = (handles.width - orig.width) * weight;
  const dr = (handles.roundness - orig.roundness) * weight;
  const dc = (handles.corner - orig.corner) * weight;
  const du = (handles.upperCurve - orig.upperCurve) * weight;
  const dl = (handles.lowerCurve - orig.lowerCurve) * weight;
  const dox = (handles.offsetX - orig.offsetX) * weight;
  const doy = (handles.offsetY - orig.offsetY) * weight;
  if (
    Math.abs(dw) < 1e-4 &&
    Math.abs(dr) < 1e-4 &&
    Math.abs(dc) < 1e-4 &&
    Math.abs(du) < 1e-4 &&
    Math.abs(dl) < 1e-4 &&
    Math.abs(dox) < 1e-4 &&
    Math.abs(doy) < 1e-4
  ) {
    return undefined;
  }
  return {
    width: 1 + dw,
    roundnessDelta: dr,
    corner: dc,
    upperCurveDelta: du,
    lowerCurveDelta: dl,
    offsetX: dox,
    offsetY: doy,
  };
}

function blendHandles(
  a: ShapeHandleValues,
  b: ShapeHandleValues,
  t: number,
): ShapeHandleValues {
  const lerp = (x: number, y: number) => x + (y - x) * t;
  return {
    width: lerp(a.width, b.width),
    open: lerp(a.open, b.open),
    roundness: lerp(a.roundness, b.roundness),
    corner: lerp(a.corner, b.corner),
    upperCurve: lerp(a.upperCurve, b.upperCurve),
    lowerCurve: lerp(a.lowerCurve, b.lowerCurve),
    interiorVisibility: lerp(a.interiorVisibility, b.interiorVisibility),
    offsetX: lerp(a.offsetX, b.offsetX),
    offsetY: lerp(a.offsetY, b.offsetY),
  };
}

function scaleKeyGroup(
  key: keyof Parameters,
  scale: Project["rigProfile"]["motionScale"],
): number {
  if (key === "angleX" || key === "angleY" || key === "angleZ") return scale.head;
  if (
    key === "eyeL" ||
    key === "eyeR" ||
    key === "eyeLSmile" ||
    key === "eyeRSmile" ||
    key === "eyeBallX" ||
    key === "eyeBallY" ||
    key === "browLY" ||
    key === "browRY" ||
    key === "browLForm" ||
    key === "browRForm"
  ) {
    return scale.eyes;
  }
  if (key === "mouth" || key === "mouthForm") return scale.mouth;
  if (key === "hair" || key === "hairSide" || key === "hairBack") {
    return scale.hair;
  }
  return scale.body;
}

function blinkPulse(timeSec: number, intervalSec: number): number {
  const period = clamp(intervalSec, 1.0, 15.0);
  const blinkDur = 0.18;
  // Offset phase slightly so t=0 starts awake rather than mid-blink
  const phase = (timeSec + period * 0.5) % period;
  if (phase > blinkDur) return 0;
  const s = Math.sin((phase / blinkDur) * Math.PI);
  return s * s;
}

export function evaluateRigFrame(input: EvaluateRigInput): EvaluatedRigFrame {
  const {
    project,
    timeSec = 0,
    playing = false,
    transitionElapsedMs = 0,
    userParameters,
    userOffsets,
    overrideShapeIds,
    overrideHandles,
    deterministic = false,
  } = input;

  const { rigProfile, expressions, playback, shapeAssets } = project;
  const neutral = rigProfile.neutralParameters;

  // Stage 1: Base neutral parameters & character default shape selection
  const pose: Parameters = { ...neutral };
  const baseShapeIds: Record<ShapeRegion, string> = {
    eye_L: rigProfile.defaultShapeIds.eye_L,
    eye_R: rigProfile.defaultShapeIds.eye_R,
    mouth: rigProfile.defaultShapeIds.mouth,
  };

  // Stage 2: Expression selection, cross-fade, shape preset overrides, and expression parameters
  const activeExpId = input.activeExpressionId ?? project.activeExpressionId;
  const activeExp: ExpressionPreset | undefined =
    expressions.find((e) => e.id === activeExpId) ?? expressions[0];
  const prevExp: ExpressionPreset | undefined =
    !deterministic &&
    input.previousExpressionId &&
    input.previousExpressionId !== activeExp?.id &&
    activeExp &&
    activeExp.transitionMs > 0 &&
    transitionElapsedMs < activeExp.transitionMs
      ? expressions.find((e) => e.id === input.previousExpressionId)
      : undefined;

  let tAlpha = 1;
  if (prevExp && activeExp && activeExp.transitionMs > 0) {
    const u = clamp(transitionElapsedMs / activeExp.transitionMs, 0, 1);
    tAlpha = u * u * (3 - 2 * u);
  }

  const activeShapeIds: Record<ShapeRegion, string> = {
    eye_L:
      overrideShapeIds?.eye_L ??
      ((tAlpha < 0.5 ? prevExp?.shapeIds.eye_L : activeExp?.shapeIds.eye_L) ??
        activeExp?.shapeIds.eye_L ??
        baseShapeIds.eye_L),
    eye_R:
      overrideShapeIds?.eye_R ??
      ((tAlpha < 0.5 ? prevExp?.shapeIds.eye_R : activeExp?.shapeIds.eye_R) ??
        activeExp?.shapeIds.eye_R ??
        baseShapeIds.eye_R),
    mouth:
      overrideShapeIds?.mouth ??
      ((tAlpha < 0.5 ? prevExp?.shapeIds.mouth : activeExp?.shapeIds.mouth) ??
        activeExp?.shapeIds.mouth ??
        baseShapeIds.mouth),
  };

  const activeShapes: Record<ShapeRegion, ShapePreset> = {
    eye_L: getShapeByIdOrFallback(project, "eye_L", activeShapeIds.eye_L),
    eye_R: getShapeByIdOrFallback(project, "eye_R", activeShapeIds.eye_R),
    mouth: getShapeByIdOrFallback(project, "mouth", activeShapeIds.mouth),
  };

  const expStrength = activeExp ? clamp(activeExp.strength, 0, 1.5) : 1;
  const prevStrength = prevExp ? clamp(prevExp.strength, 0, 1.5) : 1;

  // Apply shape parameterOverrides first (scaled by expression strength unless manually forced)
  const applyShapeOverrides = (
    exp: ExpressionPreset | undefined,
    strength: number,
    blendWeight: number,
  ) => {
    const regions: ShapeRegion[] = ["eye_L", "eye_R", "mouth"];
    for (const region of regions) {
      const explicitOverride = Boolean(overrideShapeIds?.[region]);
      const sid =
        overrideShapeIds?.[region] ?? exp?.shapeIds[region] ?? baseShapeIds[region];
      const preset = getShapeByIdOrFallback(project, region, sid);
      const fromExp = Boolean(exp?.shapeIds[region] && !explicitOverride);
      const w = fromExp ? clamp(strength, 0, 1) : 1;
      for (const [k, targetVal] of Object.entries(preset.parameterOverrides)) {
        const key = k as keyof Parameters;
        if (typeof targetVal !== "number") continue;
        const shapedVal = neutral[key] + (targetVal - neutral[key]) * w;
        pose[key] += (shapedVal - pose[key]) * blendWeight;
      }
      // Also reflect user handle openness changes not captured in parameterOverrides
      const origOpen = originalHandlesForRegion(project, region).open;
      if (region === "mouth" && preset.parameterOverrides.mouth === undefined) {
        const openDelta = preset.handles.open - origOpen;
        if (Math.abs(openDelta) > 1e-4) {
          pose.mouth += openDelta * w * blendWeight;
        }
      } else if (region === "eye_L" && preset.parameterOverrides.eyeL === undefined) {
        const openDelta = preset.handles.open - origOpen;
        if (Math.abs(openDelta) > 1e-4) {
          pose.eyeL += openDelta * w * blendWeight;
        }
      } else if (region === "eye_R" && preset.parameterOverrides.eyeR === undefined) {
        const openDelta = preset.handles.open - origOpen;
        if (Math.abs(openDelta) > 1e-4) {
          pose.eyeR += openDelta * w * blendWeight;
        }
      }
    }
  };

  if (prevExp) {
    applyShapeOverrides(prevExp, prevStrength, 1);
  }
  applyShapeOverrides(activeExp, expStrength, prevExp ? tAlpha : 1);

  // Blend expression direct parameter targets weighted by strength and transition progress
  const applyExpParams = (
    exp: ExpressionPreset | undefined,
    strength: number,
    blendWeight: number,
  ) => {
    if (!exp) return;
    for (const [k, rawVal] of Object.entries(exp.parameters)) {
      const key = k as keyof Parameters;
      if (typeof rawVal !== "number") continue;
      const weightedTarget = neutral[key] + (rawVal - neutral[key]) * strength;
      pose[key] = pose[key] + (weightedTarget - pose[key]) * blendWeight;
    }
  };

  if (prevExp) {
    applyExpParams(prevExp, prevStrength, 1);
  }
  applyExpParams(activeExp, expStrength, prevExp ? tAlpha : 1);

  // Resolve effective handles for shape deformation
  const defaultShapePresets = createDefaultShapePresets(rigProfile.proportionMode);
  const activeHandles = {} as Record<ShapeRegion, ShapeHandleValues>;
  const handleBaselines = {} as Record<ShapeRegion, ShapeHandleValues>;
  for (const region of ["eye_L", "eye_R", "mouth"] as const) {
    const preset = activeShapes[region];
    const baseline = bakedBaselineHandles(
      project,
      region,
      preset,
      defaultShapePresets,
      shapeAssets,
    );
    handleBaselines[region] = baseline;
    const fromExp = Boolean(
      activeExp?.shapeIds[region] && !overrideShapeIds?.[region],
    );
    const strengthW = fromExp ? clamp(expStrength, 0, 1) * tAlpha : 1;
    let h = blendHandles(baseline, preset.handles, strengthW);
    const tweak = activeExp?.handleTweaks?.[region];
    if (tweak) {
      h = {
        ...h,
        ...Object.fromEntries(
          Object.entries(tweak)
            .filter(([, v]) => typeof v === "number")
            .map(([k, v]) => [
              k,
              h[k as keyof ShapeHandleValues] +
                ((v as number) - baseline[k as keyof ShapeHandleValues]) * strengthW,
            ]),
        ),
      };
    }
    if (overrideHandles?.[region]) {
      h = { ...h, ...overrideHandles[region] };
    }
    activeHandles[region] = h;
    // Allow direct overrideHandles.open or expression tweaks to reflect into pose openness
    if (overrideHandles?.[region]?.open !== undefined) {
      const deltaOpen = overrideHandles[region]!.open! - preset.handles.open;
      if (region === "eye_L") pose.eyeL += deltaOpen;
      if (region === "eye_R") pose.eyeR += deltaOpen;
      if (region === "mouth") pose.mouth += deltaOpen;
    }
  }

  // Stage 3: User region adjustments + motionScale multipliers
  if (userParameters) {
    for (const key of Object.keys(REST) as (keyof Parameters)[]) {
      const delta = userParameters[key] - neutral[key];
      if (Math.abs(delta) > 1e-6) {
        pose[key] += delta;
      }
    }
  }
  if (userOffsets) {
    for (const [k, delta] of Object.entries(userOffsets)) {
      const key = k as keyof Parameters;
      if (typeof delta === "number" && Number.isFinite(delta)) {
        pose[key] += delta;
      }
    }
  }
  for (const key of Object.keys(REST) as (keyof Parameters)[]) {
    const mult = scaleKeyGroup(key, rigProfile.motionScale);
    if (Math.abs(mult - 1) > 1e-6) {
      pose[key] = neutral[key] + (pose[key] - neutral[key]) * mult;
    }
  }

  // Track deliberate eye closure/wink/smile BEFORE auto-blink so auto-blink respects expression/user priority
  const eyeClosedOrSmiling = {
    L:
      activeShapes.eye_L.kind === "closed" ||
      activeShapes.eye_L.kind === "smiling" ||
      pose.eyeL < 0.25 ||
      pose.eyeLSmile > 0.6,
    R:
      activeShapes.eye_R.kind === "closed" ||
      activeShapes.eye_R.kind === "smiling" ||
      pose.eyeR < 0.25 ||
      pose.eyeRSmile > 0.6,
  };

  // Stage 4: Allowed auto motion (NO automatic mouth movement unless playback.autoMouthDemo === true)
  const autoActive = playing && !deterministic;
  if (autoActive) {
    if (playback.autoIdle) {
      pose.angleX += Math.sin(timeSec * 0.9) * 5.5 * rigProfile.motionScale.head;
      pose.angleY += Math.sin(timeSec * 0.65) * 3.2 * rigProfile.motionScale.head;
      pose.angleZ += Math.sin(timeSec * 0.45) * 2.0 * rigProfile.motionScale.head;
      pose.eyeBallX +=
        Math.sin(timeSec * 0.55) * 0.28 * rigProfile.motionScale.eyes;
      pose.eyeBallY +=
        Math.sin(timeSec * 0.8 + 1.0) * 0.14 * rigProfile.motionScale.eyes;
      pose.browLY += Math.sin(timeSec * 0.9) * 0.12 * rigProfile.motionScale.eyes;
      pose.browRY += Math.sin(timeSec * 0.9) * 0.12 * rigProfile.motionScale.eyes;
      pose.bodyAngleX +=
        Math.sin(timeSec * 0.45) * 2.8 * rigProfile.motionScale.body;
      pose.bodyAngleZ +=
        Math.sin(timeSec * 0.45 + 0.4) * 1.6 * rigProfile.motionScale.body;
    }
    if (playback.autoBreath) {
      pose.breath = Math.max(
        pose.breath,
        (0.5 + 0.5 * Math.sin(timeSec * 1.4)) * rigProfile.motionScale.body,
      );
    }
    if (playback.autoBlink) {
      const blink = blinkPulse(timeSec, playback.blinkIntervalSec);
      if (blink > 0) {
        if (!eyeClosedOrSmiling.L) pose.eyeL *= 1 - blink;
        if (!eyeClosedOrSmiling.R) pose.eyeR *= 1 - blink;
      }
    }
    if (playback.autoMouthDemo) {
      pose.mouth = Math.max(
        pose.mouth,
        Math.max(0, Math.sin(timeSec * 2)) * 0.48 * rigProfile.motionScale.mouth,
      );
    }
  }

  // Stage 5: Safety clamp
  const safePose = clampParameters(pose, rigProfile.safeBounds);

  // Stage 6: Hair spring physics (reset when paused / deterministic / seeked backward)
  let nextPhysicsState: HairPhysicsState;
  if (autoActive) {
    nextPhysicsState = stepHairPhysics(
      input.physicsState ?? createPhysicsState(),
      timeSec,
      safePose.angleX,
      rigProfile.motionScale.hair,
    );
    safePose.hair += nextPhysicsState.position;
    safePose.hairSide += nextPhysicsState.position * 0.85;
    safePose.hairBack += nextPhysicsState.position * 0.65;
  } else {
    nextPhysicsState = {
      position: 0,
      velocity: 0,
      previousTime: deterministic ? 0 : Math.max(0, timeSec),
    };
  }
  const finalPose = clampParameters(safePose, rigProfile.safeBounds);

  // Swap open/half-closed eye shapes to per-character closed substitute during blink/closure
  const blinkAssist = { L: false, R: false };
  const maybeSwapToClosedEye = (
    region: "eye_L" | "eye_R",
    eyeVal: number,
    side: "L" | "R",
  ) => {
    const current = activeShapes[region];
    if (
      !shapeAssets ||
      eyeClosedOrSmiling[side] ||
      (current.kind !== "original" && current.kind !== "half_closed") ||
      eyeVal > 0.22
    ) {
      return;
    }
    const closedPreset = project.shapePresets.find(
      (s) => s.region === region && s.kind === "closed",
    );
    if (!closedPreset?.assetId || !shapeAssets[closedPreset.assetId]) return;
    activeShapes[region] = closedPreset;
    activeShapeIds[region] = closedPreset.id;
    blinkAssist[side] = true;
    const baked = bakedBaselineHandles(
      project,
      region,
      closedPreset,
      defaultShapePresets,
      shapeAssets,
    );
    handleBaselines[region] = baked;
    activeHandles[region] = overrideHandles?.[region]
      ? {
          ...baked,
          ...overrideHandles[region],
          open: closedPreset.handles.open,
        }
      : { ...closedPreset.handles };
  };
  maybeSwapToClosedEye("eye_L", finalPose.eyeL, "L");
  maybeSwapToClosedEye("eye_R", finalPose.eyeR, "R");

  const shapeDeform: ShapeDeformInput = {
    eyeL: toRegionDeform(activeHandles.eye_L, handleBaselines.eye_L, 1),
    eyeR: toRegionDeform(activeHandles.eye_R, handleBaselines.eye_R, 1),
    mouth: toRegionDeform(activeHandles.mouth, handleBaselines.mouth, 1),
  };

  const hiddenBasePartIds = new Set<string>();
  if (shapeAssets) {
    for (const region of ["eye_L", "eye_R", "mouth"] as const) {
      const aid = activeShapes[region].assetId;
      const asset = aid ? shapeAssets[aid] : undefined;
      if (!asset) continue;
      for (const id of asset.hidesPartIds) hiddenBasePartIds.add(id);
      for (const layer of asset.layers) {
        for (const id of layer.hidesPartIds ?? []) {
          hiddenBasePartIds.add(id);
        }
      }
    }
  }

  const hasLegacyEyeAsset = {
    L: Boolean(activeShapes.eye_L.assetId && shapeAssets?.[activeShapes.eye_L.assetId]),
    R: Boolean(activeShapes.eye_R.assetId && shapeAssets?.[activeShapes.eye_R.assetId]),
  };
  const computeEyeAlphas = (side: "L" | "R", eyeVal: number) => {
    const region = side === "L" ? "eye_L" : "eye_R";
    if (hasLegacyEyeAsset[side] && (eyeClosedOrSmiling[side] || blinkAssist[side])) {
      return { iris: 0, white: 0, lower: 0 };
    }
    const iris =
      eyeVal <= 0.06
        ? 0
        : clamp((eyeVal - 0.06) / 0.14, 0, 1) *
          clamp(activeHandles[region].interiorVisibility, 0, 1);
    const white = eyeVal <= 0.03 ? 0 : clamp((eyeVal - 0.03) / 0.11, 0, 1);
    const lower = eyeVal <= 0.15 ? 0 : clamp((eyeVal - 0.15) / 0.35, 0, 1);
    return { iris, white, lower };
  };
  const leftEyeAlphas = computeEyeAlphas("L", finalPose.eyeL);
  const rightEyeAlphas = computeEyeAlphas("R", finalPose.eyeR);
  const eyeIrisAlpha = { L: leftEyeAlphas.iris, R: rightEyeAlphas.iris };
  const eyeWhiteAlpha = { L: leftEyeAlphas.white, R: rightEyeAlphas.white };
  const eyeLowerAlpha = { L: leftEyeAlphas.lower, R: rightEyeAlphas.lower };
  const mouthInteriorAlpha =
    finalPose.mouth <= 0.03
      ? 0
      : clamp((finalPose.mouth - 0.03) / 0.12, 0, 1) *
        clamp(activeHandles.mouth.interiorVisibility, 0, 1);

  const isRest =
    !shapeDeform.eyeL &&
    !shapeDeform.eyeR &&
    !shapeDeform.mouth &&
    hiddenBasePartIds.size === 0 &&
    (Object.keys(REST) as (keyof Parameters)[]).every(
      (k) => Math.abs(finalPose[k] - REST[k]) <= 1e-5,
    );

  return {
    parameters: finalPose,
    activeShapeIds,
    activeShapes,
    activeHandles,
    shapeDeform,
    hiddenBasePartIds,
    mouthInteriorAlpha,
    eyeIrisAlpha,
    eyeWhiteAlpha,
    eyeLowerAlpha,
    eyeClosedOrSmiling,
    nextPhysicsState,
    isRest,
  };
}
