import {
  clampParameters,
  type BlendMode,
  type ExpressionPreset,
  type Parameters,
  type PlaybackSettings,
  type Project,
  type RegionHandleRange,
  type RigProfile,
  type ShapeHandleValues,
  type ShapePreset,
  type ShapeRegion,
  REST,
} from "../types";

export interface UndoHistoryState<T> {
  current: T;
  past: T[];
  future: T[];
  dragBefore: T | null;
}

export function createUndoHistory<T>(initial: T, maxDepth = 24): {
  commit: (next: T) => UndoHistoryState<T>;
  beginDrag: () => UndoHistoryState<T>;
  previewDrag: (next: T) => UndoHistoryState<T>;
  commitDrag: () => UndoHistoryState<T>;
  undo: () => UndoHistoryState<T>;
  redo: () => UndoHistoryState<T>;
  reset: (next: T) => UndoHistoryState<T>;
  snapshot: () => UndoHistoryState<T>;
} {
  let current = initial;
  let past: T[] = [];
  let future: T[] = [];
  let dragBefore: T | null = null;

  const snapshot = (): UndoHistoryState<T> => ({
    current,
    past: [...past],
    future: [...future],
    dragBefore,
  });

  const pushPast = (entry: T) => {
    past.push(entry);
    if (past.length > maxDepth) past.shift();
  };

  return {
    commit(next: T) {
      dragBefore = null;
      if (next !== current) {
        pushPast(current);
        current = next;
        future = [];
      }
      return snapshot();
    },
    beginDrag() {
      if (dragBefore === null) dragBefore = current;
      return snapshot();
    },
    previewDrag(next: T) {
      current = next;
      return snapshot();
    },
    commitDrag() {
      const before = dragBefore;
      dragBefore = null;
      if (before !== null && current !== before) {
        pushPast(before);
        future = [];
      }
      return snapshot();
    },
    undo() {
      dragBefore = null;
      const prev = past.pop();
      if (prev !== undefined) {
        future.push(current);
        current = prev;
      }
      return snapshot();
    },
    redo() {
      dragBefore = null;
      const next = future.pop();
      if (next !== undefined) {
        pushPast(current);
        current = next;
      }
      return snapshot();
    },
    reset(next: T) {
      dragBefore = null;
      past = [];
      future = [];
      current = next;
      return snapshot();
    },
    snapshot,
  };
}

export function mirrorEyeRegion(region: ShapeRegion): ShapeRegion | null {
  if (region === "eye_L") return "eye_R";
  if (region === "eye_R") return "eye_L";
  return null;
}

export function findMirrorPresetId(
  presets: ShapePreset[],
  preset: ShapePreset,
): string | null {
  const otherRegion = mirrorEyeRegion(preset.region);
  if (!otherRegion) return null;
  const expectedPrefix = preset.region;
  if (preset.id.startsWith(expectedPrefix)) {
    const candidateId = `${otherRegion}${preset.id.slice(expectedPrefix.length)}`;
    if (presets.some((p) => p.id === candidateId && p.region === otherRegion)) {
      return candidateId;
    }
  }
  const byKind = presets.find(
    (p) => p.region === otherRegion && p.kind === preset.kind,
  );
  return byKind?.id ?? null;
}

export function clampShapeHandles(
  handles: ShapeHandleValues,
): ShapeHandleValues {
  const clamp = (v: number, lo: number, hi: number) =>
    Number.isFinite(v) ? Math.max(lo, Math.min(hi, v)) : lo;
  return {
    width: clamp(handles.width, 0.5, 1.5),
    open: clamp(handles.open, 0, 1.4),
    roundness: clamp(handles.roundness, 0, 1),
    corner: clamp(handles.corner, -1, 1),
    upperCurve: clamp(handles.upperCurve, -1, 1),
    lowerCurve: clamp(handles.lowerCurve, -1, 1),
    interiorVisibility: clamp(handles.interiorVisibility, 0, 1),
    offsetX: clamp(handles.offsetX, -0.5, 0.5),
    offsetY: clamp(handles.offsetY, -0.5, 0.5),
  };
}

export function updateShapePresetHandles(
  project: Project,
  presetId: string,
  patch: Partial<ShapeHandleValues>,
  syncMirror = true,
): Project {
  const target = project.shapePresets.find((s) => s.id === presetId);
  if (!target) return project;
  const nextHandles = clampShapeHandles({ ...target.handles, ...patch });
  const mirrorId =
    syncMirror && target.mirrorLinked
      ? findMirrorPresetId(project.shapePresets, target)
      : null;

  const mirrorPatch: Partial<ShapeHandleValues> = { ...patch };
  if (typeof patch.offsetX === "number") {
    mirrorPatch.offsetX = -patch.offsetX;
  }

  return {
    ...project,
    shapePresets: project.shapePresets.map((preset) => {
      if (preset.id === presetId) {
        return { ...preset, handles: nextHandles };
      }
      if (mirrorId && preset.id === mirrorId) {
        return {
          ...preset,
          handles: clampShapeHandles({ ...preset.handles, ...mirrorPatch }),
        };
      }
      return preset;
    }),
  };
}

export function setExpressionShape(
  project: Project,
  region: ShapeRegion,
  presetId: string,
  expressionId = project.activeExpressionId,
  syncDefaultWhenNeutral = true,
): Project {
  const preset = project.shapePresets.find(
    (s) => s.id === presetId && s.region === region,
  );
  if (!preset) return project;
  const expressions = project.expressions.map((exp) =>
    exp.id === expressionId
      ? { ...exp, shapeIds: { ...exp.shapeIds, [region]: presetId } }
      : exp,
  );
  const rigProfile =
    syncDefaultWhenNeutral && expressionId === "neutral"
      ? {
          ...project.rigProfile,
          defaultShapeIds: {
            ...project.rigProfile.defaultShapeIds,
            [region]: presetId,
          },
        }
      : project.rigProfile;
  return { ...project, expressions, rigProfile };
}

export function selectExpressionPreset(
  project: Project,
  expressionId: string,
): Project {
  if (!project.expressions.some((e) => e.id === expressionId)) return project;
  return {
    ...project,
    activeExpressionId: expressionId,
  };
}

export function saveNeutralParameters(
  project: Project,
  parameters: Parameters,
): Project {
  return {
    ...project,
    rigProfile: {
      ...project.rigProfile,
      neutralParameters: clampParameters(
        { ...REST, ...parameters },
        project.rigProfile.safeBounds,
      ),
    },
  };
}

export function updateRigProfileFields(
  project: Project,
  patch: Partial<RigProfile>,
): Project {
  const nextProfile: RigProfile = {
    ...project.rigProfile,
    ...patch,
    defaultShapeIds: {
      ...project.rigProfile.defaultShapeIds,
      ...(patch.defaultShapeIds ?? {}),
    },
    neutralParameters: clampParameters(
      {
        ...project.rigProfile.neutralParameters,
        ...(patch.neutralParameters ?? {}),
      },
      patch.safeBounds ?? project.rigProfile.safeBounds,
    ),
    safeBounds: {
      ...project.rigProfile.safeBounds,
      ...(patch.safeBounds ?? {}),
    },
    motionScale: {
      ...project.rigProfile.motionScale,
      ...(patch.motionScale ?? {}),
    },
  };
  return { ...project, rigProfile: nextProfile };
}

export function setSafeBound(
  project: Project,
  key: keyof Parameters,
  range: RegionHandleRange,
): Project {
  const safeMin = Math.min(range.min, range.max);
  const safeMax = Math.max(range.min, range.max);
  return updateRigProfileFields(project, {
    safeBounds: {
      ...project.rigProfile.safeBounds,
      [key]: { min: safeMin, max: safeMax },
    },
  });
}

export function updateExpressionPreset(
  project: Project,
  expressionId: string,
  patch: Partial<Omit<ExpressionPreset, "id" | "builtIn">>,
): Project {
  return {
    ...project,
    expressions: project.expressions.map((exp) =>
      exp.id === expressionId
        ? {
            ...exp,
            ...patch,
            name:
              typeof patch.name === "string" && patch.name.trim().length > 0
                ? patch.name.slice(0, 120)
                : exp.name,
            strength:
              typeof patch.strength === "number"
                ? Math.max(0, Math.min(1.5, patch.strength))
                : exp.strength,
            transitionMs:
              typeof patch.transitionMs === "number"
                ? Math.max(0, Math.min(3000, patch.transitionMs))
                : exp.transitionMs,
          }
        : exp,
    ),
  };
}

export function createCustomExpression(
  project: Project,
  name: string,
  sourceExpressionId = project.activeExpressionId,
  overrides?: Partial<Pick<ExpressionPreset, "shapeIds" | "parameters" | "strength" | "transitionMs">>,
): { project: Project; expression: ExpressionPreset } {
  const source =
    project.expressions.find((e) => e.id === sourceExpressionId) ??
    project.expressions[0];
  const existingIds = new Set(project.expressions.map((e) => e.id));
  let idx = project.expressions.length + 1;
  let id = `expr_custom_${idx}`;
  while (existingIds.has(id)) {
    idx++;
    id = `expr_custom_${idx}`;
  }
  const created: ExpressionPreset = {
    id,
    name: name.trim().slice(0, 120) || `표정 ${idx}`,
    builtIn: false,
    shapeIds: { ...source.shapeIds, ...(overrides?.shapeIds ?? {}) },
    parameters: { ...source.parameters, ...(overrides?.parameters ?? {}) },
    ...(source.handleTweaks ? { handleTweaks: { ...source.handleTweaks } } : {}),
    strength: overrides?.strength ?? source.strength,
    transitionMs: overrides?.transitionMs ?? source.transitionMs,
  };
  return {
    expression: created,
    project: {
      ...project,
      expressions: [...project.expressions, created],
      activeExpressionId: created.id,
    },
  };
}

export function deleteExpressionPreset(
  project: Project,
  expressionId: string,
): Project {
  const target = project.expressions.find((e) => e.id === expressionId);
  if (!target || target.builtIn || target.id === "neutral" || project.expressions.length <= 1) {
    return project;
  }
  const expressions = project.expressions.filter((e) => e.id !== expressionId);
  const fallbackId =
    expressions.find((e) => e.id === "neutral")?.id ?? expressions[0].id;
  const activeExpressionId =
    project.activeExpressionId === expressionId
      ? fallbackId
      : project.activeExpressionId;
  const defaultExpressionId =
    project.playback.defaultExpressionId === expressionId
      ? fallbackId
      : project.playback.defaultExpressionId;
  const shortcuts: Record<string, string> = {};
  for (const [k, expId] of Object.entries(project.playback.shortcuts)) {
    if (expId !== expressionId) shortcuts[k] = expId;
  }
  return {
    ...project,
    expressions,
    activeExpressionId,
    playback: {
      ...project.playback,
      defaultExpressionId,
      shortcuts,
    },
  };
}

export function updatePlayback(
  project: Project,
  patch: Partial<PlaybackSettings>,
): Project {
  const validIds = new Set(project.expressions.map((e) => e.id));
  const defaultExpressionId =
    patch.defaultExpressionId && validIds.has(patch.defaultExpressionId)
      ? patch.defaultExpressionId
      : project.playback.defaultExpressionId;
  const blinkIntervalSec =
    typeof patch.blinkIntervalSec === "number" && Number.isFinite(patch.blinkIntervalSec)
      ? Math.max(0.5, Math.min(30, patch.blinkIntervalSec))
      : project.playback.blinkIntervalSec;
  return {
    ...project,
    playback: {
      ...project.playback,
      ...patch,
      defaultExpressionId,
      blinkIntervalSec,
    },
  };
}

export function setPartBlendMode(
  project: Project,
  partId: string,
  blendMode: BlendMode,
): Project {
  return {
    ...project,
    parts: project.parts.map((p) =>
      p.id === partId ? { ...p, blendMode } : p,
    ),
  };
}
