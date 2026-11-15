import { useMemo, useState } from "react";
import {
  AlertCircle,
  Link2,
  RefreshCw,
  RotateCcw,
  Sparkles,
} from "lucide-react";
import {
  createDefaultShapePresets,
  DEFAULT_RIG_SAFE_BOUNDS,
  PARAM_DEFS,
  PARAM_GROUPS,
  type Mesh,
  type Parameters,
  type Project,
  type ShapeHandleValues,
  type ShapeRegion,
} from "../types";
import type { SubstituteLayerTexture } from "../rig/render";
import {
  setExpressionShape,
  updateRigProfileFields,
  updateShapePresetHandles,
} from "../rig/state";
import type { ThumbnailSpec } from "../rig/thumbnails";
import { useThumbnailCache } from "./useThumbnailCache";

type AdjustSubTab = ShapeRegion | "pose";

interface HandleControlSpec {
  key: keyof ShapeHandleValues;
  label: string;
  min: number;
  max: number;
  step: number;
}

const SHAPE_HANDLE_CONTROLS: HandleControlSpec[] = [
  { key: "width", label: "가로 폭", min: 0.65, max: 1.35, step: 0.02 },
  { key: "open", label: "열림 높이", min: 0, max: 1.25, step: 0.02 },
  { key: "roundness", label: "둥글기", min: 0, max: 1, step: 0.02 },
  { key: "corner", label: "꼬리 / 입꼬리", min: -1, max: 1, step: 0.04 },
  { key: "upperCurve", label: "윗곡률", min: -1, max: 1, step: 0.04 },
  { key: "lowerCurve", label: "아랫곡률", min: -1, max: 1, step: 0.04 },
  { key: "interiorVisibility", label: "안쪽(눈동자·입안) 노출", min: 0, max: 1, step: 0.04 },
  { key: "offsetX", label: "좌우 위치 보정", min: -0.35, max: 0.35, step: 0.02 },
  { key: "offsetY", label: "상하 위치 보정", min: -0.35, max: 0.35, step: 0.02 },
];

interface Props {
  project: Project;
  parameters: Parameters;
  onChangeParameters: (updater: (prev: Parameters) => Parameters) => void;
  partTextures: Map<string, HTMLCanvasElement>;
  meshes: Map<string, Mesh>;
  substituteTextures: Map<string, SubstituteLayerTexture>;
  busy: boolean;
  onCommit: (next: Project, message?: string) => void;
  onBeginDrag: () => void;
  onPreviewDrag: (next: Project) => void;
  onCommitDrag: () => void;
  onRegenerateShapeAssets: () => void;
  onOpenAdvancedParts: (partId?: string) => void;
}

export default function AdjustPanel({
  project,
  parameters,
  onChangeParameters,
  partTextures,
  meshes,
  substituteTextures,
  busy,
  onCommit,
  onBeginDrag,
  onPreviewDrag,
  onCommitDrag,
  onRegenerateShapeAssets,
  onOpenAdvancedParts,
}: Props) {
  const [subTab, setSubTab] = useState<AdjustSubTab>("eye_L");
  const [mirrorSync, setMirrorSync] = useState(true);
  const [paramGroupId, setParamGroupId] = useState<string>(PARAM_GROUPS[0].id);

  const activeExp =
    project.expressions.find((e) => e.id === project.activeExpressionId) ??
    project.expressions[0];

  const regionPresets = useMemo(
    () =>
      subTab === "pose"
        ? []
        : project.shapePresets.filter((s) => s.region === subTab),
    [project.shapePresets, subTab],
  );

  const activePresetId =
    subTab === "pose"
      ? undefined
      : activeExp.shapeIds[subTab] ?? project.rigProfile.defaultShapeIds[subTab];
  const activePreset =
    regionPresets.find((s) => s.id === activePresetId) ?? regionPresets[0];

  const missingRegionPart = useMemo(() => {
    if (subTab === "pose") return false;
    const targetIds =
      subTab === "eye_L"
        ? ["eyeL_white", "eyeL_upper", "eyeL_all"]
        : subTab === "eye_R"
          ? ["eyeR_white", "eyeR_upper", "eyeR_all"]
          : ["mouth_upper", "mouth_lower", "mouth_all"];
    return !project.parts.some(
      (p) => targetIds.includes(p.id) && p.status !== "missing" && p.visible,
    );
  }, [project.parts, subTab]);

  const specs = useMemo<ThumbnailSpec[]>(() => {
    if (subTab === "pose") return [];
    return regionPresets.map((preset) => ({
      id: preset.id,
      crop: subTab,
      input: {
        project,
        activeExpressionId: activeExp.id,
        overrideShapeIds: { [subTab]: preset.id },
      },
    }));
  }, [project, activeExp.id, regionPresets, subTab]);

  const thumbs = useThumbnailCache(
    subTab !== "pose" && project.parts.length > 0,
    project,
    specs,
    partTextures,
    meshes,
    substituteTextures,
    `adj:${subTab}:${activeExp.id}:${project.revision}`,
  );

  const isWideHeadBounds = (project.rigProfile.safeBounds.angleX?.max ?? 25) >= 29;

  return (
    <>
      <section>
        <div className="param-group-tabs" role="tablist" aria-label="조정 부위 선택">
          {(
            [
              { id: "eye_L", label: "왼눈 (L)" },
              { id: "eye_R", label: "오른눈 (R)" },
              { id: "mouth", label: "입 모양" },
              { id: "pose", label: "고개 · 물리" },
            ] as const
          ).map((tabItem) => (
            <button
              key={tabItem.id}
              type="button"
              role="tab"
              aria-selected={subTab === tabItem.id}
              className={`param-group-tab${subTab === tabItem.id ? " active" : ""}`}
              onClick={() => setSubTab(tabItem.id)}
            >
              {tabItem.label}
            </button>
          ))}
        </div>

        {subTab !== "pose" && (
          <>
            {subTab !== "mouth" && (
              <div className="rig-sub-banner">
                <span>
                  {subTab === "eye_L"
                    ? "캐릭터 기준 왼눈 (화면 오른쪽)"
                    : "캐릭터 기준 오른눈 (화면 왼쪽)"}
                </span>
                <label className="toggle-label">
                  <input
                    type="checkbox"
                    checked={mirrorSync}
                    onChange={(e) => setMirrorSync(e.target.checked)}
                  />
                  <Link2 size={13} />
                  좌우 대칭 연동
                </label>
              </div>
            )}

            {missingRegionPart && (
              <div className="analysis-banner status-no-face">
                <AlertCircle size={15} />
                <div>
                  <strong>해당 부위 파츠가 비어 있습니다</strong>
                  <div className="tiny">
                    고급 파츠 보정에서 눈/입 마스크를 먼저 확인해 주세요.
                  </div>
                  <button
                    className="text-button small"
                    onClick={() =>
                      onOpenAdvancedParts(
                        subTab === "mouth" ? "mouth_upper" : `${subTab.replace("_", "")}_upper`,
                      )
                    }
                  >
                    파츠 보정으로 이동 →
                  </button>
                </div>
              </div>
            )}

            <div className="section-title" style={{ marginTop: 10 }}>
              <span>01</span>
              <h2>
                {subTab === "mouth" ? "입 모양" : "눈 모양"} ({activeExp.name} 표정에 적용)
              </h2>
            </div>

            <div className="rig-card-grid">
              {regionPresets.map((preset) => {
                const isSelected = preset.id === activePreset?.id;
                const thumb = thumbs[preset.id];
                return (
                  <button
                    key={preset.id}
                    type="button"
                    className={`rig-card${isSelected ? " active" : ""}`}
                    disabled={busy}
                    onClick={() => {
                      let next = setExpressionShape(
                        project,
                        subTab,
                        preset.id,
                        activeExp.id,
                        true,
                      );
                      if (subTab !== "mouth" && mirrorSync && preset.mirrorLinked) {
                        const other: ShapeRegion = subTab === "eye_L" ? "eye_R" : "eye_L";
                        const mirrorId = `${other}_${preset.kind}`;
                        if (next.shapePresets.some((s) => s.id === mirrorId)) {
                          next = setExpressionShape(
                            next,
                            other,
                            mirrorId,
                            activeExp.id,
                            true,
                          );
                        }
                      }
                      onCommit(next, `'${preset.label}' 모양을 적용했습니다.`);
                    }}
                  >
                    <div className="rig-card-thumb">
                      {thumb ? (
                        <img src={thumb} alt={preset.label} />
                      ) : (
                        <span className="rig-thumb-placeholder">…</span>
                      )}
                    </div>
                    <strong>{preset.label}</strong>
                  </button>
                );
              })}
            </div>

            <div className="two-buttons compact" style={{ marginTop: 10 }}>
              <button disabled={busy} onClick={onRegenerateShapeAssets}>
                <RefreshCw size={13} />
                캐릭터 대체 파츠 재합성
              </button>
              {activePreset && (
                <button
                  disabled={busy}
                  onClick={() => {
                    const defs = createDefaultShapePresets(project.rigProfile.proportionMode);
                    const def = defs.find((d) => d.id === activePreset.id);
                    if (!def) return;
                    onCommit(
                      updateShapePresetHandles(
                        project,
                        activePreset.id,
                        def.handles,
                        mirrorSync,
                      ),
                      `'${activePreset.label}' 핸들을 기본값으로 초기화했습니다.`,
                    );
                  }}
                >
                  <RotateCcw size={13} />
                  핸들 초기화
                </button>
              )}
            </div>
          </>
        )}
      </section>

      {subTab !== "pose" && activePreset && (
        <section>
          <div className="section-title">
            <span>02</span>
            <h2>형상 핸들 조정 · {activePreset.label}</h2>
          </div>
          {SHAPE_HANDLE_CONTROLS.map((ctrl) => {
            const val = activePreset.handles[ctrl.key];
            return (
              <div className="param-row" key={ctrl.key}>
                <div className="param-head">
                  <label htmlFor={`handle-${ctrl.key}`}>{ctrl.label}</label>
                  <span className="param-val">{val.toFixed(2)}</span>
                </div>
                <input
                  id={`handle-${ctrl.key}`}
                  aria-label={ctrl.label}
                  type="range"
                  min={ctrl.min}
                  max={ctrl.max}
                  step={ctrl.step}
                  disabled={busy}
                  value={val}
                  onPointerDown={onBeginDrag}
                  onChange={(e) => {
                    onBeginDrag();
                    onPreviewDrag(
                      updateShapePresetHandles(
                        project,
                        activePreset.id,
                        { [ctrl.key]: +e.target.value },
                        mirrorSync,
                      ),
                    );
                  }}
                  onPointerUp={onCommitDrag}
                  onBlur={onCommitDrag}
                />
              </div>
            );
          })}
        </section>
      )}

      {subTab === "pose" && (
        <>
          <section>
            <div className="section-title">
              <span>01</span>
              <h2>고개 · 시선 · 물리 강도</h2>
            </div>

            <div className="two-buttons compact" style={{ marginBottom: 10 }}>
              <button
                className={!isWideHeadBounds ? "active" : ""}
                disabled={busy}
                onClick={() =>
                  onCommit(
                    updateRigProfileFields(project, {
                      safeBounds: {
                        ...project.rigProfile.safeBounds,
                        angleX: { min: -25, max: 25 },
                        angleY: { min: -25, max: 25 },
                        angleZ: { min: -25, max: 25 },
                      },
                    }),
                    "고개 회전 안전 범위(±25°)를 적용했습니다.",
                  )
                }
              >
                안정 범위 (±25°)
              </button>
              <button
                className={isWideHeadBounds ? "active" : ""}
                disabled={busy}
                onClick={() =>
                  onCommit(
                    updateRigProfileFields(project, {
                      safeBounds: {
                        ...DEFAULT_RIG_SAFE_BOUNDS,
                        angleX: { min: -30, max: 30 },
                        angleY: { min: -30, max: 30 },
                        angleZ: { min: -30, max: 30 },
                      },
                    }),
                    "고개 회전 최대 범위(±30°)를 허용했습니다.",
                  )
                }
              >
                넓은 범위 (±30°)
              </button>
            </div>

            {(
              [
                ["head", "고개 움직임 배율"],
                ["eyes", "눈 · 눈썹 배율"],
                ["mouth", "입 변형 배율"],
                ["hair", "머리카락 흔들림 배율"],
                ["body", "몸통 · 호흡 배율"],
              ] as const
            ).map(([k, label]) => (
              <div className="param-row" key={k}>
                <div className="param-head">
                  <label htmlFor={`scale-${k}`}>{label}</label>
                  <span className="param-val">
                    {Math.round(project.rigProfile.motionScale[k] * 100)}%
                  </span>
                </div>
                <input
                  id={`scale-${k}`}
                  aria-label={label}
                  type="range"
                  min="0"
                  max="1.5"
                  step="0.05"
                  disabled={busy}
                  value={project.rigProfile.motionScale[k]}
                  onPointerDown={onBeginDrag}
                  onChange={(e) => {
                    onBeginDrag();
                    onPreviewDrag(
                      updateRigProfileFields(project, {
                        motionScale: {
                          ...project.rigProfile.motionScale,
                          [k]: +e.target.value,
                        },
                      }),
                    );
                  }}
                  onPointerUp={onCommitDrag}
                  onBlur={onCommitDrag}
                />
              </div>
            ))}
          </section>

          <section>
            <div className="section-title">
              <span>02</span>
              <h2>실시간 파라미터 세부 조정</h2>
            </div>
            <div className="param-group-tabs" role="tablist">
              {PARAM_GROUPS.map((grp) => (
                <button
                  key={grp.id}
                  type="button"
                  role="tab"
                  aria-selected={paramGroupId === grp.id}
                  className={`param-group-tab${paramGroupId === grp.id ? " active" : ""}`}
                  onClick={() => setParamGroupId(grp.id)}
                >
                  {grp.label}
                </button>
              ))}
            </div>
            {PARAM_DEFS.filter((d) => d.groupId === paramGroupId).map((def) => {
              const step = def.max > 1 ? 0.5 : 0.01;
              return (
                <div className="param-row" key={def.key}>
                  <div className="param-head">
                    <label htmlFor={`param-${def.key}`}>{def.name}</label>
                    <span className="param-val">
                      {parameters[def.key].toFixed(def.max > 1 ? 0 : 2)}
                    </span>
                  </div>
                  <input
                    id={`param-${def.key}`}
                    aria-label={def.name}
                    type="range"
                    min={def.previewMin}
                    max={def.previewMax}
                    step={step}
                    value={parameters[def.key]}
                    onChange={(e) =>
                      onChangeParameters((prev) => ({
                        ...prev,
                        [def.key]: +e.target.value,
                      }))
                    }
                  />
                </div>
              );
            })}
            <div className="note" style={{ marginTop: 8 }}>
              <Sparkles size={13} />
              <span>
                이 슬라이더 값은 임시 확인용이며 &apos;표정&apos; 탭에서 기본 자세나 새
                표정으로 저장할 수 있습니다.
              </span>
            </div>
          </section>
        </>
      )}
    </>
  );
}
