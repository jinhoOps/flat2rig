import { useMemo, useState } from "react";
import {
  AlertCircle,
  Copy,
  Pencil,
  Plus,
  RotateCcw,
  Save,
  Trash2,
} from "lucide-react";
import type {
  Mesh,
  Parameters,
  Project,
} from "../types";
import type { SubstituteLayerTexture } from "../rig/render";
import {
  createCustomExpression,
  deleteExpressionPreset,
  saveNeutralParameters,
  selectExpressionPreset,
  updateExpressionPreset,
} from "../rig/state";
import type { ThumbnailSpec } from "../rig/thumbnails";
import { useThumbnailCache } from "./useThumbnailCache";

interface Props {
  project: Project;
  parameters: Parameters;
  partTextures: Map<string, HTMLCanvasElement>;
  meshes: Map<string, Mesh>;
  substituteTextures: Map<string, SubstituteLayerTexture>;
  busy: boolean;
  onCommit: (next: Project, message?: string) => void;
  onBeginDrag: () => void;
  onPreviewDrag: (next: Project) => void;
  onCommitDrag: () => void;
  onResetPoseToNeutral: () => void;
  onOpenAdvancedParts: (partId?: string) => void;
}

export default function ExpressionPanel({
  project,
  parameters,
  partTextures,
  meshes,
  substituteTextures,
  busy,
  onCommit,
  onBeginDrag,
  onPreviewDrag,
  onCommitDrag,
  onResetPoseToNeutral,
  onOpenAdvancedParts,
}: Props) {
  const [newName, setNewName] = useState("");
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameText, setRenameText] = useState("");

  const activeExp =
    project.expressions.find((e) => e.id === project.activeExpressionId) ??
    project.expressions[0];

  const missingCriticalParts = useMemo(() => {
    const needed = [
      { id: "eyeL_white", label: "왼눈 흰자/눈매" },
      { id: "eyeR_white", label: "오른눈 흰자/눈매" },
      { id: "mouth_upper", label: "윗입술" },
    ];
    return needed.filter((item) => {
      const p = project.parts.find((t) => t.id === item.id);
      return !p || p.status === "missing" || !p.visible;
    });
  }, [project.parts]);

  const specs = useMemo<ThumbnailSpec[]>(
    () =>
      project.expressions.map((exp) => ({
        id: exp.id,
        crop: "face",
        input: {
          project,
          activeExpressionId: exp.id,
        },
      })),
    [project],
  );

  const thumbs = useThumbnailCache(
    project.parts.length > 0,
    project,
    specs,
    partTextures,
    meshes,
    substituteTextures,
    `expr:${project.revision}:${project.expressions.map((e) => e.id).join(",")}`,
  );

  const handleSaveCustom = () => {
    const label = newName.trim() || `${activeExp.name} 변형`;
    const baseOffsets: Partial<Parameters> = {};
    for (const k of Object.keys(parameters) as (keyof Parameters)[]) {
      const neutralVal = project.rigProfile.neutralParameters[k];
      if (Math.abs(parameters[k] - neutralVal) > 1e-4) {
        baseOffsets[k] = parameters[k];
      }
    }
    const { project: next, expression } = createCustomExpression(
      project,
      label,
      activeExp.id,
      {
        parameters: { ...activeExp.parameters, ...baseOffsets },
      },
    );
    setNewName("");
    onCommit(next, `새 표정 '${expression.name}'을 저장했습니다.`);
  };

  const handleDuplicate = () => {
    const { project: next, expression } = createCustomExpression(
      project,
      `${activeExp.name} 복사본`,
      activeExp.id,
    );
    onCommit(next, `표정 '${expression.name}'을 복제했습니다.`);
  };

  return (
    <>
      {missingCriticalParts.length > 0 && (
        <section>
          <div className="analysis-banner status-no-face">
            <AlertCircle size={15} />
            <div>
              <strong>누락·숨김 파츠 확인 필요</strong>
              <div className="tiny">
                {missingCriticalParts.map((m) => m.label).join(", ")} 파츠가 비어 있거나
                숨겨져 있어 표정이 불완전할 수 있습니다.
              </div>
              <button
                className="text-button small"
                disabled={busy}
                onClick={() => onOpenAdvancedParts(missingCriticalParts[0].id)}
              >
                고급 파츠 보정에서 확인 →
              </button>
            </div>
          </div>
        </section>
      )}

      <section>
        <div className="section-title">
          <span>01</span>
          <h2>표정 프리셋 ({project.expressions.length})</h2>
        </div>
        <div className="rig-card-grid">
          {project.expressions.map((exp, idx) => {
            const isSelected = exp.id === activeExp.id;
            const shortcut = exp.shortcutKey ?? (idx < 9 ? String(idx + 1) : undefined);
            const thumb = thumbs[exp.id];
            return (
              <button
                key={exp.id}
                type="button"
                className={`rig-card${isSelected ? " active" : ""}`}
                disabled={busy}
                onClick={() =>
                  onCommit(
                    selectExpressionPreset(project, exp.id),
                    `표정 '${exp.name}' 선택`,
                  )
                }
              >
                <div className="rig-card-thumb">
                  {thumb ? (
                    <img src={thumb} alt={exp.name} />
                  ) : (
                    <span className="rig-thumb-placeholder">…</span>
                  )}
                  {shortcut && <span className="rig-shortcut-badge">{shortcut}</span>}
                </div>
                <strong>{exp.name}</strong>
              </button>
            );
          })}
        </div>
      </section>

      <section>
        <div className="section-title">
          <span>02</span>
          <h2>선택 표정 설정 · {activeExp.name}</h2>
        </div>

        <div className="param-row">
          <div className="param-head">
            <label htmlFor="expr-strength">표정 강도</label>
            <span className="param-val">{Math.round(activeExp.strength * 100)}%</span>
          </div>
          <input
            id="expr-strength"
            aria-label="표정 강도"
            type="range"
            min="0"
            max="1.5"
            step="0.05"
            disabled={busy}
            value={activeExp.strength}
            onPointerDown={onBeginDrag}
            onChange={(e) => {
              onBeginDrag();
              onPreviewDrag(
                updateExpressionPreset(project, activeExp.id, {
                  strength: +e.target.value,
                }),
              );
            }}
            onPointerUp={onCommitDrag}
            onBlur={onCommitDrag}
          />
          <div className="range-ends">
            <span>0%</span>
            <span>100%</span>
            <span>150%</span>
          </div>
        </div>

        <div className="param-row">
          <div className="param-head">
            <label htmlFor="expr-transition">전환 시간</label>
            <span className="param-val">{activeExp.transitionMs}ms</span>
          </div>
          <input
            id="expr-transition"
            aria-label="표정 전환 시간"
            type="range"
            min="0"
            max="600"
            step="20"
            disabled={busy}
            value={activeExp.transitionMs}
            onPointerDown={onBeginDrag}
            onChange={(e) => {
              onBeginDrag();
              onPreviewDrag(
                updateExpressionPreset(project, activeExp.id, {
                  transitionMs: +e.target.value,
                }),
              );
            }}
            onPointerUp={onCommitDrag}
            onBlur={onCommitDrag}
          />
          <div className="range-ends">
            <span>즉시 (0ms)</span>
            <span>600ms</span>
          </div>
        </div>

        {renamingId === activeExp.id ? (
          <div className="rig-inline-form">
            <input
              aria-label="새 표정 이름"
              type="text"
              value={renameText}
              onChange={(e) => setRenameText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && renameText.trim()) {
                  onCommit(
                    updateExpressionPreset(project, activeExp.id, {
                      name: renameText.trim(),
                    }),
                    "표정 이름을 변경했습니다.",
                  );
                  setRenamingId(null);
                }
              }}
            />
            <button
              className="primary"
              disabled={!renameText.trim()}
              onClick={() => {
                onCommit(
                  updateExpressionPreset(project, activeExp.id, {
                    name: renameText.trim(),
                  }),
                  "표정 이름을 변경했습니다.",
                );
                setRenamingId(null);
              }}
            >
              확인
            </button>
          </div>
        ) : null}

        <div className="three-buttons compact" style={{ marginTop: 8 }}>
          <button disabled={busy} onClick={handleDuplicate}>
            <Copy size={13} />
            복제
          </button>
          <button
            disabled={busy}
            onClick={() => {
              setRenamingId(activeExp.id);
              setRenameText(activeExp.name);
            }}
          >
            <Pencil size={13} />
            이름 변경
          </button>
          <button
            disabled={busy || activeExp.builtIn || activeExp.id === "neutral"}
            title={
              activeExp.id === "neutral" || activeExp.builtIn
                ? "기본 내장 표정은 삭제할 수 없습니다."
                : "사용자 표정 삭제"
            }
            onClick={() =>
              onCommit(
                deleteExpressionPreset(project, activeExp.id),
                `표정 '${activeExp.name}'을 삭제했습니다.`,
              )
            }
          >
            <Trash2 size={13} />
            삭제
          </button>
        </div>

        <div className="rig-save-box">
          <label className="field-label" htmlFor="new-expr-name">
            현재 눈·입·포즈 상태로 새 표정 저장
          </label>
          <div className="rig-inline-form">
            <input
              id="new-expr-name"
              type="text"
              placeholder="새 표정 이름 (예: 반짝 웃음)"
              value={newName}
              disabled={busy}
              onChange={(e) => setNewName(e.target.value)}
            />
            <button className="primary" disabled={busy} onClick={handleSaveCustom}>
              <Plus size={14} />
              저장
            </button>
          </div>
        </div>

        <div className="two-buttons compact" style={{ marginTop: 10 }}>
          <button disabled={busy} onClick={onResetPoseToNeutral}>
            <RotateCcw size={13} />
            임시 슬라이더 초기화
          </button>
          <button
            disabled={busy}
            onClick={() =>
              onCommit(
                saveNeutralParameters(project, parameters),
                "현재 자세를 캐릭터 기본(중립) 자세로 저장했습니다.",
              )
            }
          >
            <Save size={13} />
            기본 자세로 저장
          </button>
        </div>
      </section>
    </>
  );
}
