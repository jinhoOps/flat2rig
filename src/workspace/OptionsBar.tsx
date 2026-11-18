import type { ReactNode } from "react";
import {
  Eraser,
  Maximize,
  Paintbrush,
  Redo2,
  Undo2,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import type { Tool } from "../types";
import type { EditTarget } from "./layers";
import type { NavTool } from "./shortcuts";
import { EDIT_TOOLS, NAV_TOOLS } from "./Toolbox";

export type WorkTab = "prepare" | "expressions" | "adjust" | "play" | "advanced";

/** The one-line canvas hint (was under the canvas); now leads the options bar. */
export function canvasHint(tab: WorkTab, preview: boolean, tool: Tool, navTool: NavTool | null): string {
  if (navTool === "hand") return "드래그: 캔버스 이동 · 가운데 버튼 드래그도 이동";
  if (navTool === "zoom") return "클릭: 확대 · Alt+클릭: 축소 · 배율 버튼: 화면 맞춤";
  if (tab === "play") return "Space: 재생/일시정지 · 1~9 키: 표정 빠른 전환";
  if (tab === "expressions") return "우측 표정 카드를 클릭하거나 1~9 키로 즉시 전환할 수 있습니다";
  if (tab === "adjust") return "왼쪽/오른쪽 눈·입 모양을 고르고 슬라이더로 각도를 다듬으세요";
  if (preview) return "고급 메쉬/변형 확인 모드입니다";
  if (tool === "face") return "얼굴 피부 영역의 좌상단에서 우하단까지 드래그하세요 · Space+드래그: 이동";
  if (tool === "landmark") return "눈 · 입 · 턱 · 얼굴 상단의 점을 드래그하세요 · Space+드래그: 이동";
  if (tool === "prompt") return "좌클릭: 포함(+) · 우클릭/Alt/Shift+클릭: 제외(-) · Space+드래그: 이동";
  if (tool === "add" || tool === "erase") return "드래그하여 마스크 수정 · [ ]: 브러시 크기 · Space+드래그: 캔버스 이동";
  return "파츠를 선택하여 경계를 확인하세요 · Space+드래그: 이동";
}

/**
 * T11a slot for T11b: shows which mask a brush stroke is meant for. T11a records the choice (layer
 * thumbnails set it) but strokes still edit the visible `mask`; the hidden target says so plainly.
 */
export function EditTargetIndicator({ target }: { target: EditTarget }) {
  return (
    <span className={`ob-target is-${target}`} data-edit-target={target} title="레이어 패널의 가시/숨은 썸네일을 눌러 바꿉니다">
      <span className="ob-k">편집 대상</span>
      <span className={`ob-chip${target === "visible" ? " is-on" : ""}`}>가시</span>
      <span className={`ob-chip${target === "hidden" ? " is-on" : ""}`}>숨은</span>
      {target === "hidden" && (
        <span className="ob-warn">숨은 영역 붓질은 T11b 예정 · 지금은 가시 mask에 적용</span>
      )}
    </span>
  );
}

export interface ViewToggle {
  key: string;
  label: string;
  checked: boolean;
  disabled: boolean;
  onChange: (v: boolean) => void;
}

export interface OptionsBarProps {
  tab: WorkTab;
  tool: Tool;
  navTool: NavTool | null;
  preview: boolean;
  isRigStage: boolean;
  hasProject: boolean;
  busy: boolean;
  radius: number;
  onRadius: (r: number) => void;
  onTool: (t: Tool) => void;
  editTarget: EditTarget;
  /** Replaces the default target indicator (T11b). */
  targetSlot?: ReactNode;
  selectedName?: string;
  selectedColor?: string;
  canUndo: boolean;
  canRedo: boolean;
  onUndo: () => void;
  onRedo: () => void;
  toggles: ViewToggle[];
  zoom: number;
  onZoomOut: () => void;
  onZoomIn: () => void;
  onFit: () => void;
}

export default function OptionsBar(props: OptionsBarProps) {
  const { tool, navTool, preview, isRigStage, busy } = props;
  const current =
    (navTool && NAV_TOOLS.find((t) => t.key === navTool)) ||
    (!preview && EDIT_TOOLS.find((t) => t.key === tool)) ||
    null;
  const brush = !navTool && !preview && (tool === "add" || tool === "erase");
  return (
    <div className="ws-options" role="toolbar" aria-label="도구 옵션">
      <span className="ob-tool" title={current ? `${current.label} (${current.shortcut})` : undefined}>
        {current ? <current.icon size={15} /> : null}
        <span>{current ? current.label : isRigStage ? "미리보기" : "보기"}</span>
      </span>
      {brush && (
        <>
          <span className="ob-sep" />
          <span className="ob-seg" role="group" aria-label="브러시 모드">
            <button
              type="button"
              className={tool === "add" ? "is-on" : ""}
              aria-pressed={tool === "add"}
              disabled={busy}
              onClick={() => props.onTool("add")}
              title="추가 (B)"
            >
              <Paintbrush size={13} />
              추가
            </button>
            <button
              type="button"
              className={tool === "erase" ? "is-on" : ""}
              aria-pressed={tool === "erase"}
              disabled={busy}
              onClick={() => props.onTool("erase")}
              title="지우기 (E)"
            >
              <Eraser size={13} />
              지우기
            </button>
          </span>
          <label className="ob-range">
            <span>크기</span>
            <input
              aria-label="브러시 크기"
              type="range"
              min="2"
              max="100"
              value={props.radius}
              disabled={busy}
              onChange={(e) => props.onRadius(+e.target.value)}
            />
            <output>{props.radius}px</output>
          </label>
        </>
      )}
      {!isRigStage && props.hasProject && !navTool && !preview && (
        <>
          <span className="ob-sep" />
          {props.targetSlot ?? <EditTargetIndicator target={props.editTarget} />}
        </>
      )}
      {props.selectedName && !isRigStage && (
        <span className="ob-part" title="선택 파츠">
          <i style={{ background: props.selectedColor }} />
          {props.selectedName}
        </span>
      )}
      <span className="ob-hint">{canvasHint(props.tab, preview, tool, navTool)}</span>
      <span className="ob-end">
        {!isRigStage && props.toggles.length > 0 && (
          <span className="ob-toggles">
            {props.toggles.map((t) => (
              <label key={t.key} className="toggle-label">
                <input
                  type="checkbox"
                  checked={t.checked}
                  disabled={t.disabled}
                  onChange={(e) => t.onChange(e.target.checked)}
                />
                {t.label}
              </label>
            ))}
            <span className="ob-sep" />
          </span>
        )}
        <button
          type="button"
          className="icon-button"
          aria-label="실행 취소"
          title="실행 취소 (Ctrl+Z)"
          disabled={!props.canUndo || busy}
          onClick={props.onUndo}
        >
          <Undo2 size={15} />
        </button>
        <button
          type="button"
          className="icon-button"
          aria-label="다시 실행"
          title="다시 실행 (Ctrl+Shift+Z)"
          disabled={!props.canRedo || busy}
          onClick={props.onRedo}
        >
          <Redo2 size={15} />
        </button>
        <span className="ob-sep" />
        <button type="button" className="icon-button" aria-label="축소" title="축소" disabled={!props.hasProject} onClick={props.onZoomOut}>
          <ZoomOut size={15} />
        </button>
        <button type="button" className="ob-zoom" aria-label="화면에 맞춤" title="화면에 맞춤" disabled={!props.hasProject} onClick={props.onFit}>
          {Math.round(props.zoom * 100)}%
        </button>
        <button type="button" className="icon-button" aria-label="확대" title="확대" disabled={!props.hasProject} onClick={props.onZoomIn}>
          <ZoomIn size={15} />
        </button>
        <button type="button" className="icon-button ob-fit" aria-label="맞춤" title="맞춤 (100%)" disabled={!props.hasProject} onClick={props.onFit}>
          <Maximize size={14} />
        </button>
      </span>
    </div>
  );
}
