import {
  Crosshair,
  Eraser,
  Hand,
  MousePointer2,
  Move,
  Paintbrush,
  ScanFace,
  ZoomIn,
  type LucideIcon,
} from "lucide-react";
import type { Tool } from "../types";
import type { NavTool } from "./shortcuts";

export interface ToolDef<K extends string> {
  key: K;
  label: string;
  shortcut: string;
  icon: LucideIcon;
}

/** Edit tools — the existing `Tool` values, unchanged. */
export const EDIT_TOOLS: readonly ToolDef<Tool>[] = [
  { key: "select", label: "선택", shortcut: "V", icon: MousePointer2 },
  { key: "face", label: "얼굴 영역 지정", shortcut: "F", icon: ScanFace },
  { key: "landmark", label: "랜드마크 이동", shortcut: "L", icon: Move },
  { key: "prompt", label: "클릭 보정", shortcut: "P", icon: Crosshair },
  { key: "add", label: "마스크 추가 브러시", shortcut: "B", icon: Paintbrush },
  { key: "erase", label: "마스크 지우개", shortcut: "E", icon: Eraser },
];

export const NAV_TOOLS: readonly ToolDef<NavTool>[] = [
  { key: "hand", label: "손 (캔버스 이동)", shortcut: "H", icon: Hand },
  { key: "zoom", label: "돋보기 (클릭 확대 · Alt+클릭 축소)", shortcut: "Z", icon: ZoomIn },
];

export interface ToolboxProps {
  tool: Tool;
  navTool: NavTool | null;
  /** Edit tools are live (prepare / advanced part mode with a project). */
  editEnabled: boolean;
  hasParts: boolean;
  hasProject: boolean;
  busy: boolean;
  onTool: (t: Tool) => void;
  onNav: (t: NavTool) => void;
}

export default function Toolbox(props: ToolboxProps) {
  const { tool, navTool, editEnabled, hasParts, hasProject, busy } = props;
  return (
    <nav className="ws-toolbox" aria-label="도구 모음">
      <div className="tb-group" role="toolbar" aria-orientation="vertical" aria-label="편집 도구">
        {EDIT_TOOLS.map((t) => {
          const needsParts = t.key === "add" || t.key === "erase" || t.key === "prompt";
          const active = editEnabled && !navTool && tool === t.key;
          return (
            <button
              type="button"
              key={t.key}
              className={`tb-tool${active ? " is-active" : ""}`}
              aria-label={`${t.label} (${t.shortcut})`}
              aria-pressed={active}
              title={`${t.label} (${t.shortcut})`}
              data-tool={t.key}
              disabled={!hasProject || !editEnabled || busy || (needsParts && !hasParts)}
              onClick={() => props.onTool(t.key)}
            >
              <t.icon size={17} />
            </button>
          );
        })}
      </div>
      <span className="tb-sep" />
      <div className="tb-group" role="toolbar" aria-orientation="vertical" aria-label="보기 도구">
        {NAV_TOOLS.map((t) => (
          <button
            type="button"
            key={t.key}
            className={`tb-tool${navTool === t.key ? " is-active" : ""}`}
            aria-label={`${t.label} (${t.shortcut})`}
            aria-pressed={navTool === t.key}
            title={`${t.label} (${t.shortcut})`}
            data-tool={t.key}
            disabled={!hasProject || busy}
            onClick={() => props.onNav(t.key)}
          >
            <t.icon size={17} />
          </button>
        ))}
      </div>
    </nav>
  );
}
