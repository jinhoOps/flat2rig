import { isTextEditingTarget } from "../clipboard";
import type { Tool } from "../types";

/** T11a: canvas navigation tools. They sit beside the edit `Tool` and never edit project data. */
export type NavTool = "hand" | "zoom";

export type ShortcutAction =
  | { type: "undo" }
  | { type: "redo" }
  | { type: "save" }
  | { type: "tool"; tool: Tool }
  | { type: "nav"; nav: NavTool }
  | { type: "brush"; delta: 1 | -1 }
  | { type: "togglePlay" }
  | { type: "expression"; key: string }
  | { type: "escape" };

export interface ShortcutContext {
  /** A long task is running (`task()`); nothing but the busy overlay should react. */
  busy: boolean;
  hasProject: boolean;
  hasParts: boolean;
  /** Prepare tab, or advanced tab in part/mask mode — the edit tools are live. */
  editTools: boolean;
  /** Expressions / adjust / play tab. */
  rigStage: boolean;
  modal: boolean;
}

export interface ShortcutKey {
  key: string;
  code?: string;
  ctrlKey?: boolean;
  metaKey?: boolean;
  shiftKey?: boolean;
  altKey?: boolean;
  target: EventTarget | null;
}

export const TOOL_KEYS: Record<string, Tool> = {
  v: "select",
  f: "face",
  l: "landmark",
  p: "prompt",
  b: "add",
  e: "erase",
};
export const NAV_KEYS: Record<string, NavTool> = { h: "hand", z: "zoom" };

function isSelectTarget(target: EventTarget | null): boolean {
  return (
    typeof HTMLSelectElement !== "undefined" && target instanceof HTMLSelectElement
  );
}

function isActivatableTarget(target: EventTarget | null): boolean {
  if (typeof HTMLElement === "undefined" || !(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  return tag === "BUTTON" || tag === "A" || tag === "INPUT" || target.getAttribute("role") === "checkbox";
}

/**
 * Maps a key press to a workspace action. Typing in a text field or a `<select>` never triggers a
 * shortcut (`isTextEditingTarget`), so names and numbers can be typed freely.
 */
export function resolveShortcut(e: ShortcutKey, ctx: ShortcutContext): ShortcutAction | null {
  if (ctx.busy) return null;
  if (isTextEditingTarget(e.target) || isSelectTarget(e.target)) return null;
  const key = e.key.toLowerCase();
  if (e.ctrlKey || e.metaKey) {
    if (key === "z") return e.shiftKey ? { type: "redo" } : { type: "undo" };
    if (key === "y") return { type: "redo" };
    if (key === "s") return { type: "save" };
    return null;
  }
  if (e.key === "Escape") return { type: "escape" };
  if (e.altKey) return null;
  if (ctx.rigStage && ctx.hasParts && !ctx.modal) {
    if ((e.code === "Space" || e.key === " ") && !isActivatableTarget(e.target))
      return { type: "togglePlay" };
    if (/^[1-9]$/.test(e.key)) return { type: "expression", key: e.key };
  }
  if (ctx.modal || !ctx.hasProject) return null;
  if (NAV_KEYS[key] && !e.shiftKey) return { type: "nav", nav: NAV_KEYS[key] };
  if (ctx.editTools) {
    if (TOOL_KEYS[key] && !e.shiftKey) return { type: "tool", tool: TOOL_KEYS[key] };
    if (e.key === "[") return { type: "brush", delta: -1 };
    if (e.key === "]") return { type: "brush", delta: 1 };
  }
  return null;
}

/** `[` / `]` step: finer for small brushes, coarser for large ones. Range matches the slider. */
export function stepBrushRadius(radius: number, delta: 1 | -1): number {
  const step = radius < 10 ? 1 : radius < 40 ? 2 : 5;
  return Math.max(2, Math.min(100, radius + delta * step));
}
