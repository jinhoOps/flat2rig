import { inferPartRole, type Part, type PartRole } from "../types";

/**
 * T11a: layer-panel model. Rows follow the real composite order (top of the stack first, the way a
 * layer panel reads) and cover every part in `project.parts` — including the variable hair layers
 * (`hair_front_*`, `hair_side_*`, `hair_back_*`) and wearables — so nothing depends on the fixed
 * 18-part `PART_DEFS` list. Roles come from `inferPartRole` (explicit `role` first, id prefix last).
 */

/** Which mask a brush will edit. T11a only records and shows it; T11b routes strokes by it. */
export type EditTarget = "visible" | "hidden";
export type LayerThumbKind = "texture" | "mask" | "fill";

export const ROLE_LABELS: Record<PartRole, string> = {
  hair_front: "앞머리",
  hair_side_L: "옆머리",
  hair_side_R: "옆머리",
  hair_back: "뒷머리",
  hat: "모자",
  accessory: "소품",
  hands: "손",
  body: "몸",
  clothes: "의상",
  neck: "목",
  face: "얼굴",
  eye: "눈",
  brow: "눈썹",
  mouth: "입",
  other: "기타",
};

export interface LayerRow {
  part: Part;
  role: PartRole;
  roleLabel: string;
  /** Stack position, 0 = bottom (drawn first). */
  order: number;
  hasFill: boolean;
  statusLabel: string;
  group: string;
}

const fillCache = new WeakMap<Uint8Array, boolean>();

/** Whether a part has any hidden-area (`fillMask`) pixel. Cached per mask array. */
export function partHasFill(part: Pick<Part, "fillMask">): boolean {
  const fm = part.fillMask;
  if (!fm || fm.length === 0) return false;
  const hit = fillCache.get(fm);
  if (hit !== undefined) return hit;
  let any = false;
  for (let i = 0; i < fm.length; i++) {
    if (fm[i] > 0) {
      any = true;
      break;
    }
  }
  fillCache.set(fm, any);
  return any;
}

export function statusLabel(status: Part["status"]): string {
  return status === "approved" ? "확인됨" : status === "missing" ? "미해당" : "확인 필요";
}

/** Rows top-first in composite order (ties keep array order, as the renderer does). */
export function layerRows(parts: readonly Part[]): LayerRow[] {
  return parts
    .map((part, i) => ({ part, i }))
    .sort((a, b) => (a.part.compositeOrder ?? a.i) - (b.part.compositeOrder ?? b.i))
    .map(({ part }, order) => {
      const role = inferPartRole(part.id, part.role);
      return {
        part,
        role,
        roleLabel: ROLE_LABELS[role] ?? "기타",
        order,
        hasFill: partHasFill(part),
        statusLabel: statusLabel(part.status),
        group: part.group || "기타",
      };
    })
    .reverse();
}

export interface LayerSelection {
  selected: string;
  editTarget: EditTarget;
}

/**
 * Clicking a thumbnail selects that row and picks the edit target: the texture or visible-mask
 * thumbnail means "visible", the hidden-area thumbnail means "hidden". A part without hidden pixels
 * cannot become a hidden target (its slot is empty/disabled), so the target falls back to visible.
 */
export function selectLayerThumb(
  _prev: LayerSelection,
  row: Pick<LayerRow, "part" | "hasFill">,
  kind: LayerThumbKind,
): LayerSelection {
  const editTarget: EditTarget = kind === "fill" && row.hasFill ? "hidden" : "visible";
  return { selected: row.part.id, editTarget };
}

/** Selecting a row by name/area keeps the target only if that part can still hold it. */
export function selectLayerRow(
  prev: LayerSelection,
  row: Pick<LayerRow, "part" | "hasFill">,
): LayerSelection {
  return {
    selected: row.part.id,
    editTarget: prev.editTarget === "hidden" && row.hasFill ? "hidden" : "visible",
  };
}
