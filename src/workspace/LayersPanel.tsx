import { Fragment, useEffect, useRef, useState } from "react";
import {
  Box,
  CheckCircle2,
  ChevronDown,
  Eye,
  EyeOff,
  FolderOpen,
  ImagePlus,
  Layers,
  Lock,
  Plus,
} from "lucide-react";
import type { BlendMode, HairRole, Part } from "../types";
import { isPreservedPart } from "../io";
import LayerThumb from "./LayerThumb";
import type { EditTarget, LayerRow, LayerThumbKind } from "./layers";

export type DynamicLayerKind =
  | HairRole
  | "hat"
  | "clothes_outer"
  | "acc_glasses"
  | "acc_hairpin";

export const DYNAMIC_LAYER_KINDS: readonly (readonly [DynamicLayerKind, string])[] = [
  ["hair_front", "앞머리"],
  ["hair_side_R", "오른옆머리"],
  ["hair_side_L", "왼옆머리"],
  ["hair_back", "뒷머리"],
  ["hat", "모자"],
  ["clothes_outer", "의상"],
  ["acc_glasses", "안경"],
  ["acc_hairpin", "악세서리"],
];

const PROVENANCE: Record<string, string> = {
  model: "모델",
  prompted: "클릭 보정",
  manual: "수동",
};

function diagnosticShort(d: string): string {
  if (d === "empty") return "공백";
  if (d === "lr_swapped") return "좌우반전";
  if (d === "low_confidence") return "저신뢰";
  if (d.startsWith("fragmented")) return "파편화";
  if (d.startsWith("background_leak")) return "배경누출";
  return d;
}

/** Only `normal` is exported to PSD (AGENTS.md §5–6); other modes are flagged, never hidden. */
export function blendDisplay(mode: BlendMode | undefined): { label: string; supported: boolean } {
  const m = mode ?? "normal";
  if (m === "normal") return { label: "일반 (Normal)", supported: true };
  return {
    label: `${m === "multiply" ? "곱하기" : "가산"} · PSD 미지원`,
    supported: false,
  };
}

export interface LayersPanelProps {
  rows: LayerRow[];
  width: number;
  height: number;
  selected: string;
  editTarget: EditTarget;
  busy: boolean;
  hasProject: boolean;
  partTextures: Map<string, HTMLCanvasElement>;
  approved: number;
  total: number;
  onSelectRow: (row: LayerRow) => void;
  onSelectThumb: (row: LayerRow, kind: LayerThumbKind) => void;
  onToggleVisible: (id: string) => void;
  onToggleGroup: (group: string) => void;
  onOpenImage: () => void;
  onOpenProject: () => void;
  onAddPart: () => void;
  onAddDynamic: (kind: DynamicLayerKind) => void;
}

function rowSubtitle(row: LayerRow): string {
  const p = row.part;
  const bits = [row.roleLabel, row.statusLabel, PROVENANCE[p.provenance ?? ""] ?? "기하"];
  if (p.diagnostics?.length) bits.push(`⚠${diagnosticShort(p.diagnostics[0])}`);
  return bits.join(" · ");
}

export default function LayersPanel(props: LayersPanelProps) {
  const {
    rows,
    width,
    height,
    selected,
    editTarget,
    busy,
    hasProject,
    partTextures,
    approved,
    total,
  } = props;
  const [addOpen, setAddOpen] = useState(false);
  const addRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!addOpen) return;
    const close = (e: MouseEvent) => {
      if (!addRef.current?.contains(e.target as Node)) setAddOpen(false);
    };
    window.addEventListener("mousedown", close);
    return () => window.removeEventListener("mousedown", close);
  }, [addOpen]);
  const selectedPart: Part | undefined = rows.find((r) => r.part.id === selected)?.part;
  const blend = blendDisplay(selectedPart?.blendMode);
  const groupStats = new Map<string, { all: boolean; any: boolean; count: number }>();
  for (const r of rows) {
    const g = groupStats.get(r.group) ?? { all: true, any: false, count: 0 };
    g.count++;
    if (r.part.visible) g.any = true;
    else g.all = false;
    groupStats.set(r.group, g);
  }
  return (
    <section className="lp" aria-label="레이어">
      <div className="lp-options">
        <span className="lp-blend" title="레이어 합성 모드 — PSD 내보내기는 일반(Normal)만 지원">
          <span className="lp-k">합성</span>
          <span className={blend.supported ? "lp-v" : "lp-v is-warn"}>{blend.label}</span>
        </span>
        <span className="lp-count" title="파츠 레이어 수">
          <Layers size={12} />
          {total}
        </span>
      </div>
      <div className="lp-colhead" aria-hidden="true">
        <span>표시</span>
        <span>텍스처</span>
        <span>가시</span>
        <span>숨은</span>
        <span>이름</span>
      </div>
      <div className="lp-list" role="list">
        {!rows.length ? (
          <div className="layer-empty">
            <Box size={25} />
            <span>파츠를 준비해 보세요</span>
            <p>
              이미지를 열고 얼굴 위치를 맞추면
              <br />
              분리 초안을 만들 수 있습니다.
            </p>
          </div>
        ) : (
          rows.map((row, i) => {
            const p = row.part;
            const isSel = p.id === selected;
            const showGroup = i === 0 || rows[i - 1].group !== row.group;
            const g = groupStats.get(row.group)!;
            const target = isSel ? editTarget : null;
            const thumb = (kind: LayerThumbKind, label: string, children: React.ReactNode, disabled = false) => (
              <button
                type="button"
                className={`lp-thumb lp-thumb-${kind}${
                  target === (kind === "fill" ? "hidden" : "visible") && kind !== "texture"
                    ? " is-target"
                    : ""
                }${disabled ? " is-empty" : ""}`}
                aria-label={`${p.name} ${label}`}
                aria-pressed={kind === "texture" ? undefined : target === (kind === "fill" ? "hidden" : "visible")}
                title={
                  disabled
                    ? `${label} 없음`
                    : kind === "fill"
                      ? `${label} — 클릭: 편집 대상을 숨은 영역으로 (붓질 연결은 T11b)`
                      : kind === "mask"
                        ? `${label} — 클릭: 편집 대상을 가시 영역으로`
                        : label
                }
                disabled={busy || disabled}
                onClick={(e) => {
                  e.stopPropagation();
                  props.onSelectThumb(row, kind);
                }}
              >
                {children}
              </button>
            );
            return (
              <Fragment key={p.id}>
                {showGroup && (
                  <div className="lp-group" role="listitem">
                    <button
                      type="button"
                      className={`lp-eye${g.any && !g.all ? " is-partial" : !g.any ? " is-hidden" : ""}`}
                      role="checkbox"
                      aria-checked={g.all ? true : g.any ? "mixed" : false}
                      aria-label={`${row.group} 그룹 일괄 표시`}
                      title={`${row.group} 그룹 일괄 표시/숨김`}
                      disabled={busy}
                      onClick={() => props.onToggleGroup(row.group)}
                    >
                      {g.any ? <Eye size={13} /> : <EyeOff size={13} />}
                    </button>
                    <span className="lp-group-name">{row.group}</span>
                    <span className="lp-group-count">{g.count}</span>
                  </div>
                )}
                <div
                  role="listitem"
                  className={`lp-row${isSel ? " is-selected" : ""}${p.status === "missing" ? " is-missing" : ""}${!p.visible ? " is-off" : ""}`}
                  data-part-id={p.id}
                  data-role={row.role}
                  data-order={row.order}
                >
                  <button
                    type="button"
                    className="lp-eye"
                    role="checkbox"
                    aria-checked={p.visible}
                    aria-label={`${p.name} 표시`}
                    disabled={busy}
                    onClick={() => props.onToggleVisible(p.id)}
                  >
                    {p.visible ? <Eye size={13} /> : <EyeOff size={13} />}
                  </button>
                  {thumb(
                    "texture",
                    "텍스처",
                    partTextures.get(p.id) ? (
                      <LayerThumb width={width} height={height} image={partTextures.get(p.id)} version={p} />
                    ) : (
                      <i className="lp-swatch" style={{ background: p.color }} />
                    ),
                  )}
                  {thumb(
                    "mask",
                    "가시 영역 마스크",
                    <LayerThumb width={width} height={height} mask={p.mask} />,
                  )}
                  {thumb(
                    "fill",
                    "숨은 영역 마스크",
                    row.hasFill ? (
                      <LayerThumb width={width} height={height} mask={p.fillMask} />
                    ) : (
                      <i className="lp-none" aria-hidden="true" />
                    ),
                    !row.hasFill,
                  )}
                  <button
                    type="button"
                    className="lp-name"
                    aria-current={isSel ? "true" : undefined}
                    disabled={busy}
                    onClick={() => props.onSelectRow(row)}
                    title={`${p.name} (${p.id})`}
                  >
                    <strong>
                      <i className="lp-color" style={{ background: p.color }} />
                      {p.name}
                    </strong>
                    <small>{rowSubtitle(row)}</small>
                  </button>
                  <span className="lp-flags">
                    {isPreservedPart(p) && (
                      <span
                        className="lp-flag"
                        title="재생성 시 보존 (확인됨 · 수동 · 클릭 보정)"
                        aria-label="재생성 시 보존"
                      >
                        <Lock size={11} />
                      </span>
                    )}
                    {p.status === "approved" && (
                      <span className="lp-flag is-ok" title="확인됨" aria-label="확인됨">
                        <CheckCircle2 size={12} />
                      </span>
                    )}
                  </span>
                </div>
              </Fragment>
            );
          })
        )}
      </div>
      <div className="lp-footer">
        <div className="lp-progress">
          <span>파츠 확인</span>
          <strong>
            {approved}
            <small> / {total}</small>
          </strong>
          <div className="progress-track">
            <div style={{ width: `${total ? (approved / total) * 100 : 0}%` }} />
          </div>
        </div>
        <p className="lp-note">좌우(L/R)는 캐릭터 기준입니다. 화면 오른쪽 = 캐릭터 왼쪽(L).</p>
        <div className="lp-actions">
          <button type="button" className="icon-button" aria-label="이미지 열기" title="이미지 열기" disabled={busy} onClick={props.onOpenImage}>
            <ImagePlus size={15} />
          </button>
          <button type="button" className="icon-button" aria-label="프로젝트 열기" title="프로젝트 열기" disabled={busy} onClick={props.onOpenProject}>
            <FolderOpen size={15} />
          </button>
          <div className="lp-add" ref={addRef}>
            <button
              type="button"
              className="lp-add-btn"
              aria-haspopup="menu"
              aria-expanded={addOpen}
              disabled={busy || !total}
              onClick={() => setAddOpen((v) => !v)}
              title="머리카락·모자·의상·소품 분리 레이어 추가"
            >
              분리 레이어
              <ChevronDown size={12} />
            </button>
            {addOpen && (
              <div className="lp-add-menu" role="menu">
                {DYNAMIC_LAYER_KINDS.map(([kind, label]) => (
                  <button
                    type="button"
                    role="menuitem"
                    key={kind}
                    onClick={() => {
                      setAddOpen(false);
                      props.onAddDynamic(kind);
                    }}
                  >
                    + {label}
                  </button>
                ))}
              </div>
            )}
          </div>
          <button
            type="button"
            className="icon-button"
            aria-label="파츠 추가"
            title="빈 파츠 추가"
            disabled={!hasProject || busy}
            onClick={props.onAddPart}
          >
            <Plus size={15} />
          </button>
        </div>
      </div>
    </section>
  );
}
