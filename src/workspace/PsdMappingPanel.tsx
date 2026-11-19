import { useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  ChevronDown,
  ChevronRight,
  Eye,
  EyeOff,
  FileStack,
  Layers,
  Scissors,
  SplitSquareHorizontal,
  Wand2,
  X,
} from "lucide-react";
import {
  effectiveBlend,
  expandTarget,
  isFoldable,
  mappingTargetOptions,
  missingDefaultParts,
  resolveLayerTarget,
  type LayerMapping,
  type PsdLayerNode,
  type PsdLayerPixels,
} from "../psdLayers";
import { interleavedParts, workingSize } from "../layerParts";
import { describeDynamicPart, PARENT_PART_DEFS, PART_DEFS } from "../types";
import type { PsdSession } from "../layerProject";

/**
 * B1 mapping screen: PSD layer tree (left, T11a layer-panel rows) → flat2rig parts (right).
 * Three clicks: open the PSD, look at the suggestion, "파츠 생성".
 */

const META = new Map<string, readonly [string, string, string, string]>(
  [...PART_DEFS, ...PARENT_PART_DEFS].map((d) => [d[0], d]),
);

const TEMPLATE_LABELS: Record<string, string> = {
  "eye{S}_white": "눈흰자 · 좌우 자동",
  "eye{S}_iris": "눈동자 · 좌우 자동",
  "eye{S}_lower": "눈언더 · 좌우 자동",
  "eye{S}_upper": "눈꺼풀 · 좌우 자동",
  "brow{S}": "눈썹 · 좌우 자동",
  "hair_side_{S}": "옆머리 · 좌우 자동",
};

export function targetLabel(id: string): string {
  if (TEMPLATE_LABELS[id]) return TEMPLATE_LABELS[id];
  const known = META.get(id);
  if (known) return known[1];
  return describeDynamicPart(id).name;
}

function targetColor(id: string | null): string {
  if (!id) return "transparent";
  const concrete = id.replace("{S}", "L");
  return META.get(concrete)?.[3] ?? describeDynamicPart(concrete).color;
}

const SOURCE_LABEL: Record<LayerMapping["source"], string> = {
  name: "이름 규칙",
  group: "그룹 이름",
  seethrough: "See-through 태그",
  manual: "직접 지정",
  none: "제안 없음",
};

function nextHairIds(targets: Iterable<string>): string[] {
  const used = new Set(targets);
  return (["hair_front", "hair_side_L", "hair_side_R", "hair_back"] as const).map((base) => {
    let n = 2;
    while (used.has(`${base}_${n}`)) n++;
    return used.has(base) ? `${base}_${n}` : base;
  });
}

function LayerPixelsThumb({ px, size = 30 }: { px?: PsdLayerPixels; size?: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c || !px || !px.width || !px.height) return;
    const scale = Math.min(size / px.width, size / px.height);
    const w = Math.max(1, Math.round(px.width * scale)),
      h = Math.max(1, Math.round(px.height * scale));
    c.width = w;
    c.height = h;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    const tmp = document.createElement("canvas");
    tmp.width = px.width;
    tmp.height = px.height;
    const tctx = tmp.getContext("2d");
    if (!tctx) return;
    tctx.putImageData(new ImageData(new Uint8ClampedArray(px.rgba), px.width, px.height), 0, 0);
    ctx.imageSmoothingQuality = "high";
    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(tmp, 0, 0, w, h);
  }, [px, size]);
  return px ? <canvas ref={ref} /> : <i className="lp-none" aria-hidden="true" />;
}

export interface PsdMappingPanelProps {
  session: PsdSession;
  busy: boolean;
  onChange: (mappings: LayerMapping[]) => void;
  onGenerate: () => void;
  onCancel: () => void;
}

type Filter = "all" | "todo" | "warn";

export default function PsdMappingPanel({ session, busy, onChange, onGenerate, onCancel }: PsdMappingPanelProps) {
  const { psd, mappings } = session;
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  const [filter, setFilter] = useState<Filter>("all");
  const [selected, setSelected] = useState<string | null>(null);
  const byPath = useMemo(() => new Map(mappings.map((m) => [m.path, m])), [mappings]);
  const nodeByPath = useMemo(() => new Map(psd.nodes.map((n) => [n.path, n])), [psd]);
  const warningsByPath = useMemo(() => {
    const out = new Map<string, string[]>();
    for (const w of session.suggestion.warnings) {
      const list = out.get(w.path) ?? [];
      list.push(w.message);
      out.set(w.path, list);
    }
    return out;
  }, [session.suggestion]);
  const work = workingSize(psd.width, psd.height);
  const interleaved = useMemo(() => interleavedParts(psd, mappings), [psd, mappings]);
  const missing = useMemo(() => missingDefaultParts(mappings), [mappings]);
  const included = mappings.filter((m) => m.include);
  const unassigned = included.filter((m) => !m.target);
  const nonNormal = mappings.filter((m) => m.blend);
  /** Parts (top first) with their layer names. */
  const parts = useMemo(() => {
    const map = new Map<string, { id: string; layers: string[]; order: number; split: boolean }>();
    for (const m of mappings) {
      if (!m.include || !m.target) continue;
      const node = nodeByPath.get(m.path);
      for (const id of expandTarget(m.target)) {
        const e = map.get(id) ?? { id, layers: [], order: -1, split: false };
        e.layers.push(node?.name ?? m.path);
        e.order = Math.max(e.order, node?.order ?? -1);
        if (m.target.includes("{S}")) e.split = true;
        map.set(id, e);
      }
    }
    return [...map.values()].sort((a, b) => b.order - a.order || a.id.localeCompare(b.id));
  }, [mappings, nodeByPath]);
  const options = useMemo(() => {
    const current = mappings.map((m) => m.target).filter((t): t is string => !!t);
    const base = mappingTargetOptions([...current.filter((t) => !t.includes("{S}")), ...nextHairIds(current)]);
    return [...base, ...Object.keys(TEMPLATE_LABELS)];
  }, [mappings]);

  const setMapping = (path: string, patch: Partial<LayerMapping>) =>
    onChange(mappings.map((m) => (m.path === path ? { ...m, ...patch, source: patch.source ?? "manual" } : m)));
  const assignTarget = (path: string, target: string | null) =>
    setMapping(path, {
      target: resolveLayerTarget(psd, path, target, session.midlineX),
      include: target ? true : (byPath.get(path)?.include ?? true),
      note: undefined,
    });
  const descendants = (group: string) =>
    psd.layers.filter((n) => n.path.startsWith(`${group}/`)).map((n) => n.path);
  const assignGroup = (group: string, target: string | null) => {
    const paths = new Set(descendants(group));
    onChange(
      mappings.map((m) =>
        paths.has(m.path)
          ? {
              ...m,
              target: resolveLayerTarget(psd, m.path, target, session.midlineX),
              source: "manual",
              note: undefined,
            }
          : m,
      ),
    );
  };
  const toggleGroupInclude = (group: string, include: boolean) => {
    const paths = new Set(descendants(group));
    onChange(mappings.map((m) => (paths.has(m.path) ? { ...m, include, source: "manual" } : m)));
  };
  const layerWarn = (n: PsdLayerNode) => {
    const m = byPath.get(n.path);
    return Boolean(m?.blend || warningsByPath.has(n.path) || (m?.include && !m.target));
  };
  const visibleNodes = psd.nodes.filter((n) => {
    for (let p = n.parentPath; p; p = nodeByPath.get(p)?.parentPath ?? null) if (collapsed.has(p)) return false;
    if (n.isGroup || filter === "all") return true;
    const m = byPath.get(n.path);
    if (filter === "todo") return Boolean(m?.include && !m.target);
    return layerWarn(n);
  });
  // Layer panels read top-first: reverse sibling order while keeping parents above children.
  const topFirst = useMemo(() => {
    const kids = new Map<string | null, PsdLayerNode[]>();
    for (const n of psd.nodes) {
      const list = kids.get(n.parentPath) ?? [];
      list.push(n);
      kids.set(n.parentPath, list);
    }
    const out: PsdLayerNode[] = [];
    const walk = (parent: string | null) => {
      for (const n of [...(kids.get(parent) ?? [])].reverse()) {
        out.push(n);
        if (n.isGroup) walk(n.path);
      }
    };
    walk(null);
    return out;
  }, [psd]);
  const shown = new Set(visibleNodes.map((n) => n.path));
  const sel = selected ? nodeByPath.get(selected) : undefined;
  const selMap = selected ? byPath.get(selected) : undefined;

  const targetSelect = (
    value: string | null,
    label: string,
    onSelect: (t: string | null) => void,
    disabled = false,
    placeholder = "미지정",
  ) => (
    <select
      className="pm-target"
      aria-label={label}
      value={value ?? ""}
      disabled={busy || disabled}
      onChange={(e) => onSelect(e.target.value || null)}
      onClick={(e) => e.stopPropagation()}
    >
      <option value="">{placeholder}</option>
      {value && !options.includes(value) && <option value={value}>{value} · {targetLabel(value)}</option>}
      {options.map((id) => (
        <option key={id} value={id}>
          {id.includes("{S}") ? id.replace("{S}", "L/R") : id} · {targetLabel(id)}
        </option>
      ))}
    </select>
  );

  return (
    <section className="pm" aria-label="PSD 레이어 → 파츠 매핑">
      <header className="pm-head">
        <FileStack size={15} />
        <strong title={session.fileName}>{session.fileName}</strong>
        <span className="pm-meta">
          {psd.width}×{psd.height}
          {work.width !== psd.width ? ` → ${work.width}×${work.height}` : ""} · 레이어 {psd.layers.length}
          {psd.seeThrough ? " · See-through" : ""}
        </span>
        <span className="pm-meta" title="좌우(L/R)는 캐릭터 기준: 화면 오른쪽 = 캐릭터 왼쪽(L). 이름의 좌우 표기 대신 픽셀 위치로 정합니다.">
          중심선 x {Math.round(session.midlineX)} ({session.suggestion.midlineSource === "face" ? "얼굴" : session.suggestion.midlineSource})
        </span>
        <span className="pm-spacer" />
        <div className="pm-filter" role="radiogroup" aria-label="레이어 필터">
          {(
            [
              ["all", "전체"],
              ["todo", `미지정 ${unassigned.length}`],
              ["warn", "확인 필요"],
            ] as const
          ).map(([k, label]) => (
            <button
              key={k}
              type="button"
              role="radio"
              aria-checked={filter === k}
              className={filter === k ? "is-active" : undefined}
              onClick={() => setFilter(k)}
            >
              {label}
            </button>
          ))}
        </div>
      </header>
      <div className="pm-body">
        <div className="pm-tree lp" aria-label="PSD 레이어">
          <div className="pm-colhead lp-colhead" aria-hidden="true">
            <span>포함</span>
            <span>레이어</span>
            <span>이름 · 제안 근거</span>
            <span>파츠</span>
          </div>
          <div className="lp-list" role="list">
            {topFirst
              .filter((n) => shown.has(n.path))
              .map((n) => {
                const indent = { paddingLeft: 6 + n.depth * 14 };
                if (n.isGroup) {
                  const kids = descendants(n.path).map((p) => byPath.get(p)!).filter(Boolean);
                  const inc = kids.filter((m) => m.include).length;
                  const targets = new Set(kids.map((m) => m.target ?? ""));
                  const isCollapsed = collapsed.has(n.path);
                  return (
                    <div key={n.path} role="listitem" className="pm-group lp-group" style={indent} data-path={n.path}>
                      <button
                        type="button"
                        className="pm-chev"
                        aria-label={`${n.name} ${isCollapsed ? "펼치기" : "접기"}`}
                        aria-expanded={!isCollapsed}
                        onClick={() =>
                          setCollapsed((s) => {
                            const next = new Set(s);
                            if (next.has(n.path)) next.delete(n.path);
                            else next.add(n.path);
                            return next;
                          })
                        }
                      >
                        {isCollapsed ? <ChevronRight size={12} /> : <ChevronDown size={12} />}
                      </button>
                      <button
                        type="button"
                        className={`lp-eye${inc && inc < kids.length ? " is-partial" : !inc ? " is-hidden" : ""}`}
                        role="checkbox"
                        aria-checked={inc === kids.length ? true : inc ? "mixed" : false}
                        aria-label={`${n.name} 그룹 포함`}
                        disabled={busy || !kids.length}
                        onClick={() => toggleGroupInclude(n.path, inc < kids.length)}
                      >
                        {inc ? <Eye size={13} /> : <EyeOff size={13} />}
                      </button>
                      <span className="lp-group-name">
                        {n.name}
                        {n.effectiveHidden && <em className="pm-tag">숨김</em>}
                        {n.guide && <em className="pm-tag">가이드</em>}
                      </span>
                      <span className="lp-group-count">{kids.length}</span>
                      {targetSelect(
                        targets.size === 1 ? [...targets][0] || null : null,
                        `${n.name} 그룹 일괄 지정`,
                        (t) => assignGroup(n.path, t),
                        !kids.length,
                        targets.size === 1 && targets.has("") ? "미지정" : "일괄 지정…",
                      )}
                    </div>
                  );
                }
                const m = byPath.get(n.path);
                if (!m) return null;
                const blend = effectiveBlend(n);
                const warn = warningsByPath.get(n.path);
                return (
                  <div
                    key={n.path}
                    role="listitem"
                    className={`pm-row lp-row${selected === n.path ? " is-selected" : ""}${!m.include ? " is-off" : ""}${m.include && !m.target ? " is-todo" : ""}`}
                    style={indent}
                    data-path={n.path}
                    onClick={() => setSelected(n.path)}
                  >
                    <button
                      type="button"
                      className="lp-eye"
                      role="checkbox"
                      aria-checked={m.include}
                      aria-label={`${n.name} 포함`}
                      disabled={busy}
                      onClick={(e) => {
                        e.stopPropagation();
                        setMapping(n.path, { include: !m.include });
                      }}
                    >
                      {m.include ? <Eye size={13} /> : <EyeOff size={13} />}
                    </button>
                    <span className="lp-thumb pm-thumb">
                      <LayerPixelsThumb px={psd.pixels.get(n.path)} />
                    </span>
                    <span className="lp-name pm-name" title={n.path}>
                      <strong>
                        <i className="lp-color" style={{ background: targetColor(m.target) }} />
                        {n.name}
                        {n.clipping && (
                          <span className="pm-icon" title="클리핑 마스크">
                            <Scissors size={10} />
                          </span>
                        )}
                        {m.target?.includes("{S}") && (
                          <span className="pm-icon" title="좌우가 한 레이어: 연결 요소로 L/R 분리">
                            <SplitSquareHorizontal size={10} />
                          </span>
                        )}
                      </strong>
                      <small>
                        {m.note ?? SOURCE_LABEL[m.source]}
                        {n.effectiveOpacity < 1 ? ` · 불투명도 ${Math.round(n.effectiveOpacity * 100)}%` : ""}
                      </small>
                    </span>
                    <span className="pm-cell">
                      {targetSelect(m.target, `${n.name} 파츠`, (t) => assignTarget(n.path, t))}
                      {blend !== "normal" && (
                        <span className="pm-blend" title={warn?.join("\n")}>
                          <AlertTriangle size={11} />
                          <span>{blend}</span>
                          <select
                            aria-label={`${n.name} 블렌드 처리`}
                            value={m.include ? (m.blend ?? "normal") : "exclude"}
                            disabled={busy}
                            onClick={(e) => e.stopPropagation()}
                            onChange={(e) => {
                              const v = e.target.value;
                              if (v === "exclude") setMapping(n.path, { include: false });
                              else setMapping(n.path, { include: true, blend: v as "fold" | "normal" });
                            }}
                          >
                            <option value="exclude">제외</option>
                            {isFoldable(blend) && <option value="fold">파츠 안에서 {blend} 적용</option>}
                            <option value="normal">normal로 합성</option>
                          </select>
                        </span>
                      )}
                    </span>
                  </div>
                );
              })}
          </div>
        </div>
        <aside className="pm-side" aria-label="생성될 파츠">
          <div className="pm-side-head">
            <Layers size={13} />
            <span>생성될 파츠</span>
            <strong>{parts.length}</strong>
          </div>
          <ol className="pm-parts">
            {parts.map((p) => (
              <li key={p.id} data-part-id={p.id}>
                <i className="lp-color" style={{ background: targetColor(p.id) }} />
                <span className="pm-part-id">{p.id}</span>
                <span className="pm-part-name">{targetLabel(p.id)}</span>
                <small title={p.layers.join(", ")}>
                  {p.split ? "L/R 분리 · " : ""}
                  {p.layers.length}장
                </small>
              </li>
            ))}
            {missing.map((id) => (
              <li key={id} className="is-missing" data-part-id={id}>
                <i className="lp-color" style={{ background: targetColor(id) }} />
                <span className="pm-part-id">{id}</span>
                <span className="pm-part-name">{targetLabel(id)}</span>
                <small className="pm-none">없음</small>
              </li>
            ))}
          </ol>
          {(unassigned.length > 0 || interleaved.length > 0 || nonNormal.length > 0 || session.suggestion.warnings.length > 0) && (
            <div className="pm-notes">
              {unassigned.length > 0 && (
                <p className="is-warn">
                  <AlertTriangle size={11} /> 미지정 {unassigned.length}개 레이어는 생성에서 빠집니다.
                </p>
              )}
              {missing.length > 0 && <p>없는 기본 파츠는 자동으로 채우지 않습니다: {missing.join(", ")}</p>}
              {interleaved.map((x) => (
                <p key={x.id}>
                  <AlertTriangle size={11} /> {x.id}: PSD에서 {x.between.join(", ")} 레이어와 교차 — 파츠는 가장 위 레이어 위치에 쌓입니다.
                </p>
              ))}
              {session.suggestion.warnings
                .filter((w) => w.kind !== "blend")
                .map((w, i) => (
                  <p key={i}>{w.message}</p>
                ))}
              {nonNormal.map((m) => (
                <p key={m.path}>
                  <AlertTriangle size={11} /> {nodeByPath.get(m.path)?.name}: {effectiveBlend(nodeByPath.get(m.path)!)} →{" "}
                  {!m.include ? "제외" : m.blend === "fold" ? "파츠 안에서 적용" : "normal로 합성"}
                </p>
              ))}
            </div>
          )}
          {sel && selMap && (
            <div className="pm-detail">
              <span className="lp-thumb pm-preview">
                <LayerPixelsThumb px={psd.pixels.get(sel.path)} size={120} />
              </span>
              <small>{sel.path}</small>
            </div>
          )}
          <div className="pm-foot">
            <button type="button" disabled={busy} onClick={onCancel}>
              <X size={14} />
              취소
            </button>
            <button type="button" className="primary" data-action="generate-layer-parts" disabled={busy || !parts.length} onClick={onGenerate}>
              <Wand2 size={14} />
              파츠 생성 ({parts.length})
            </button>
          </div>
        </aside>
      </div>
    </section>
  );
}

