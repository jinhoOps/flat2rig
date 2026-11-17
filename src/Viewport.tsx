import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type {
  FaceCandidate,
  Landmark28Point,
  Landmarks,
  LandmarkId,
  LandmarkProvenance,
  Mesh,
  Parameters,
  Part,
  Point,
  Project,
  ShapeHandleValues,
  ShapeRegion,
  Tool,
} from "./types";
import { canvas, loadImage, pixels, pixelsCanvas } from "./images";
import { clamp, compositePartTexture, effectiveMask, meshForPart, paint } from "./engine";
import {
  createPhysicsState,
  evaluateRigFrame,
  type HairPhysicsState,
} from "./rig/evaluate";
import {
  buildProjectSubstituteTextures,
  renderCharacterFrame,
  type RenderScratch,
  type SubstituteLayerTexture,
} from "./rig/render";
import type { ProjectTextureSet } from "./rig/useProjectTextures";

interface Props {
  project: Project;
  selected: string;
  tool: Tool;
  radius: number;
  zoom: number;
  original: boolean;
  overlay: boolean;
  fillOverlay?: boolean;
  /** T13b exposure-frequency heatmap (canvas-sized, rest space), drawn over the working canvas. */
  exposureOverlay?: CanvasImageSource | null;
  compareDiff?: boolean;
  wire: boolean;
  solo: boolean;
  preview: boolean;
  playing: boolean;
  cleanStage?: boolean;
  parameters: Parameters;
  meshes: Map<string, Mesh>;
  sharedTextures?: ProjectTextureSet;
  overrideShapeIds?: Partial<Record<ShapeRegion, string>>;
  overrideHandles?: Partial<Record<ShapeRegion, Partial<ShapeHandleValues>>>;
  onStart: (point: Point, exclude?: boolean) => void;
  onMove: (point: Point) => void;
  onEnd: (point: Point) => void;
  onCommitMask?: (partId: string, mask: Uint8Array, fillMask?: Uint8Array) => void;
}
const landmarkLabels: Record<LandmarkId, string> = {
  eyeL: "L · 눈",
  eyeR: "R · 눈",
  mouth: "입",
  chin: "턱",
  crown: "얼굴 상단",
};
export default function Viewport(props: Props) {
  const {
    project: p,
    parameters,
    selected,
    tool,
    radius,
    zoom,
    preview,
    playing,
    meshes,
  } = props;
  const root = useRef<HTMLDivElement>(null),
    element = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState({ w: 600, h: 700 });
  const [localPartTextures, setLocalPartTextures] = useState<Map<string, HTMLCanvasElement>>(
      () => new Map(),
    ),
    [localSubTextures, setLocalSubTextures] = useState<Map<string, SubstituteLayerTexture>>(
      () => new Map(),
    ),
    [localSource, setLocalSource] = useState<HTMLImageElement>();
  const partTextures = props.sharedTextures?.partTextures ?? localPartTextures;
  const subTextures = props.sharedTextures?.substituteTextures ?? localSubTextures;
  const source = props.sharedTextures?.sourceImage ?? localSource;
  const [cursor, setCursor] = useState<Point | null>(null),
    [frame, setFrame] = useState(0);
  const physics = useRef<HairPhysicsState>(createPhysicsState());
  const renderScratch = useRef<RenderScratch | undefined>(undefined);
  useEffect(
    () => () => {
      renderScratch.current?.gl?.dispose();
    },
    [],
  );
  const exprTransition = useRef<{
    currentId: string;
    prevId: string | null;
    startedAt: number;
  }>({ currentId: p.activeExpressionId, prevId: null, startedAt: 0 });
  const fallbackMeshCache = useRef<{
    source: string;
    lm: Landmarks;
    w: number;
    h: number;
    map: Map<string, { part: Part; mesh: Mesh }>;
  }>({ source: "", lm: p.landmarks, w: 0, h: 0, map: new Map() });
  const fallbackMeshes = useMemo(() => {
    const cache = fallbackMeshCache.current;
    if (cache.source !== p.source || cache.lm !== p.landmarks || cache.w !== p.width || cache.h !== p.height) {
      cache.source = p.source;
      cache.lm = p.landmarks;
      cache.w = p.width;
      cache.h = p.height;
      cache.map.clear();
    }
    const nextMap = new Map<string, Mesh>();
    for (const part of p.parts) {
      const provided = meshes.get(part.id);
      if (provided) {
        nextMap.set(part.id, provided);
        continue;
      }
      const prev = cache.map.get(part.id);
      if (prev && prev.part === part) {
        nextMap.set(part.id, prev.mesh);
      } else {
        const built = meshForPart(effectiveMask(part), p.width, p.height, 14, p.landmarks, part);
        cache.map.set(part.id, { part, mesh: built });
        nextMap.set(part.id, built);
      }
    }
    return nextMap;
  }, [p.source, p.parts, p.width, p.height, p.landmarks, meshes]);
  const [spaceHeld, setSpaceHeld] = useState(false);
  const [panDragging, setPanDragging] = useState(false);
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });
  const panGesture = useRef<{
    startClientX: number;
    startClientY: number;
    startScrollLeft: number;
    startScrollTop: number;
    startPanX: number;
    startPanY: number;
  } | null>(null);
  const sourcePixelsCache = useRef<{ img: HTMLImageElement; w: number; h: number; data: Uint8ClampedArray } | null>(null);
  const getSourcePixels = useCallback(() => {
    if (!source) return null;
    const cur = sourcePixelsCache.current;
    if (cur && cur.img === source && cur.w === p.width && cur.h === p.height) {
      return cur.data;
    }
    const data = pixels(source, p.width, p.height);
    sourcePixelsCache.current = { img: source, w: p.width, h: p.height, data };
    return data;
  }, [source, p.width, p.height]);
  const brushWork = useRef<{
    partId: string;
    mask: Uint8Array;
    fillMask?: Uint8Array;
    workTexCanvas: HTMLCanvasElement;
    workTexCtx: CanvasRenderingContext2D;
    srcTexData?: Uint8ClampedArray;
    srcData: Uint8ClampedArray | null;
    workTintCanvas?: HTMLCanvasElement;
    workTintCtx?: CanvasRenderingContext2D;
    partRgb: [number, number, number];
    last: Point;
    rafId: number;
  } | null>(null);
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.code !== "Space") return;
      const target = e.target instanceof Element ? e.target : null;
      if (target?.matches("input,textarea,select,[role=checkbox]")) return;
      if (!preview) {
        e.preventDefault();
        setSpaceHeld(true);
      }
    };
    const onKeyUp = (e: KeyboardEvent) => {
      if (e.code === "Space") setSpaceHeld(false);
    };
    const clearSpace = () => setSpaceHeld(false);
    window.addEventListener("keydown", onKeyDown, true);
    window.addEventListener("keyup", onKeyUp, true);
    window.addEventListener("blur", clearSpace);
    return () => {
      window.removeEventListener("keydown", onKeyDown, true);
      window.removeEventListener("keyup", onKeyUp, true);
      window.removeEventListener("blur", clearSpace);
    };
  }, [preview]);
  useEffect(() => {
    if (zoom === 1) setPanOffset({ x: 0, y: 0 });
  }, [zoom]);
  const textureCache = useRef(
    new Map<string, { part: Part; canvas: HTMLCanvasElement }>(),
  );
  useEffect(() => {
    const node = root.current!;
    const observer = new ResizeObserver((entries) =>
      setSize({
        w: entries[0].contentRect.width - 90,
        h: entries[0].contentRect.height - 80,
      }),
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    textureCache.current.clear();
    physics.current = createPhysicsState();
    setFrame(0);
    setPanOffset({ x: 0, y: 0 });
    if (props.sharedTextures) return;
    let active = true;
    loadImage(p.source).then((img) => {
      if (active) setLocalSource(img);
    });
    return () => {
      active = false;
    };
  }, [p.source, props.sharedTextures]);
  useEffect(() => {
    if (props.sharedTextures || !localSource || localSource.src !== p.source) return;
    let active = true;
    const data = pixels(localSource, p.width, p.height);
    Promise.all(
      p.parts.map(async (part) => {
        const cached = textureCache.current.get(part.id);
        if (cached?.part === part) return [part.id, cached.canvas] as const;
        const texPixels = part.texture
          ? pixels(await loadImage(part.texture), p.width, p.height)
          : undefined;
        const result = pixelsCanvas(
          compositePartTexture(part, data, texPixels),
          p.width,
          p.height,
        );
        if (active) textureCache.current.set(part.id, { part, canvas: result });
        return [part.id, result] as const;
      }),
    ).then((entries) => {
      if (active) setLocalPartTextures(new Map(entries));
    });
    return () => {
      active = false;
    };
  }, [props.sharedTextures, localSource, p.source, p.parts, p.width, p.height]);
  useEffect(() => {
    if (props.sharedTextures) return;
    if (!p.shapeAssets) {
      setLocalSubTextures(new Map());
      return;
    }
    let active = true;
    buildProjectSubstituteTextures(p).then((entries) => {
      if (active) setLocalSubTextures(entries);
    });
    return () => {
      active = false;
    };
  }, [props.sharedTextures, p.shapeAssets, p.width, p.height, p.landmarks]);
  const [transitionElapsedMs, setTransitionElapsedMs] = useState(0);
  useEffect(() => {
    if (p.activeExpressionId === exprTransition.current.currentId) return;
    const prevId = exprTransition.current.currentId;
    const targetExp = p.expressions.find((e) => e.id === p.activeExpressionId);
    const dur = targetExp?.transitionMs ?? 0;
    exprTransition.current = {
      currentId: p.activeExpressionId,
      prevId,
      startedAt: performance.now(),
    };
    if (dur <= 0 || playing) {
      setTransitionElapsedMs(dur);
      return;
    }
    let raf = 0;
    const tickTransition = (now: number) => {
      const elapsed = now - exprTransition.current.startedAt;
      if (elapsed >= dur) {
        setTransitionElapsedMs(dur);
        return;
      }
      setTransitionElapsedMs(elapsed);
      raf = requestAnimationFrame(tickTransition);
    };
    setTransitionElapsedMs(0);
    raf = requestAnimationFrame(tickTransition);
    return () => cancelAnimationFrame(raf);
  }, [p.activeExpressionId, p.expressions, playing]);
  const tint = useMemo(() => {
    if (!props.overlay) return null;
    const part = p.parts.find((t) => t.id === selected);
    if (!part) return null;
    const c = canvas(p.width, p.height),
      ctx = c.getContext("2d")!,
      data = ctx.createImageData(p.width, p.height);
    const rgb = part.color
      .slice(1)
      .match(/../g)!
      .map((t) => parseInt(t, 16));
    for (let i = 0; i < part.mask.length; i++) {
      const m = part.mask[i];
      if (!m) continue;
      data.data[i * 4] = rgb[0];
      data.data[i * 4 + 1] = rgb[1];
      data.data[i * 4 + 2] = rgb[2];
      data.data[i * 4 + 3] = Math.round(m * 0.48);
    }
    ctx.putImageData(data, 0, 0);
    return c;
  }, [p.parts, p.width, p.height, selected, props.overlay]);
  const fillTint = useMemo(() => {
    if (!props.fillOverlay) return null;
    const anyFill = p.parts.some((t) => t.visible && t.status !== "missing" && !!t.fillMask);
    if (!anyFill) return null;
    const c = canvas(p.width, p.height),
      ctx = c.getContext("2d")!,
      data = ctx.createImageData(p.width, p.height);
    let drawn = false;
    for (const part of p.parts) {
      if (!part.fillMask || !part.visible || part.status === "missing") continue;
      if (props.solo && part.id !== selected) continue;
      const alphaScale = part.id === selected ? 0.62 : 0.34;
      for (let i = 0; i < part.fillMask.length; i++) {
        const a = part.fillMask[i];
        if (!a) continue;
        drawn = true;
        data.data[i * 4] = 217;
        data.data[i * 4 + 1] = 119;
        data.data[i * 4 + 2] = 6;
        data.data[i * 4 + 3] = Math.max(data.data[i * 4 + 3], Math.round(a * alphaScale));
      }
    }
    if (!drawn) return null;
    ctx.putImageData(data, 0, 0);
    return c;
  }, [p.parts, p.width, p.height, selected, props.solo, props.fillOverlay]);
  useEffect(() => {
    if (!playing || !preview) {
      // Explicit physics & frame reset when paused, exited preview, or stopped
      physics.current = createPhysicsState();
      setFrame(0);
      return;
    }
    let id = 0;
    const start = performance.now();
    let last = 0;
    const tick = (time: number) => {
      if (time - last > 32) {
        setFrame((time - start) / 1000);
        last = time;
      }
      id = requestAnimationFrame(tick);
    };
    id = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(id);
  }, [playing, preview]);
  const scale =
    Math.max(0.05, Math.min(size.w / p.width, size.h / p.height)) * zoom;
  const drawOverlays = useCallback(
    (
      ctx: CanvasRenderingContext2D,
      activeTint: HTMLCanvasElement | null,
      activeCursor: Point | null,
    ) => {
      if (!preview && !props.original && props.compareDiff && source && p.parts.length)
        drawDiffOverlay(ctx, source, p.width, p.height);
      if (!preview && props.overlay && activeTint) ctx.drawImage(activeTint, 0, 0);
      if (!preview && props.fillOverlay && fillTint) ctx.drawImage(fillTint, 0, 0);
      if (!preview && props.exposureOverlay) ctx.drawImage(props.exposureOverlay, 0, 0);
      if (!preview && (tool === "landmark" || tool === "face" || !p.parts.length))
        drawAnalysisOverlay(
          ctx,
          p.landmarks,
          p.landmarkProvenance,
          p.faceCandidates,
          p.selectedFaceId ?? 0,
          p.landmarks28,
          scale,
        );
      if (!preview && tool === "prompt") {
        const activePart = p.parts.find((t) => t.id === selected);
        if (activePart?.prompts?.length) {
          for (const pt of activePart.prompts) {
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, 5 / scale, 0, Math.PI * 2);
            ctx.fillStyle = pt.label === 1 ? "#18794E" : "#C9372C";
            ctx.fill();
            ctx.strokeStyle = "#ffffff";
            ctx.lineWidth = 1.6 / scale;
            ctx.stroke();
          }
        }
      }
      if (activeCursor && !preview && (tool === "add" || tool === "erase")) {
        ctx.strokeStyle = tool === "erase" ? "#ffa8a8" : "#7ff5d7";
        ctx.lineWidth = 1.3 / scale;
        ctx.beginPath();
        ctx.arc(activeCursor.x, activeCursor.y, radius, 0, Math.PI * 2);
        ctx.stroke();
      }
    },
    [
      preview,
      props.original,
      props.compareDiff,
      props.overlay,
      props.fillOverlay,
      props.exposureOverlay,
      source,
      p.parts,
      p.width,
      p.height,
      p.landmarks,
      p.landmarkProvenance,
      p.faceCandidates,
      p.selectedFaceId,
      p.landmarks28,
      fillTint,
      tool,
      selected,
      scale,
      radius,
    ],
  );
  const drawBrushFrame = useCallback(() => {
    const bw = brushWork.current;
    const el = element.current;
    if (!bw || !el) return;
    bw.rafId = 0;
    const ctx = el.getContext("2d")!;
    const liveTextures = new Map(partTextures);
    liveTextures.set(bw.partId, bw.workTexCanvas);
    const evaluated = evaluateRigFrame({
      project: p,
      timeSec: 0,
      playing: false,
      previousExpressionId: null,
      transitionElapsedMs: 0,
      physicsState: physics.current,
      deterministic: true,
    });
    renderScratch.current = renderCharacterFrame({
      ctx,
      project: p,
      evaluated,
      partTextures: liveTextures,
      meshes: fallbackMeshes,
      substituteTextures: subTextures,
      sourceImage: source,
      showOriginal: props.original,
      soloPartId: props.solo ? selected : undefined,
      previewDeform: false,
      wireframe: props.wire,
      selectedPartId: selected,
      scale,
      scratch: renderScratch.current,
      enableGpu: false,
    });
    drawOverlays(ctx, bw.workTintCanvas ?? tint, cursor);
  }, [
    p,
    partTextures,
    fallbackMeshes,
    subTextures,
    source,
    props.original,
    props.solo,
    props.wire,
    selected,
    scale,
    tint,
    cursor,
    drawOverlays,
  ]);
  const updateBrushDirtyRect = (
    bw: NonNullable<typeof brushWork.current>,
    a: Point,
    b: Point,
  ) => {
    const w = p.width;
    const h = p.height;
    const pad = Math.ceil(radius + 2);
    const x0 = clamp(Math.floor(Math.min(a.x, b.x) - pad), 0, w - 1);
    const x1 = clamp(Math.ceil(Math.max(a.x, b.x) + pad), 0, w - 1);
    const y0 = clamp(Math.floor(Math.min(a.y, b.y) - pad), 0, h - 1);
    const y1 = clamp(Math.ceil(Math.max(a.y, b.y) + pad), 0, h - 1);
    const rw = x1 - x0 + 1;
    const rh = y1 - y0 + 1;
    if (rw <= 0 || rh <= 0) return;

    const texImg = bw.workTexCtx.createImageData(rw, rh);
    const tintImg = bw.workTintCtx ? bw.workTintCtx.createImageData(rw, rh) : null;
    const [pr, pg, pb] = bw.partRgb;

    for (let dy = 0; dy < rh; dy++) {
      const gy = y0 + dy;
      const rowOff = gy * w;
      const imgRow = dy * rw;
      for (let dx = 0; dx < rw; dx++) {
        const gx = x0 + dx;
        const idx = rowOff + gx;
        const di = (imgRow + dx) * 4;
        const m = bw.mask[idx];
        const eff = bw.fillMask ? Math.max(m, bw.fillMask[idx]) : m;
        if (eff > 0) {
          const k = idx * 4;
          const useSrc =
            !bw.srcTexData ||
            (m > 0 && bw.srcTexData[k + 3] < m && (bw.srcData?.[k + 3] ?? 0) > bw.srcTexData[k + 3]);
          const rgb = useSrc ? bw.srcData : bw.srcTexData;
          if (rgb) {
            texImg.data[di] = rgb[k];
            texImg.data[di + 1] = rgb[k + 1];
            texImg.data[di + 2] = rgb[k + 2];
            texImg.data[di + 3] = Math.min(rgb[k + 3], eff);
          }
        }
        if (tintImg && m > 0) {
          tintImg.data[di] = pr;
          tintImg.data[di + 1] = pg;
          tintImg.data[di + 2] = pb;
          tintImg.data[di + 3] = Math.round(m * 0.48);
        }
      }
    }
    bw.workTexCtx.putImageData(texImg, x0, y0);
    if (tintImg && bw.workTintCtx) {
      bw.workTintCtx.putImageData(tintImg, x0, y0);
    }
  };
  useEffect(() => {
    const el = element.current;
    if (!el || brushWork.current) return;
    const ctx = el.getContext("2d")!;
    const trElapsed = playing
      ? performance.now() - exprTransition.current.startedAt
      : transitionElapsedMs;
    const evaluated = evaluateRigFrame({
      project: p,
      timeSec: preview && playing ? frame : 0,
      playing: preview && playing,
      previousExpressionId: exprTransition.current.prevId,
      transitionElapsedMs: trElapsed,
      userParameters: preview ? parameters : undefined,
      overrideShapeIds: preview ? props.overrideShapeIds : undefined,
      overrideHandles: preview ? props.overrideHandles : undefined,
      physicsState: physics.current,
      deterministic: !preview,
    });
    physics.current = evaluated.nextPhysicsState;

    renderScratch.current = renderCharacterFrame({
      ctx,
      project: p,
      evaluated,
      partTextures,
      meshes: fallbackMeshes,
      substituteTextures: subTextures,
      sourceImage: source,
      showOriginal: props.original,
      soloPartId: props.solo ? selected : undefined,
      previewDeform: preview,
      wireframe: props.wire,
      selectedPartId: selected,
      scale,
      scratch: renderScratch.current,
      enableGpu: preview,
    });

    drawOverlays(ctx, tint, cursor);
  }, [
    p,
    partTextures,
    subTextures,
    source,
    parameters,
    props.original,
    props.overlay,
    props.fillOverlay,
    props.compareDiff,
    props.wire,
    props.solo,
    props.overrideShapeIds,
    props.overrideHandles,
    fallbackMeshes,
    tint,
    fillTint,
    preview,
    playing,
    selected,
    tool,
    radius,
    meshes,
    frame,
    transitionElapsedMs,
    cursor,
    scale,
    drawOverlays,
  ]);
  const point = (event: React.PointerEvent) => {
    const rect = element.current!.getBoundingClientRect();
    return {
      x: clamp(
        ((event.clientX - rect.left) * p.width) / rect.width,
        0,
        p.width - 1,
      ),
      y: clamp(
        ((event.clientY - rect.top) * p.height) / rect.height,
        0,
        p.height - 1,
      ),
    };
  };
  const useLocalBrush = !preview && !!props.onCommitMask && (tool === "add" || tool === "erase");
  const isPanCursor = (!preview && spaceHeld) || panDragging;
  return (
    <div className="viewport" ref={root}>
      <div className="canvas-space">
        <div
          className={props.cleanStage ? "canvas-frame clean" : "canvas-frame"}
          style={{
            width: p.width * scale,
            height: p.height * scale,
            transform:
              panOffset.x !== 0 || panOffset.y !== 0
                ? `translate(${panOffset.x}px, ${panOffset.y}px)`
                : undefined,
          }}
        >
          <canvas
            aria-label="일러스트 편집 캔버스"
            ref={element}
            width={p.width}
            height={p.height}
            style={{
              width: p.width * scale,
              height: p.height * scale,
              cursor: panDragging
                ? "grabbing"
                : isPanCursor
                  ? "grab"
                  : preview
                    ? "default"
                    : tool === "select"
                      ? "default"
                      : "crosshair",
            }}
            onContextMenu={(e) => {
              if (tool === "prompt") e.preventDefault();
            }}
            onPointerDown={(e) => {
              if (e.button === 1 || (!preview && spaceHeld)) {
                e.preventDefault();
                const sc = root.current;
                e.currentTarget.setPointerCapture(e.pointerId);
                panGesture.current = {
                  startClientX: e.clientX,
                  startClientY: e.clientY,
                  startScrollLeft: sc?.scrollLeft ?? 0,
                  startScrollTop: sc?.scrollTop ?? 0,
                  startPanX: panOffset.x,
                  startPanY: panOffset.y,
                };
                setPanDragging(true);
                return;
              }
              if (preview) return;
              e.currentTarget.setPointerCapture(e.pointerId);
              const pt = point(e);
              if (useLocalBrush && e.button === 0) {
                const activePart = p.parts.find((t) => t.id === selected);
                if (!activePart) return;
                const wTex = canvas(p.width, p.height);
                const wCtx = wTex.getContext("2d")!;
                const existingTex = partTextures.get(selected);
                if (existingTex) wCtx.drawImage(existingTex, 0, 0);
                const srcTexData = activePart.texture
                  ? wCtx.getImageData(0, 0, p.width, p.height).data
                  : undefined;
                let wTint: HTMLCanvasElement | undefined;
                let wTintCtx: CanvasRenderingContext2D | undefined;
                if (props.overlay) {
                  wTint = canvas(p.width, p.height);
                  wTintCtx = wTint.getContext("2d")!;
                  if (tint) wTintCtx.drawImage(tint, 0, 0);
                }
                const rgbHex = activePart.color.slice(1).match(/../g) ?? ["3b", "82", "f6"];
                const partRgb: [number, number, number] = [
                  parseInt(rgbHex[0], 16),
                  parseInt(rgbHex[1], 16),
                  parseInt(rgbHex[2], 16),
                ];
                const nextMask = activePart.mask.slice();
                const nextFill = activePart.fillMask?.slice();
                const erase = tool === "erase";
                paint(nextMask, p.width, p.height, pt, pt, radius, erase);
                if (!erase && nextFill) paint(nextFill, p.width, p.height, pt, pt, radius, true);
                const bw: NonNullable<typeof brushWork.current> = {
                  partId: activePart.id,
                  mask: nextMask,
                  fillMask: nextFill,
                  workTexCanvas: wTex,
                  workTexCtx: wCtx,
                  srcTexData,
                  srcData: getSourcePixels(),
                  workTintCanvas: wTint,
                  workTintCtx: wTintCtx,
                  partRgb,
                  last: pt,
                  rafId: 0,
                };
                brushWork.current = bw;
                updateBrushDirtyRect(bw, pt, pt);
                bw.rafId = requestAnimationFrame(drawBrushFrame);
                return;
              }
              const exclude = e.button === 2 || e.altKey || e.shiftKey;
              props.onStart(pt, exclude);
            }}
            onPointerMove={(e) => {
              if (panGesture.current) {
                const pg = panGesture.current;
                const sc = root.current;
                const dx = e.clientX - pg.startClientX;
                const dy = e.clientY - pg.startClientY;
                let remX = dx;
                let remY = dy;
                if (sc) {
                  const maxSL = Math.max(0, sc.scrollWidth - sc.clientWidth);
                  const maxST = Math.max(0, sc.scrollHeight - sc.clientHeight);
                  const wantSL = clamp(pg.startScrollLeft - dx, 0, maxSL);
                  const wantST = clamp(pg.startScrollTop - dy, 0, maxST);
                  sc.scrollLeft = wantSL;
                  sc.scrollTop = wantST;
                  remX = dx - (pg.startScrollLeft - wantSL);
                  remY = dy - (pg.startScrollTop - wantST);
                }
                setPanOffset({
                  x: clamp(pg.startPanX + remX, -size.w * 0.75, size.w * 0.75),
                  y: clamp(pg.startPanY + remY, -size.h * 0.75, size.h * 0.75),
                });
                return;
              }
              const pt = point(e);
              setCursor(pt);
              const bw = brushWork.current;
              if (bw && e.buttons === 1) {
                const erase = tool === "erase";
                paint(bw.mask, p.width, p.height, bw.last, pt, radius, erase);
                if (!erase && bw.fillMask) {
                  paint(bw.fillMask, p.width, p.height, bw.last, pt, radius, true);
                }
                updateBrushDirtyRect(bw, bw.last, pt);
                bw.last = pt;
                if (!bw.rafId) {
                  bw.rafId = requestAnimationFrame(drawBrushFrame);
                }
                return;
              }
              if (e.buttons === 1 && !preview) props.onMove(pt);
            }}
            onPointerUp={(e) => {
              if (panGesture.current) {
                panGesture.current = null;
                setPanDragging(false);
                if (e.currentTarget.hasPointerCapture(e.pointerId))
                  e.currentTarget.releasePointerCapture(e.pointerId);
                return;
              }
              const pt = point(e);
              const bw = brushWork.current;
              if (bw) {
                if (bw.rafId) cancelAnimationFrame(bw.rafId);
                paint(bw.mask, p.width, p.height, bw.last, pt, radius, tool === "erase");
                if (tool !== "erase" && bw.fillMask) {
                  paint(bw.fillMask, p.width, p.height, bw.last, pt, radius, true);
                }
                brushWork.current = null;
                props.onCommitMask?.(bw.partId, bw.mask, bw.fillMask);
              } else if (!preview) {
                props.onEnd(pt);
              }
              if (e.currentTarget.hasPointerCapture(e.pointerId))
                e.currentTarget.releasePointerCapture(e.pointerId);
            }}
            onPointerCancel={(e) => {
              if (panGesture.current) {
                panGesture.current = null;
                setPanDragging(false);
                return;
              }
              const bw = brushWork.current;
              if (bw) {
                if (bw.rafId) cancelAnimationFrame(bw.rafId);
                brushWork.current = null;
                props.onCommitMask?.(bw.partId, bw.mask, bw.fillMask);
              } else {
                props.onEnd(point(e));
              }
            }}
            onPointerLeave={() => setCursor(null)}
          />
          {!props.cleanStage && (
            <>
              <span className="corner tl" />
              <span className="corner tr" />
              <span className="corner bl" />
              <span className="corner br" />
            </>
          )}
        </div>
      </div>
    </div>
  );
}
function drawAnalysisOverlay(
  ctx: CanvasRenderingContext2D,
  landmarks: Landmarks,
  provenance: LandmarkProvenance | undefined,
  candidates: FaceCandidate[] | undefined,
  selectedFaceId: number,
  landmarks28: Landmark28Point[] | undefined,
  scale: number,
) {
  if (candidates?.length) {
    for (const cand of candidates) {
      const active = cand.id === selectedFaceId;
      const [x1, y1, x2, y2] = cand.bbox;
      ctx.save();
      ctx.strokeStyle = active ? "#2456D6" : "#6E7485";
      ctx.lineWidth = (active ? 2.0 : 1.2) / scale;
      if (!active) ctx.setLineDash([5 / scale, 4 / scale]);
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.font = `600 ${13 / scale}px sans-serif`;
      const tag = `얼굴 #${cand.id + 1} (${Math.round(cand.score * 100)}%)`;
      ctx.fillStyle = active ? "#2456D6" : "#6E7485";
      ctx.fillText(tag, x1 + 4 / scale, Math.max(12 / scale, y1 - 4 / scale));
      ctx.restore();
    }
  }
  if (landmarks28?.length) {
    ctx.save();
    ctx.fillStyle = "rgba(36, 86, 214, 0.72)";
    for (const pt of landmarks28) {
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 2.2 / scale, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }
  const keys = Object.keys(landmarks) as LandmarkId[];
  for (const id of keys) {
    const pt = landmarks[id];
    const src = provenance?.[id];
    const ringColor =
      src === "manual"
        ? "#18794E"
        : src === "estimated"
          ? "#9A6700"
          : "#2456D6";
    const suffix =
      src === "manual" ? " [수동]" : src === "estimated" ? " [추정]" : "";
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, 5.2 / scale, 0, Math.PI * 2);
    ctx.fillStyle = "#161920";
    ctx.fill();
    ctx.strokeStyle = ringColor;
    ctx.lineWidth = 2 / scale;
    ctx.stroke();
    ctx.font = `600 ${13 / scale}px sans-serif`;
    ctx.lineWidth = 3 / scale;
    ctx.strokeStyle = "#ffffff";
    const label = `${landmarkLabels[id]}${suffix}`;
    ctx.strokeText(label, pt.x + 10 / scale, pt.y - 9 / scale);
    ctx.fillStyle = ringColor;
    ctx.fillText(label, pt.x + 10 / scale, pt.y - 9 / scale);
  }
}

function drawDiffOverlay(
  ctx: CanvasRenderingContext2D,
  source: HTMLImageElement,
  w: number,
  h: number,
) {
  const comp = ctx.getImageData(0, 0, w, h);
  const orig = pixels(source, w, h);
  const cd = comp.data;
  for (let i = 0; i < w * h; i++) {
    const k = i * 4;
    const oa = orig[k + 3],
      ca = cd[k + 3];
    if (oa > 8 && ca <= 8) {
      // Missing from composite
      cd[k] = 201;
      cd[k + 1] = 55;
      cd[k + 2] = 44;
      cd[k + 3] = 200;
    } else if (oa <= 8 && ca > 8) {
      // Extra pixels added outside original alpha (occlusion fill / overpaint)
      cd[k] = 217;
      cd[k + 1] = 119;
      cd[k + 2] = 6;
      cd[k + 3] = 220;
    } else {
      const diff =
        (Math.abs(orig[k] - cd[k]) +
          Math.abs(orig[k + 1] - cd[k + 1]) +
          Math.abs(orig[k + 2] - cd[k + 2])) /
        3;
      if (diff > 18 && oa > 8) {
        cd[k] = 36;
        cd[k + 1] = 86;
        cd[k + 2] = 214;
        cd[k + 3] = 210;
      } else {
        cd[k] = Math.round(cd[k] * 0.45);
        cd[k + 1] = Math.round(cd[k + 1] * 0.45);
        cd[k + 2] = Math.round(cd[k + 2] * 0.45);
        cd[k + 3] = Math.round(cd[k + 3] * 0.45);
      }
    }
  }
  ctx.putImageData(comp, 0, 0);
}
