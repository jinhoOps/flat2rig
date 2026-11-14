import { useEffect, useMemo, useRef, useState } from "react";
import type {
  Landmarks,
  LandmarkId,
  Mesh,
  Parameters,
  Part,
  Point,
  Project,
  Tool,
} from "./types";
import { canvas, loadImage, pixels, pixelsCanvas } from "./images";
import { clamp, deform, maskedPixels, textureMesh } from "./engine";

interface Props {
  project: Project;
  selected: string;
  tool: Tool;
  radius: number;
  zoom: number;
  original: boolean;
  overlay: boolean;
  wire: boolean;
  solo: boolean;
  preview: boolean;
  playing: boolean;
  parameters: Parameters;
  meshes: Map<string, Mesh>;
  onStart: (point: Point) => void;
  onMove: (point: Point) => void;
  onEnd: (point: Point) => void;
}
const landmarkLabels: Record<LandmarkId, string> = {
  eyeL: "L · 눈",
  eyeR: "R · 눈",
  mouth: "입",
  chin: "턱",
  crown: "얼굴 상단",
};
function triangle(
  ctx: CanvasRenderingContext2D,
  texture: HTMLCanvasElement,
  src: Point[],
  dst: Point[],
) {
  const [a, b, c] = src,
    [p, q, r] = dst;
  const det = a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y);
  if (Math.abs(det) < 0.01) return;
  const aa = (p.x * (b.y - c.y) + q.x * (c.y - a.y) + r.x * (a.y - b.y)) / det;
  const bb = (p.y * (b.y - c.y) + q.y * (c.y - a.y) + r.y * (a.y - b.y)) / det;
  const cc = (p.x * (c.x - b.x) + q.x * (a.x - c.x) + r.x * (b.x - a.x)) / det;
  const dd = (p.y * (c.x - b.x) + q.y * (a.x - c.x) + r.y * (b.x - a.x)) / det;
  const ee =
    (p.x * (b.x * c.y - c.x * b.y) +
      q.x * (c.x * a.y - a.x * c.y) +
      r.x * (a.x * b.y - b.x * a.y)) /
    det;
  const ff =
    (p.y * (b.x * c.y - c.x * b.y) +
      q.y * (c.x * a.y - a.x * c.y) +
      r.y * (a.x * b.y - b.x * a.y)) /
    det;
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(p.x, p.y);
  ctx.lineTo(q.x, q.y);
  ctx.lineTo(r.x, r.y);
  ctx.closePath();
  ctx.clip();
  ctx.transform(aa, bb, cc, dd, ee, ff);
  ctx.drawImage(texture, 0, 0);
  ctx.restore();
}
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
  const [textures, setTextures] = useState<HTMLCanvasElement[]>([]),
    [source, setSource] = useState<HTMLImageElement>();
  const [cursor, setCursor] = useState<Point | null>(null),
    [frame, setFrame] = useState(0);
  const physics = useRef({ position: 0, velocity: 0, previous: 0 });
  const renderMeshes = useMemo(
    () =>
      new Map(
        p.parts.map((part) => [
          part.id,
          textureMesh(part.mask, p.width, p.height),
        ]),
      ),
    [p.parts, p.width, p.height],
  );
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
    let active = true;
    textureCache.current.clear();
    loadImage(p.source).then((img) => {
      if (active) setSource(img);
    });
    return () => {
      active = false;
    };
  }, [p.source]);
  useEffect(() => {
    if (!source || source.src !== p.source) return;
    let active = true;
    const data = pixels(source, p.width, p.height);
    Promise.all(
      p.parts.map(async (part) => {
        const cached = textureCache.current.get(part.id);
        if (cached?.part === part) return cached.canvas;
        const sourcePixels = part.texture
          ? pixels(await loadImage(part.texture), p.width, p.height)
          : data;
        const result = pixelsCanvas(
          maskedPixels(sourcePixels, part.mask),
          p.width,
          p.height,
        );
        if (active) textureCache.current.set(part.id, { part, canvas: result });
        return result;
      }),
    ).then((images) => {
      if (active) setTextures(images);
    });
    return () => {
      active = false;
    };
  }, [source, p.parts, p.width, p.height]);
  const tint = useMemo(() => {
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
      data.data[i * 4] = rgb[0];
      data.data[i * 4 + 1] = rgb[1];
      data.data[i * 4 + 2] = rgb[2];
      data.data[i * 4 + 3] = part.mask[i] * 0.48;
    }
    ctx.putImageData(data, 0, 0);
    return c;
  }, [p.parts, p.revision, p.width, p.height, selected]);
  useEffect(() => {
    if (!playing || !preview) return;
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
  useEffect(() => {
    const el = element.current;
    if (!el) return;
    const ctx = el.getContext("2d")!;
    ctx.clearRect(0, 0, p.width, p.height);
    const pose = { ...parameters };
    if (playing && preview) {
      pose.angleX += Math.sin(frame * 0.9) * 7;
      pose.angleY += Math.sin(frame * 0.65) * 4;
      const blink = Math.pow(Math.max(0, Math.cos(frame * 1.5)), 24);
      pose.eyeL *= 1 - blink;
      pose.eyeR *= 1 - blink;
      pose.mouth = Math.max(pose.mouth, Math.max(0, Math.sin(frame * 2)) * 0.5);
      const ph = physics.current;
      const dt = clamp(frame - ph.previous, 0, 0.05);
      ph.previous = frame;
      ph.velocity +=
        ((Math.sin(frame * 0.9) * 0.6 - ph.position) * 35 - ph.velocity * 7) *
        dt;
      ph.position += ph.velocity * dt;
      pose.hair += ph.position;
    }
    const moving =
      preview &&
      Object.entries(pose).some(([k, v]) =>
        k === "eyeL" || k === "eyeR" ? v !== 1 : v !== 0,
      );
    if (props.original || !p.parts.length) {
      if (source) ctx.drawImage(source, 0, 0);
    } else
      p.parts.forEach((part: Part, i: number) => {
        if (
          !part.visible ||
          part.status === "missing" ||
          (props.solo && part.id !== selected) ||
          !textures[i]
        )
          return;
        const mesh = meshes.get(part.id);
        const renderMesh = renderMeshes.get(part.id);
        if (moving && renderMesh && renderMesh.triangles.length) {
          for (let t = 0; t < renderMesh.triangles.length; t += 3) {
            const ids = renderMesh.triangles.slice(t, t + 3);
            const src = ids.map((j) => renderMesh.vertices[j]);
            triangle(
              ctx,
              textures[i],
              src,
              src.map((pt) => deform(pt, part, p.landmarks, pose)),
            );
          }
        } else ctx.drawImage(textures[i], 0, 0);
        if (props.wire && mesh) {
          ctx.beginPath();
          ctx.strokeStyle = part.id === selected ? "#49efcc" : "#9faecb88";
          ctx.lineWidth = Math.max(0.6, 1 / scale);
          for (let t = 0; t < mesh.triangles.length; t += 3) {
            const pts = mesh.triangles
              .slice(t, t + 3)
              .map((j) =>
                preview
                  ? deform(mesh.vertices[j], part, p.landmarks, pose)
                  : mesh.vertices[j],
              );
            ctx.moveTo(pts[0].x, pts[0].y);
            ctx.lineTo(pts[1].x, pts[1].y);
            ctx.lineTo(pts[2].x, pts[2].y);
            ctx.closePath();
          }
          ctx.stroke();
        }
      });
    if (!preview && props.overlay && tint) ctx.drawImage(tint, 0, 0);
    if (!preview && (tool === "landmark" || tool === "face" || !p.parts.length))
      drawLandmarks(ctx, p.landmarks, scale);
    if (cursor && !preview && (tool === "add" || tool === "erase")) {
      ctx.strokeStyle = tool === "erase" ? "#ffa8a8" : "#7ff5d7";
      ctx.lineWidth = 1.3 / scale;
      ctx.beginPath();
      ctx.arc(cursor.x, cursor.y, radius, 0, Math.PI * 2);
      ctx.stroke();
    }
  }, [
    p,
    textures,
    source,
    parameters,
    props.original,
    props.overlay,
    props.wire,
    props.solo,
    preview,
    playing,
    selected,
    tool,
    radius,
    meshes,
    frame,
    tint,
    cursor,
    scale,
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
  return (
    <div className="viewport" ref={root}>
      <div className="canvas-space">
        <div
          className="canvas-frame"
          style={{ width: p.width * scale, height: p.height * scale }}
        >
          <canvas
            aria-label="일러스트 편집 캔버스"
            ref={element}
            width={p.width}
            height={p.height}
            style={{
              width: p.width * scale,
              height: p.height * scale,
              cursor: preview
                ? "default"
                : tool === "select"
                  ? "default"
                  : "crosshair",
            }}
            onPointerDown={(e) => {
              if (preview) return;
              e.currentTarget.setPointerCapture(e.pointerId);
              props.onStart(point(e));
            }}
            onPointerMove={(e) => {
              const pt = point(e);
              setCursor(pt);
              if (e.buttons === 1 && !preview) props.onMove(pt);
            }}
            onPointerUp={(e) => {
              if (!preview) props.onEnd(point(e));
              if (e.currentTarget.hasPointerCapture(e.pointerId))
                e.currentTarget.releasePointerCapture(e.pointerId);
            }}
            onPointerCancel={(e) => props.onEnd(point(e))}
            onPointerLeave={() => setCursor(null)}
          />
          <span className="corner tl" />
          <span className="corner tr" />
          <span className="corner bl" />
          <span className="corner br" />
        </div>
      </div>
    </div>
  );
}
function drawLandmarks(
  ctx: CanvasRenderingContext2D,
  landmarks: Landmarks,
  scale: number,
) {
  const keys = Object.keys(landmarks) as LandmarkId[];
  for (const id of keys) {
    const pt = landmarks[id];
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, 5 / scale, 0, Math.PI * 2);
    ctx.fillStyle = "#151b24";
    ctx.fill();
    ctx.strokeStyle = "#61dfc1";
    ctx.lineWidth = 1.5 / scale;
    ctx.stroke();
    ctx.font = `${11 / scale}px sans-serif`;
    ctx.lineWidth = 3 / scale;
    ctx.strokeStyle = "#121923";
    ctx.strokeText(landmarkLabels[id], pt.x + 10 / scale, pt.y - 9 / scale);
    ctx.fillStyle = "#a1f0dc";
    ctx.fillText(landmarkLabels[id], pt.x + 10 / scale, pt.y - 9 / scale);
  }
}
