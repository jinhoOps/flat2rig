// B4 evaluation only: renders one exported pose (`b4_render --gl-export`) with the app's WebGL
// compositor (`src/rig/glRenderer.ts`), exactly as `renderCharacterFrame` hands it layers.
import { GlMeshCompositor, type GlLayerDrawSpec } from "../../../src/rig/glRenderer";

interface ExportedLayer {
  id: string;
  texture: string; // data URL (filled in by run.mjs)
  opacity: number;
  vertices: [number, number][];
  posed: [number, number][] | null;
  triangles: number[];
}
interface ExportedPose {
  pose: string;
  width: number;
  height: number;
  clip: Record<string, string>;
  layers: ExportedLayer[];
}

async function toCanvas(url: string, w: number, h: number): Promise<HTMLCanvasElement> {
  const img = new Image();
  img.src = url;
  await img.decode();
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  c.getContext("2d")!.drawImage(img, 0, 0);
  return c;
}

async function renderPose(doc: ExportedPose): Promise<{ png: string; renderer: string }> {
  const gl = GlMeshCompositor.create(doc.width, doc.height);
  if (!gl) throw new Error("WebGL unavailable");
  const specs: GlLayerDrawSpec[] = [];
  for (const l of doc.layers) {
    specs.push({
      cacheKey: `part:${l.id}`,
      id: l.id,
      blendMode: "normal",
      texCanvas: await toCanvas(l.texture, doc.width, doc.height),
      mesh: {
        vertices: l.vertices.map(([x, y]) => ({ x, y })),
        triangles: l.triangles,
        uv: [],
      },
      posedPoints: l.posed ? l.posed.map(([x, y]) => ({ x, y })) : null,
      alpha: l.opacity,
    });
  }
  const ok = gl.render(specs, doc.width, doc.height, doc.clip, Object.keys(doc.clip).length > 0);
  if (!ok) throw new Error("GlMeshCompositor.render returned false");
  const out = document.createElement("canvas");
  out.width = doc.width;
  out.height = doc.height;
  out.getContext("2d")!.drawImage(gl.canvas, 0, 0);
  const ctx = gl.canvas.getContext("webgl2") ?? gl.canvas.getContext("webgl");
  const dbg = ctx?.getExtension("WEBGL_debug_renderer_info");
  const renderer = ctx ? String(ctx.getParameter(dbg ? dbg.UNMASKED_RENDERER_WEBGL : ctx.RENDERER)) : "?";
  gl.dispose();
  return { png: out.toDataURL("image/png"), renderer };
}

(window as unknown as { renderPose: typeof renderPose }).renderPose = renderPose;
