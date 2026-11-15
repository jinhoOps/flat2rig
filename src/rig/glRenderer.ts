import type { BlendMode, Mesh, Point } from "../types";

export interface GlLayerDrawSpec {
  cacheKey: string;
  id: string;
  blendMode?: BlendMode;
  texCanvas: HTMLCanvasElement;
  maskRef?: Uint8Array;
  fillRef?: Uint8Array;
  mesh?: Mesh;
  posedPoints?: Point[] | null;
  alpha: number;
}

interface CachedTexture {
  tex: WebGLTexture;
  canvas: HTMLCanvasElement;
  maskRef?: Uint8Array;
  fillRef?: Uint8Array;
  width: number;
  height: number;
}

interface ScleraFbo {
  fbo: WebGLFramebuffer;
  tex: WebGLTexture;
  width: number;
  height: number;
}

const VERT_SRC = `
attribute vec2 a_dst;
attribute vec2 a_src;
uniform vec2 u_canvasSize;
uniform vec2 u_texSize;
varying vec2 v_uv;
varying vec2 v_screenUv;
void main() {
  vec2 clip = vec2(
    (a_dst.x / u_canvasSize.x) * 2.0 - 1.0,
    1.0 - (a_dst.y / u_canvasSize.y) * 2.0
  );
  gl_Position = vec4(clip, 0.0, 1.0);
  v_uv = vec2(a_src.x / u_texSize.x, a_src.y / u_texSize.y);
  v_screenUv = clip * 0.5 + 0.5;
}
`;

const FRAG_SRC = `
precision mediump float;
uniform sampler2D u_tex;
uniform sampler2D u_mask;
uniform float u_hasMask;
uniform float u_alpha;
varying vec2 v_uv;
varying vec2 v_screenUv;
void main() {
  vec4 color = texture2D(u_tex, v_uv) * u_alpha;
  if (u_hasMask > 0.5) {
    color *= texture2D(u_mask, v_screenUv).a;
  }
  gl_FragColor = color;
}
`;

function compileShader(
  gl: WebGLRenderingContext,
  type: number,
  source: string,
): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) return null;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}

export class GlMeshCompositor {
  readonly canvas: HTMLCanvasElement;
  private gl: WebGLRenderingContext;
  private program: WebGLProgram;
  private aDstLoc: number;
  private aSrcLoc: number;
  private uCanvasSizeLoc: WebGLUniformLocation | null;
  private uTexSizeLoc: WebGLUniformLocation | null;
  private uTexLoc: WebGLUniformLocation | null;
  private uMaskLoc: WebGLUniformLocation | null;
  private uHasMaskLoc: WebGLUniformLocation | null;
  private uAlphaLoc: WebGLUniformLocation | null;
  private vbo: WebGLBuffer;
  private ibo: WebGLBuffer;
  private vBuf = new Float32Array(4096);
  private iBuf = new Uint16Array(4096);
  private textures = new Map<string, CachedTexture>();
  private scleraFbos = new Map<string, ScleraFbo>();
  private disposed = false;

  private constructor(
    canvas: HTMLCanvasElement,
    gl: WebGLRenderingContext,
    program: WebGLProgram,
    vbo: WebGLBuffer,
    ibo: WebGLBuffer,
  ) {
    this.canvas = canvas;
    this.gl = gl;
    this.program = program;
    this.vbo = vbo;
    this.ibo = ibo;
    this.aDstLoc = gl.getAttribLocation(program, "a_dst");
    this.aSrcLoc = gl.getAttribLocation(program, "a_src");
    this.uCanvasSizeLoc = gl.getUniformLocation(program, "u_canvasSize");
    this.uTexSizeLoc = gl.getUniformLocation(program, "u_texSize");
    this.uTexLoc = gl.getUniformLocation(program, "u_tex");
    this.uMaskLoc = gl.getUniformLocation(program, "u_mask");
    this.uHasMaskLoc = gl.getUniformLocation(program, "u_hasMask");
    this.uAlphaLoc = gl.getUniformLocation(program, "u_alpha");
  }

  static create(width: number, height: number): GlMeshCompositor | null {
    if (typeof document === "undefined") return null;
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, width);
    canvas.height = Math.max(1, height);
    const attrs: WebGLContextAttributes = {
      alpha: true,
      premultipliedAlpha: true,
      antialias: false,
      depth: false,
      stencil: false,
      preserveDrawingBuffer: true,
    };
    const gl =
      (canvas.getContext("webgl2", attrs) as WebGLRenderingContext | null) ??
      (canvas.getContext("webgl", attrs) as WebGLRenderingContext | null);
    if (!gl) return null;

    const vs = compileShader(gl, gl.VERTEX_SHADER, VERT_SRC);
    const fs = compileShader(gl, gl.FRAGMENT_SHADER, FRAG_SRC);
    if (!vs || !fs) {
      if (vs) gl.deleteShader(vs);
      if (fs) gl.deleteShader(fs);
      return null;
    }
    const program = gl.createProgram();
    if (!program) {
      gl.deleteShader(vs);
      gl.deleteShader(fs);
      return null;
    }
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      gl.deleteProgram(program);
      return null;
    }
    const vbo = gl.createBuffer();
    const ibo = gl.createBuffer();
    if (!vbo || !ibo) {
      if (vbo) gl.deleteBuffer(vbo);
      if (ibo) gl.deleteBuffer(ibo);
      gl.deleteProgram(program);
      return null;
    }
    return new GlMeshCompositor(canvas, gl, program, vbo, ibo);
  }

  resize(width: number, height: number): void {
    const w = Math.max(1, width);
    const h = Math.max(1, height);
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
  }

  private uploadTexture(spec: GlLayerDrawSpec): WebGLTexture | null {
    const gl = this.gl;
    const src = spec.texCanvas;
    let cached = this.textures.get(spec.cacheKey);
    if (
      cached &&
      cached.canvas === src &&
      cached.maskRef === spec.maskRef &&
      cached.fillRef === spec.fillRef &&
      cached.width === src.width &&
      cached.height === src.height
    ) {
      return cached.tex;
    }
    if (!cached) {
      const tex = gl.createTexture();
      if (!tex) return null;
      cached = {
        tex,
        canvas: src,
        maskRef: spec.maskRef,
        fillRef: spec.fillRef,
        width: src.width,
        height: src.height,
      };
      this.textures.set(spec.cacheKey, cached);
    } else {
      cached.canvas = src;
      cached.maskRef = spec.maskRef;
      cached.fillRef = spec.fillRef;
      cached.width = src.width;
      cached.height = src.height;
    }
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, cached.tex);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, 1);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, src);
    return cached.tex;
  }

  private getScleraFbo(id: string, width: number, height: number): ScleraFbo | null {
    const gl = this.gl;
    let entry = this.scleraFbos.get(id);
    if (entry && entry.width === width && entry.height === height) {
      return entry;
    }
    if (!entry) {
      const fbo = gl.createFramebuffer();
      const tex = gl.createTexture();
      if (!fbo || !tex) {
        if (fbo) gl.deleteFramebuffer(fbo);
        if (tex) gl.deleteTexture(tex);
        return null;
      }
      entry = { fbo, tex, width, height };
      this.scleraFbos.set(id, entry);
    } else {
      entry.width = width;
      entry.height = height;
    }
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, entry.tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA,
      width,
      height,
      0,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      null,
    );
    gl.bindFramebuffer(gl.FRAMEBUFFER, entry.fbo);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER,
      gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D,
      entry.tex,
      0,
    );
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.activeTexture(gl.TEXTURE0);
    return entry;
  }

  private uploadGeometry(
    spec: GlLayerDrawSpec,
    width: number,
    height: number,
  ): number {
    const gl = this.gl;
    const mesh = spec.mesh;
    const posed = spec.posedPoints;
    if (mesh && posed && mesh.triangles.length > 0) {
      const vCount = mesh.vertices.length;
      if (vCount > 65535) return -1;
      const neededV = vCount * 4;
      if (this.vBuf.length < neededV) {
        this.vBuf = new Float32Array(Math.max(neededV, this.vBuf.length * 2));
      }
      for (let i = 0; i < vCount; i++) {
        const dst = posed[i];
        const src = mesh.vertices[i];
        const off = i * 4;
        this.vBuf[off] = dst.x;
        this.vBuf[off + 1] = dst.y;
        this.vBuf[off + 2] = src.x;
        this.vBuf[off + 3] = src.y;
      }
      const iCount = mesh.triangles.length;
      if (this.iBuf.length < iCount) {
        this.iBuf = new Uint16Array(Math.max(iCount, this.iBuf.length * 2));
      }
      for (let i = 0; i < iCount; i++) {
        this.iBuf[i] = mesh.triangles[i];
      }
      gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
      gl.bufferData(
        gl.ARRAY_BUFFER,
        this.vBuf.subarray(0, neededV),
        gl.DYNAMIC_DRAW,
      );
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ibo);
      gl.bufferData(
        gl.ELEMENT_ARRAY_BUFFER,
        this.iBuf.subarray(0, iCount),
        gl.DYNAMIC_DRAW,
      );
      return iCount;
    }

    const w = width;
    const h = height;
    const tw = spec.texCanvas.width;
    const th = spec.texCanvas.height;
    const qv = this.vBuf;
    qv[0] = 0; qv[1] = 0; qv[2] = 0; qv[3] = 0;
    qv[4] = w; qv[5] = 0; qv[6] = tw; qv[7] = 0;
    qv[8] = w; qv[9] = h; qv[10] = tw; qv[11] = th;
    qv[12] = 0; qv[13] = h; qv[14] = 0; qv[15] = th;
    this.iBuf[0] = 0; this.iBuf[1] = 1; this.iBuf[2] = 2;
    this.iBuf[3] = 0; this.iBuf[4] = 2; this.iBuf[5] = 3;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
    gl.bufferData(gl.ARRAY_BUFFER, this.vBuf.subarray(0, 16), gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ibo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, this.iBuf.subarray(0, 6), gl.DYNAMIC_DRAW);
    return 6;
  }

  render(
    layers: GlLayerDrawSpec[],
    width: number,
    height: number,
    clippingTargets: Record<string, string>,
    clipActive: boolean,
  ): boolean {
    if (this.disposed) return false;
    const gl = this.gl;
    if (gl.isContextLost()) return false;
    for (const spec of layers) {
      if (spec.blendMode === "multiply") return false;
      if (spec.mesh && spec.posedPoints && spec.mesh.vertices.length > 65535) {
        return false;
      }
    }

    this.resize(width, height);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, width, height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendEquation(gl.FUNC_ADD);
    gl.useProgram(this.program);
    gl.uniform2f(this.uCanvasSizeLoc, width, height);

    const scleraSet = new Set<string>([
      "eyeL_white",
      "eyeR_white",
      ...Object.values(clippingTargets),
    ]);
    const renderedSclera = new Map<string, ScleraFbo>();

    for (const spec of layers) {
      if (spec.alpha <= 0.01) continue;
      const tex = this.uploadTexture(spec);
      if (!tex) return false;
      const count = this.uploadGeometry(spec, width, height);
      if (count <= 0) continue;

      gl.vertexAttribPointer(this.aDstLoc, 2, gl.FLOAT, false, 16, 0);
      gl.enableVertexAttribArray(this.aDstLoc);
      gl.vertexAttribPointer(this.aSrcLoc, 2, gl.FLOAT, false, 16, 8);
      gl.enableVertexAttribArray(this.aSrcLoc);
      gl.uniform2f(this.uTexSizeLoc, spec.texCanvas.width, spec.texCanvas.height);

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.uniform1i(this.uTexLoc, 0);

      const maskTargetId = clipActive ? clippingTargets[spec.id] : undefined;
      const maskFbo = maskTargetId ? renderedSclera.get(maskTargetId) : undefined;
      if (maskFbo) {
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, maskFbo.tex);
        gl.uniform1i(this.uMaskLoc, 1);
        gl.uniform1f(this.uHasMaskLoc, 1);
      } else {
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, null);
        gl.uniform1f(this.uHasMaskLoc, 0);
      }

      if (spec.blendMode === "additive") {
        gl.blendFunc(gl.ONE, gl.ONE);
      } else {
        gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      }
      gl.uniform1f(this.uAlphaLoc, spec.alpha);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.drawElements(gl.TRIANGLES, count, gl.UNSIGNED_SHORT, 0);

      if (clipActive && scleraSet.has(spec.id)) {
        const fboEntry = this.getScleraFbo(spec.id, width, height);
        if (fboEntry) {
          gl.activeTexture(gl.TEXTURE1);
          gl.bindTexture(gl.TEXTURE_2D, null);
          gl.activeTexture(gl.TEXTURE0);
          gl.bindTexture(gl.TEXTURE_2D, tex);
          gl.uniform1i(this.uTexLoc, 0);
          gl.bindFramebuffer(gl.FRAMEBUFFER, fboEntry.fbo);
          gl.clearColor(0, 0, 0, 0);
          gl.clear(gl.COLOR_BUFFER_BIT);
          gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
          gl.uniform1f(this.uHasMaskLoc, 0);
          gl.uniform1f(this.uAlphaLoc, 1);
          gl.drawElements(gl.TRIANGLES, count, gl.UNSIGNED_SHORT, 0);
          gl.bindFramebuffer(gl.FRAMEBUFFER, null);
          renderedSclera.set(spec.id, fboEntry);
        }
      }
    }

    return true;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    const gl = this.gl;
    for (const c of this.textures.values()) gl.deleteTexture(c.tex);
    this.textures.clear();
    for (const f of this.scleraFbos.values()) {
      gl.deleteFramebuffer(f.fbo);
      gl.deleteTexture(f.tex);
    }
    this.scleraFbos.clear();
    gl.deleteBuffer(this.vbo);
    gl.deleteBuffer(this.ibo);
    gl.deleteProgram(this.program);
  }
}
