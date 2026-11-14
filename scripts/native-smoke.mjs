import { spawn } from "node:child_process";
import { existsSync, mkdirSync, writeFileSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { createCanvas } from "@napi-rs/canvas";
import WebSocket from "ws";

const root = resolve(import.meta.dirname, "..");
const binary = resolve(
  root,
  process.env.FLAT2RIG_TEST_BINARY || "src-tauri/target/debug/flat2rig.exe",
);
if (!existsSync(binary))
  throw new Error(
    "Build the desktop app first: npm run desktop:build -- --debug",
  );
const dir = resolve(root, "artifacts");
mkdirSync(dir, { recursive: true });
const c = createCanvas(128, 128),
  ctx = c.getContext("2d");
ctx.fillStyle = "#fff";
ctx.fillRect(0, 0, 128, 128);
ctx.fillStyle = "#6080a0";
ctx.fillRect(22, 10, 84, 118);
ctx.fillStyle = "#f0ccba";
ctx.fillRect(38, 20, 52, 55);
const fixture = resolve(dir, "native-input.png");
writeFileSync(fixture, c.toBuffer("image/png"));
const port = 19228;
const app = spawn(binary, [], {
  cwd: root,
  windowsHide: true,
  env: {
    ...process.env,
    FLAT2RIG_HEADLESS_SMOKE: "1",
    WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS: `--remote-debugging-port=${port}`,
  },
  stdio: "pipe",
});
let logs = "";
app.stderr.on("data", (data) => {
  logs += String(data);
});
app.stdout.on("data", (data) => {
  logs += String(data);
});
let socket,
  serial = 0;
const pending = new Map(),
  exceptions = [];
const delay = (ms) => new Promise((r) => setTimeout(r, ms));
const rpc = (method, params = {}) =>
  new Promise((resolve, reject) => {
    const id = ++serial;
    const timer = setTimeout(() => {
      pending.delete(id);
      reject(new Error(`Timeout: ${method}`));
    }, 60000);
    pending.set(id, {
      resolve: (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      reject,
    });
    socket.send(JSON.stringify({ id, method, params }));
  });
const evaluate = async (expression) => {
  const r = await rpc("Runtime.evaluate", {
    expression,
    awaitPromise: true,
    returnByValue: true,
  });
  if (r.exceptionDetails) throw new Error(JSON.stringify(r.exceptionDetails));
  return r.result.value;
};
const waitFor = async (expression) => {
  for (let i = 0; i < 160; i++) {
    if (await evaluate(expression)) return;
    await delay(200);
  }
  throw new Error(`UI timeout: ${expression}`);
};
try {
  let page;
  for (let i = 0; i < 100; i++) {
    try {
      const pages = await (await fetch(`http://127.0.0.1:${port}/json`)).json();
      page = pages.find((p) => p.type === "page" && p.webSocketDebuggerUrl);
      if (page) break;
    } catch {}
    await delay(200);
  }
  if (!page) throw new Error(`WebView debug target unavailable. ${logs}`);
  socket = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((r, j) => {
    socket.once("open", r);
    socket.once("error", j);
  });
  socket.on("message", (data) => {
    const m = JSON.parse(String(data));
    if (m.method === "Runtime.exceptionThrown") exceptions.push(m.params);
    if (m.id && pending.has(m.id)) {
      const item = pending.get(m.id);
      pending.delete(m.id);
      m.error
        ? item.reject(new Error(JSON.stringify(m.error)))
        : item.resolve(m.result);
    }
  });
  await rpc("Runtime.enable");
  await rpc("Page.enable");
  await waitFor("document.body.innerText.includes('일러스트 가져오기')");
  const initial = await evaluate(
    "({text:document.body.innerText,tauri:!!window.__TAURI_INTERNALS__,surface:getComputedStyle(document.querySelector('.topbar')).backdropFilter})",
  );
  if (!initial.tauri) throw new Error("Native bridge not active");
  console.log("Native WebView and Rust bridge loaded");
  const { root: documentRoot } = await rpc("DOM.getDocument");
  const { nodeId } = await rpc("DOM.querySelector", {
    nodeId: documentRoot.nodeId,
    selector: 'input[aria-label="이미지 파일 입력"]',
  });
  await rpc("DOM.setFileInputFiles", { nodeId, files: [fixture] });
  await waitFor(
    "document.querySelector('.project-title').textContent.includes('native-input')",
  );
  const png = readFileSync(fixture).toString("base64");
  await evaluate(
    `(()=>{const dt=new DataTransfer();dt.items.add(new File(['invalid'],'notes.txt',{type:'text/plain'}));document.querySelector('.app').dispatchEvent(new DragEvent('drop',{bubbles:true,cancelable:true,dataTransfer:dt}));})()`,
  );
  await waitFor(
    "document.querySelector('[role=alert]')?.textContent.includes('PNG 또는 JPG')",
  );
  await evaluate(
    `(()=>{const dt=new DataTransfer();dt.items.add(new File([Uint8Array.from(atob('${png}'),c=>c.charCodeAt(0))],'dropped-character.png',{type:'image/png'}));window.__smokeDrop=dt;document.querySelector('.app').dispatchEvent(new DragEvent('dragenter',{bubbles:true,cancelable:true,dataTransfer:dt}));})()`,
  );
  await waitFor("!!document.querySelector('.drop-target')");
  await evaluate(
    "document.querySelector('.app').dispatchEvent(new DragEvent('drop',{bubbles:true,cancelable:true,dataTransfer:window.__smokeDrop}));delete window.__smokeDrop",
  );
  await waitFor(
    "document.querySelector('.project-title').textContent.includes('dropped-character') && !document.querySelector('.drop-target') && !document.querySelector('.busy-overlay')",
  );
  console.log(
    "Drag/drop import, drop highlight and invalid type rejection passed",
  );
  await evaluate(
    "[...document.querySelectorAll('button')].find(b=>b.textContent.trim()==='파츠 초안 생성').click()",
  );
  await waitFor(
    "document.querySelectorAll('.layer').length===18 && !document.querySelector('.busy-overlay')",
  );
  console.log("Image import and Rust part generation passed");
  const lm = {
    eyeL: { x: 78, y: 40 },
    eyeR: { x: 50, y: 40 },
    mouth: { x: 64, y: 53 },
    chin: { x: 64, y: 64 },
    crown: { x: 64, y: 10 },
  };
  const rgba = Buffer.from(ctx.getImageData(0, 0, 128, 128).data).toString(
    "base64",
  );
  const model = resolve(root, "models/u2netp.onnx");
  const args = {
    rgba,
    width: 128,
    height: 128,
    landmarks: lm,
    tolerance: 32,
    modelPath: existsSync(model) ? model : null,
  };
  const native = await evaluate(
    `(async()=>{const invoke=window.__TAURI_INTERNALS__.invoke;const a=${JSON.stringify(args)};const r=await invoke('generate_parts',a);const ids=${JSON.stringify(["hair_back", "body", "neck", "face", "eyeR_white", "eyeR_iris", "eyeR_lower", "eyeR_upper", "eyeL_white", "eyeL_iris", "eyeL_lower", "eyeL_upper", "browR", "browL", "mouth_inner", "mouth_lower", "mouth_upper", "hair_front"])};const rig=await invoke('build_rig',{parts:r.masks.map((mask,i)=>({id:ids[i],mask})),width:128,height:128,landmarks:a.landmarks});const fill=await invoke('fill_part',{rgba:a.rgba,mask:r.masks[3],width:128,height:128,radius:12});return {parts:r.masks.length,rigParts:rig.length,vertices:rig.reduce((n,p)=>n+p.mesh.vertices.length,0),samples:rig[3].samples.length,filledBytes:atob(fill.mask).length,provider:r.provider,modelHash:r.modelHash};})()`,
  );
  if (
    native.parts !== 18 ||
    native.rigParts !== 18 ||
    native.filledBytes !== 128 * 128 ||
    native.samples < 10
  )
    throw new Error("Native command contract failed");
  console.log("ort, delaunator, glam and fill commands passed");
  await evaluate(
    "[...document.querySelectorAll('[role=tab]')].find(b=>b.textContent.includes('리깅')).click()",
  );
  await waitFor(
    "document.querySelectorAll('input[type=range]').length===7 && !document.body.innerText.includes('메쉬 계산 중')",
  );
  await rpc("Emulation.setEmulatedMedia", {
    features: [{ name: "prefers-reduced-transparency", value: "reduce" }],
  });
  const reduced = await evaluate(
    "getComputedStyle(document.querySelector('.topbar')).backdropFilter",
  );
  if (reduced !== "none")
    throw new Error("Reduced transparency fallback failed");
  await rpc("Emulation.setEmulatedMedia", { features: [] });
  const screenshot = await rpc("Page.captureScreenshot", { format: "png" });
  writeFileSync(
    resolve(dir, "native-ui.png"),
    Buffer.from(screenshot.data, "base64"),
  );
  if (exceptions.length) throw new Error(JSON.stringify(exceptions));
  const report = {
    passed: true,
    dragDrop: true,
    invalidDropRejected: true,
    initialSurface: initial.surface,
    reducedTransparency: reduced,
    native,
    exceptions,
  };
  writeFileSync(
    resolve(dir, "native-ui-smoke.json"),
    JSON.stringify(report, null, 2),
  );
  console.log(JSON.stringify(report, null, 2));
} catch (error) {
  console.error(logs);
  if (socket?.readyState === WebSocket.OPEN) {
    try {
      console.error(
        await evaluate(
          "({text:document.body.innerText,html:document.documentElement.outerHTML.slice(0,1800)})",
        ),
      );
    } catch {}
  }
  throw error;
} finally {
  socket?.close();
  app.kill();
}
