import { spawn } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
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
const userDataDir = mkdtempSync(join(tmpdir(), "flat2rig-smoke-"));
const app = spawn(binary, [], {
  cwd: root,
  windowsHide: true,
  env: {
    ...process.env,
    FLAT2RIG_HEADLESS_SMOKE: "1",
    WEBVIEW2_USER_DATA_FOLDER: userDataDir,
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
  await waitFor("Boolean(document.body?.innerText?.includes('일러스트 가져오기'))");
  const initial = await evaluate(
    "(()=>{const rgb=s=>{const m=s.match(/rgba?\\((\\d+),\\s*(\\d+),\\s*(\\d+)/);return m?[+m[1],+m[2],+m[3]]:[0,0,0];};const luma=([r,g,b])=>0.2126*r+0.7152*g+0.0722*b;const panelEl=document.querySelector('.layers-panel')||document.querySelector('.inspector');return {text:document.body.innerText,tauri:!!window.__TAURI_INTERNALS__,surface:getComputedStyle(document.querySelector('.topbar')).backdropFilter,bodyBg:getComputedStyle(document.body).backgroundColor,bodyLuma:luma(rgb(getComputedStyle(document.body).backgroundColor)),panelBg:getComputedStyle(panelEl).backgroundColor,panelLuma:luma(rgb(getComputedStyle(panelEl).backgroundColor))};})()",
  );
  if (!initial.tauri) throw new Error("Native bridge not active");
  if (initial.bodyLuma < 200 || initial.panelLuma < 200)
    throw new Error(`Expected Daylight Light Mode default, got ${JSON.stringify({ bodyBg: initial.bodyBg, panelBg: initial.panelBg })}`);
  console.log("Native WebView, Daylight Light Mode, and Rust bridge loaded");
  const { root: documentRoot } = await rpc("DOM.getDocument");
  const { nodeId } = await rpc("DOM.querySelector", {
    nodeId: documentRoot.nodeId,
    selector: 'input[aria-label="이미지 파일 입력"]',
  });
  await rpc("DOM.setFileInputFiles", { nodeId, files: [fixture] });
  await waitFor(
    "document.querySelector('.project-title').textContent.includes('native-input') && !document.querySelector('.busy-overlay')",
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
    "document.querySelectorAll('.layer').length>=18 && !document.querySelector('.busy-overlay')",
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
    `(async()=>{const invoke=window.__TAURI_INTERNALS__.invoke;const a=${JSON.stringify(args)};const r=await invoke('generate_parts',a);const prop=await invoke('propose_parts',{rgba:a.rgba,width:128,height:128,landmarks:a.landmarks,tolerance:32,detailMode:'parent'});const ids=${JSON.stringify(["hair_back", "body", "neck", "face", "eyeR_white", "eyeR_iris", "eyeR_lower", "eyeR_upper", "eyeL_white", "eyeL_iris", "eyeL_lower", "eyeL_upper", "browR", "browL", "mouth_inner", "mouth_lower", "mouth_upper", "hair_front"])};const rig=await invoke('build_rig',{parts:r.masks.map((mask,i)=>({id:ids[i],mask})),width:128,height:128,landmarks:a.landmarks});const meshed=rig.find(p=>p.mesh.triangles.length>0);const fill=await invoke('fill_part',{rgba:a.rgba,mask:r.masks[3],width:128,height:128,radius:12});const occ=await invoke('fill_part_occlusion',{rgba:a.rgba,visibleMask:r.masks[4],width:128,height:128,radius:6,partId:'eyeR_white'});const models=await invoke('get_model_status');return {parts:r.masks.length,parentParts:prop.partIds.length,parentHasEyeR:prop.partIds.includes('eyeR_all'),rigParts:rig.length,vertices:rig.reduce((n,p)=>n+p.mesh.vertices.length,0),samples:meshed?.samples.length??0,extremePosesTested:meshed?.quality?.extremePosesTested??0,filledBytes:atob(fill.mask).length,occFillKind:occ.fillKind,occWarning:!!occ.warning,modelsCount:models.length,provider:r.provider,modelHash:r.modelHash};})()`,
  );
  if (
    native.parts !== 18 ||
    !native.parentHasEyeR ||
    native.parentParts >= 18 ||
    native.rigParts !== 18 ||
    native.extremePosesTested !== 12 ||
    native.filledBytes !== 128 * 128 ||
    native.occFillKind !== "neighbor" ||
    !native.occWarning ||
    native.modelsCount !== 5 ||
    native.samples < 10
  )
    throw new Error(`Native command contract failed: ${JSON.stringify(native)}`);
  console.log("ort, delaunator, glam, occlusion and model registry commands passed");
  const checkMinFont = () =>
    evaluate(
      "(()=>{let minFont=999,minEl='';for(const el of document.querySelectorAll('*')){if(el.closest('.sr-only')||el.classList.contains('sr-only'))continue;const st=getComputedStyle(el);if(st.display==='none'||st.visibility==='hidden'||parseFloat(st.opacity)===0)continue;const r=el.getBoundingClientRect();if(r.width<=2||r.height<=2)continue;const hasText=[...el.childNodes].some(n=>n.nodeType===3&&n.nodeValue.trim().length>0);if(!hasText)continue;const fs=parseFloat(st.fontSize);if(fs<minFont){minFont=fs;minEl=el.tagName+'.'+el.className+':'+el.textContent.trim().slice(0,20);}}return {minFont,minEl};})()",
    );

  // Step 2: 표정 (Expressions)
  await evaluate(
    "[...document.querySelectorAll('[role=tab]')].find(b=>b.textContent.includes('표정')).click()",
  );
  await waitFor(
    "document.querySelectorAll('.rig-card').length>=8 && document.querySelectorAll('.rig-card-thumb img').length>=4 && !document.body.innerText.includes('메쉬 계산 중')",
  );
  const exprFont = await checkMinFont();
  if (exprFont.minFont < 12)
    throw new Error(`Expression font < 12px: ${JSON.stringify(exprFont)}`);
  await evaluate(
    "[...document.querySelectorAll('.rig-card')].find(b=>b.textContent.includes('미소'))?.click()",
  );
  await delay(120);
  const exprScreenshot = await rpc("Page.captureScreenshot", { format: "png" });
  writeFileSync(
    resolve(dir, "native-ui.png"),
    Buffer.from(exprScreenshot.data, "base64"),
  );

  // Step 3: 조정 (Adjust)
  await evaluate(
    "[...document.querySelectorAll('[role=tab]')].find(b=>b.textContent.includes('조정')).click()",
  );
  await waitFor("document.querySelectorAll('.rig-card').length>=4");
  const adjustFont = await checkMinFont();
  if (adjustFont.minFont < 12)
    throw new Error(`Adjust font < 12px: ${JSON.stringify(adjustFont)}`);

  // Step 4: 재생 (Play)
  await evaluate(
    "[...document.querySelectorAll('[role=tab]')].find(b=>b.textContent.includes('재생')).click()",
  );
  await waitFor(
    "!!document.querySelector('.canvas-frame.clean') && document.body.innerText.includes('라이브 재생 시작')",
  );
  const playFont = await checkMinFont();
  if (playFont.minFont < 12)
    throw new Error(`Play font < 12px: ${JSON.stringify(playFont)}`);

  // Step 5: 고급 (Advanced Parts + Mesh + PSD)
  await evaluate(
    "[...document.querySelectorAll('[role=tab]')].find(b=>b.textContent.includes('고급')).click()",
  );
  await waitFor("!!document.querySelector('.handoff-card')");
  await evaluate(
    "[...document.querySelectorAll('button')].find(b=>b.textContent.includes('메쉬 · 파라미터 확인')).click()",
  );
  await waitFor(
    "document.querySelectorAll('input[type=range]').length>=22 && !document.body.innerText.includes('메쉬 계산 중')",
  );
  const typography = await evaluate(
    "(()=>{let minFont=999,minEl='';for(const el of document.querySelectorAll('*')){if(el.closest('.sr-only')||el.classList.contains('sr-only'))continue;const st=getComputedStyle(el);if(st.display==='none'||st.visibility==='hidden'||parseFloat(st.opacity)===0)continue;const r=el.getBoundingClientRect();if(r.width<=2||r.height<=2)continue;const hasText=[...el.childNodes].some(n=>n.nodeType===3&&n.nodeValue.trim().length>0);if(!hasText)continue;const fs=parseFloat(st.fontSize);if(fs<minFont){minFont=fs;minEl=el.tagName+'.'+el.className+':'+el.textContent.trim().slice(0,20);}}return {minFont,minEl,groupHeaders:document.querySelectorAll('.layer-group-header').length,paramTabs:document.querySelectorAll('.param-group-tab').length};})()",
  );
  if (typography.minFont < 12)
    throw new Error(`Minimum UI font size below 12px: ${JSON.stringify(typography)}`);
  if (typography.paramTabs < 7)
    throw new Error(`Expected parameter group tabs, got ${JSON.stringify(typography)}`);
  console.log("Daylight font-size floor (>=12px), 4+1 tabs, thumbnails, clean stage, and parameter tabs verified:", JSON.stringify(typography));
  await rpc("Emulation.setEmulatedMedia", {
    features: [{ name: "prefers-reduced-transparency", value: "reduce" }],
  });
  const reduced = await evaluate(
    "getComputedStyle(document.querySelector('.topbar')).backdropFilter",
  );
  if (reduced !== "none")
    throw new Error("Reduced transparency fallback failed");
  await rpc("Emulation.setEmulatedMedia", { features: [] });
  if (exceptions.length) throw new Error(JSON.stringify(exceptions));
  const report = {
    passed: true,
    dragDrop: true,
    invalidDropRejected: true,
    workflowTabs: ["prepare", "expressions", "adjust", "play", "advanced"],
    initialSurface: initial.surface,
    lightMode: { bodyBg: initial.bodyBg, panelBg: initial.panelBg },
    typography,
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
          "({text:document.body?.innerText,html:document.documentElement.outerHTML.slice(0,1800)})",
        ),
      );
      if (exceptions.length) console.error("RUNTIME_EXCEPTIONS:", JSON.stringify(exceptions));
    } catch {}
  }
  throw error;
} finally {
  socket?.close();
  app.kill();
  try {
    rmSync(userDataDir, { recursive: true, force: true });
  } catch {}
}
