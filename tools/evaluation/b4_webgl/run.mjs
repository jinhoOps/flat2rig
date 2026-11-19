// B4 evaluation only: app WebGL render of exported poses in headless Chrome.
//
//   npx esbuild tools/evaluation/b4_webgl/page.ts --bundle --format=iife --outfile=artifacts/b4/webgl/page.js
//   node tools/evaluation/b4_webgl/run.mjs artifacts/b4/webgl/page.js EXPORT_DIR OUT_DIR [CHROME_EXE]
//
// EXPORT_DIR holds `<pose>.json` from `cargo run --example b4_render -- ... --gl-export`; each is
// rendered by `GlMeshCompositor` and saved as OUT_DIR/<pose>.png (+ OUT_DIR/renderer.json).
import { chromium } from "playwright-core";
import fs from "node:fs";
import path from "node:path";

const [pageJs, exportDir, outDir, exe = "C:/Program Files/Google/Chrome/Application/chrome.exe"] = process.argv.slice(2);
fs.mkdirSync(outDir, { recursive: true });
const browser = await chromium.launch({
  executablePath: exe,
  headless: true,
  // B4_GPU=1: ANGLE on the real GPU (D3D11) instead of SwiftShader.
  args: process.env.B4_GPU
    ? ["--enable-webgl", "--ignore-gpu-blocklist", "--use-angle=d3d11", "--enable-gpu"]
    : ["--enable-webgl", "--ignore-gpu-blocklist", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
});
const page = await browser.newPage();
page.on("console", (m) => console.log("[page]", m.text()));
page.on("pageerror", (e) => console.log("[pageerror]", e.message));
await page.setContent("<!doctype html><html><body></body></html>");
await page.addScriptTag({ content: fs.readFileSync(pageJs, "utf8") });
const files = fs.readdirSync(exportDir).filter((f) => f.endsWith(".json") && f !== "summary.json");
let renderer = "";
for (const f of files) {
  const doc = JSON.parse(fs.readFileSync(path.join(exportDir, f), "utf8"));
  for (const l of doc.layers) {
    l.texture = `data:image/png;base64,${fs.readFileSync(l.texture).toString("base64")}`;
  }
  const res = await page.evaluate((d) => window.renderPose(d), doc);
  renderer = res.renderer;
  fs.writeFileSync(path.join(outDir, f.replace(/\.json$/, ".png")), Buffer.from(res.png.split(",")[1], "base64"));
  console.log("rendered", doc.pose);
}
fs.writeFileSync(path.join(outDir, "renderer.json"), JSON.stringify({ renderer, chrome: browser.version() }, null, 2));
await browser.close();
