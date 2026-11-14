// Serialize the production React mark so SVG downloads and native icons match it.
// Run with the Vite dev server available; no external images or fonts are fetched.
import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { launch } from "../tools/stageshow/scripts/browser.mjs";

const url = process.argv[2] ?? "http://127.0.0.1:1420/brand.html";
const browser = await launch();
try {
  const page = await browser.newPage();
  await page.goto(url);
  const markup = await page.locator(".brand-mark svg").evaluate((element) => {
    const svg = element.cloneNode(true);
    svg.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    svg.setAttribute("width", "80");
    svg.setAttribute("height", "80");
    svg.setAttribute("color", "#161920");
    svg.removeAttribute("aria-hidden");
    svg.querySelector(".rig-mesh").setAttribute("opacity", ".65");
    for (const node of [svg, ...svg.querySelectorAll("*")]) {
      node.removeAttribute("class");
      node.removeAttribute("data-on");
    }
    const title = document.createElementNS(
      "http://www.w3.org/2000/svg",
      "title",
    );
    title.textContent = "flat2rig · character rig";
    svg.prepend(title);
    return svg.outerHTML;
  });
  mkdirSync(resolve("public/brand"), { recursive: true });
  writeFileSync(resolve("public/brand/mark.svg"), markup + "\n");
  const inverse = markup
    .replaceAll("#161920", "__GRAPHITE__")
    .replaceAll("#f6f7fa", "#161920")
    .replaceAll("__GRAPHITE__", "#f6f7fa")
    .replaceAll("#2456d6", "#7fa3ff");
  writeFileSync(resolve("public/brand/mark-reverse.svg"), inverse + "\n");
  const tile = markup
    .replace('width="80" height="80"', 'width="512" height="512"')
    .replace(
      /<title>/,
      '<rect x="1.5" y="1.5" width="77" height="77" rx="18" fill="#f6f7fa" stroke="#d3d9e3" stroke-width="1.5"/><title>',
    );
  writeFileSync(resolve("public/logo.svg"), tile + "\n");
  console.log(
    "Generated character mark, reverse mark and app tile from Brand.tsx",
  );
} finally {
  await browser.close();
}
