import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

// Derive the transparent brand marks from the source character logo.
const logo = readFileSync(resolve("public/logo.svg"), "utf8");
const transparent = logo
  .replace('width="512" height="512"', 'width="80" height="80"')
  .replace(/<title>[\s\S]*?<\/title>/, "<title>flat2rig character mark</title>")
  .replace(/\s*<desc>[\s\S]*?<\/desc>/, "")
  .replace(/\s*<defs>[\s\S]*?<\/defs>/, "")
  .replace(/\s*<rect x="12" y="12" width="1000" height="1000" rx="210"[^>]*\/>/, "");
const reverse = transparent
  .replaceAll("#17151b", "#f6f7fa")
  .replaceAll('fill="#fff"', 'fill="#161920"')
  .replaceAll('stroke="#fff"', 'stroke="#161920"')
  .replaceAll("#2868d8", "#8eafff")
  .replaceAll("#3979e6", "#8eafff")
  .replace("<title>flat2rig character mark</title>", "<title>flat2rig reverse character mark</title>");
writeFileSync(resolve("public/brand/mark.svg"), transparent);
writeFileSync(resolve("public/brand/mark-reverse.svg"), reverse);
console.log("Generated transparent and reverse marks from public/logo.svg");
