---
name: cubism-psd-handoff
description: >-
  Use when modifying PSD export, multi-layer part/hair grouping, fillMask compositing,
  parameter CSV generation, handoff.md, or validation checks in src/io.ts and related export tests.
---

# Cubism PSD & Handoff Verification

`flat2rig`는 `ag-psd`로 레이어드 PSD를 생성하고 `fflate`로 인계 ZIP을 묶는다. Cubism Editor에 실제로 가져올 때 레이어 깨짐이나 과장된 호환성 표기가 나지 않도록 아래 절차를 따른다.

## Steps

1. **Check layer hierarchy, Z-order, and naming**
   - In [src/io.ts](../../src/io.ts), inspect `exportPsd`, `partCanvases`, `handoff`, and `exportBundle`.
   - Preserve bottom-to-top Z-order (`hair_back*` -> `body` -> `neck` -> `face` -> eyes/brows/mouth -> `hair_side*` -> `hair_front*`, respecting user `compositeOrder` / `shiftPart`).
   - If grouping variable hair parts into PSD folders (e.g., `Hair/Back`, `Hair/Side`, `Hair/Front`), ensure `ag-psd` group nodes keep `normal` blend mode and valid child canvases, and do **not** invent unverified "Cubism magic layer IDs" claiming automatic rigging.
   - Exclude `status === "missing"` and empty-mask parts from PSD output; ignore viewport-only `visible`/solo flags during export.

2. **Check pixel format, fillMask compositing, and canvas offsets**
   - Keep RGB 8-bit `normal` layers with clean alpha.
   - When `fillMask` or filled textures exist, composite the part's exportable texture (visible + reconstructed under-occlusion pixels) without mutating the project's non-overlapping `visibleMask` contract.
   - Keep `trimImageData: true`, `noBackground: true` in `writePsdUint8Array`, and verify trimmed layers preserve exact canvas offsets.

3. **Check parameter CSV & handoff documentation honesty**
   - Without a user-supplied Editor CSV template, emit `parameters-reference.csv` and explicitly state it is a reference list (not an official import CSV).
   - With a user-supplied CSV template (`readCsvTemplate` / `cubismCsv`), preserve template headers, column order, and `type` values, leaving `GUID` empty.
   - Keep `rig-draft.json`, `handoff.md`, and `validation.json` explicit about boundaries: PSD transfers images/layers only; meshes, deformers, clipping, and physics are NOT auto-created in Cubism.
   - Keep `validation.json` distinction between `readPsd` structural re-read checks and actual manual Cubism Editor verification.

4. **Verify build and export tests**
   - Run frontend unit/pipeline tests: `npm test` (inspect `tests/pipeline.test.ts` for PSD re-read, ZIP contents, and schema v1/v2 migration checks).
   - Run TypeScript & Vite build: `npm run build`.
   - Confirm `readPsd` re-read check passes for both default 18 parts and variable/multi-layer hair projects.
