# Rehearsal: flat2rig mark

Gold: Stageshow tilted-stack (captured as artifacts/brand/gold.png). Kept a centered focal object and retained elements; replaced the paper composition with the requested original logo. The deliberately simpler silhouette follows the user's request to remove detailed facial features.

Beats: neutral mesh, gentle skew, return. Loop: 10.8 seconds. The human outline stays fixed. No cursor, hair separation, facial expression or simulated face-detection result.

Captured in production Chrome: 400 x 220 stage, 358 x 200 compact stage in a 390px viewport, marked states, reduced-motion poster. Files: artifacts/brand/simple-wide.png, simple-compact.png, simple-states.png, simple-poster.png and associated JSON reports.

Gate: all three motion captures ended with `rehearsal passed (necessary, not sufficient: references/verify.md and the fresh eye in review.md still stand)` and no visual findings. The wide capture completed with exit 0 and a 19.4ms maximum sampling gap. Compact and states captures have sampling gaps (up to 46.4ms and 42.5ms); their coverage is inconclusive despite a clean visual gate. These are not reported as full motion passes.

Review: inspected the poster and every frame of the wide, compact and states contact sheets. No clipping, feature swaps or abrupt loop reset visible. The mesh remains contained within the head in both poses. The mark also reads at header size in the Orca preview screenshot. Self-review only; no independent reviewer.

Controls: used Orca browser to switch to the neutral poster and restart playback; labels changed correctly. Reduced-motion poster captured separately. TypeScript/production build and all 10 existing frontend tests passed. Native icon assets regenerated from the same SVG.

Not checked: installed native window/taskbar icons, physical phone, offscreen/hidden-tab runtime behavior beyond Stage's implementation, every animation frame inside the reported sampling gaps, and a screenshot of the full editor host (the Orca tab was closed before capture).
