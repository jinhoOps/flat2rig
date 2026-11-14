# flat2rig logo and motion

The mark combines a simple head-and-shoulders outline with a triangular mesh and four control points. Eyes, mouth, hair and expression are omitted. The outline uses Graphite (#161920); the mesh and the wordmark's 2 use Studio Cobalt (#2456D6).

The header, welcome screen and /brand.html share the React geometry. scripts/brand-assets.mjs generates transparent mark.svg, dark-background mark-reverse.svg and the pale app tile public/logo.svg from that geometry.

The 10.8-second loop keeps the silhouette still and gently skews the internal mesh before returning to rest. There is no cursor or part separation. It is a logo animation, not a demonstration of working face detection. The stop control and reduced-motion setting show the neutral pose. Stage suspends playback offscreen and in hidden tabs.

The Stageshow tilted-stack reference supplied the centered composition and retained-element approach. No faux product UI is added. Stage size is 400 x 220, with a 358 x 200 compact layout. See src/brand/REHEARSAL.md for validation.
