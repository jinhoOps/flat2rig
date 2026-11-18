/**
 * T11a: an original 16×12 pixel cat (side view, facing right) with a 4-frame run cycle, drawn for
 * flat2rig's progress bar. Rows are plain strings so the art is reviewable in the diff; each
 * character is one pixel ('.' = transparent). Frames share the body and differ only in the legs;
 * frames 1 and 3 also bob the body down by one pixel.
 */
export const CAT_W = 16;
export const CAT_H = 12;

export const CAT_PALETTE: Record<string, string> = {
  o: "#f0a04b", // ginger coat
  d: "#b9662a", // stripes, tail, paws
  w: "#ffe8c9", // cream muzzle and belly
  k: "#2a1d17", // eye
  p: "#f58fa8", // nose
};

const BODY = [
  "..........o...o.",
  "..........oo.oo.",
  "..d.......ooooo.",
  ".d.......oookooo",
  ".d.dddddoooooowp",
  "..dooooooooowww.",
  "..ooodoodoooww..",
  "..oooooooooow...",
  "..wwwwwwwwww....",
];

const LEGS = [
  // 0: full stretch — back legs reach back, front legs reach forward
  [".oo........oo...", "oo..........oo..", "d............dd."],
  // 1: gather — all four paws under the body
  ["...oo....oo.....", "...oo....oo.....", "...dd....dd....."],
  // 2: push-off — back legs still under, front legs reaching
  ["..oo......oo....", "..o........oo...", ".dd.........dd.."],
  // 3: cross — legs pass each other mid-stride
  ["....oo..oo......", ".....oooo.......", ".....d..d......."],
];

/** Vertical bob per frame (pixels). */
export const CAT_BOB = [0, 1, 0, 1];

export const CAT_FRAMES: string[][] = LEGS.map((legs) => [...BODY, ...legs]);

export interface PixelRun {
  x: number;
  y: number;
  w: number;
  color: string;
}

/** Merges horizontal runs of one colour so each frame is a handful of `<rect>`s. */
export function frameRuns(frame: readonly string[], bob = 0): PixelRun[] {
  const runs: PixelRun[] = [];
  frame.forEach((row, y) => {
    let x = 0;
    while (x < row.length) {
      const ch = row[x];
      if (ch === ".") {
        x++;
        continue;
      }
      let end = x + 1;
      while (end < row.length && row[end] === ch) end++;
      runs.push({ x, y: y + bob, w: end - x, color: CAT_PALETTE[ch] ?? "#000" });
      x = end;
    }
  });
  return runs;
}
