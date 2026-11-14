import { separate } from "./engine";
self.onmessage = (
  event: MessageEvent<{
    id: number;
    pixels: Uint8ClampedArray;
    width: number;
    height: number;
    landmarks: import("./types").Landmarks;
    tolerance: number;
  }>,
) => {
  const { id, pixels, width, height, landmarks, tolerance } = event.data;
  try {
    const parts = separate(pixels, width, height, landmarks, tolerance);
    self.postMessage(
      { id, parts },
      { transfer: parts.map((p) => p.mask.buffer) },
    );
  } catch (error) {
    self.postMessage({ id, error: String(error) });
  }
};
