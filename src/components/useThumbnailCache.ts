import { useEffect, useState } from "react";
import type { Mesh, Project } from "../types";
import type { SubstituteLayerTexture } from "../rig/render";
import {
  renderRigThumbnailDataUrl,
  type ThumbnailSpec,
} from "../rig/thumbnails";

export function useThumbnailCache(
  enabled: boolean,
  project: Project | null,
  specs: ThumbnailSpec[],
  partTextures: Map<string, HTMLCanvasElement>,
  meshes: Map<string, Mesh>,
  substituteTextures: Map<string, SubstituteLayerTexture>,
  cacheKey: string,
): Record<string, string> {
  const [thumbs, setThumbs] = useState<Record<string, string>>({});

  useEffect(() => {
    if (!enabled || !project || !partTextures.size || !specs.length) {
      return;
    }
    let cancelled = false;
    let index = 0;
    const nextMap: Record<string, string> = {};
    let handle = 0;

    const step = () => {
      if (cancelled) return;
      const spec = specs[index++];
      if (spec) {
        try {
          nextMap[spec.id] = renderRigThumbnailDataUrl(
            spec,
            partTextures,
            meshes,
            substituteTextures,
            88,
          );
        } catch {
          // Keep missing thumbnail graceful on partial state
        }
      }
      if (index < specs.length) {
        handle = requestAnimationFrame(step);
      } else if (!cancelled) {
        setThumbs({ ...nextMap });
      }
    };

    // Publish partial progress every 4 items so cards feel instant
    const timer = window.setTimeout(() => {
      if (!cancelled) handle = requestAnimationFrame(step);
    }, 30);

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
      if (handle) cancelAnimationFrame(handle);
    };
  }, [enabled, project, partTextures, meshes, substituteTextures, cacheKey]);

  return thumbs;
}
