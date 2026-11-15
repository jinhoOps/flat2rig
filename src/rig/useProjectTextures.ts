import { useEffect, useRef, useState } from "react";
import { compositePartTexture } from "../engine";
import { loadImage, pixels, pixelsCanvas } from "../images";
import type { Part, Project } from "../types";
import {
  buildProjectSubstituteTextures,
  type SubstituteLayerTexture,
} from "./render";

export interface ProjectTextureSet {
  sourceImage?: HTMLImageElement;
  partTextures: Map<string, HTMLCanvasElement>;
  substituteTextures: Map<string, SubstituteLayerTexture>;
}

export function useProjectTextures(project: Project | null): ProjectTextureSet {
  const [sourceImage, setSourceImage] = useState<HTMLImageElement>();
  const [partTextures, setPartTextures] = useState<Map<string, HTMLCanvasElement>>(
    () => new Map(),
  );
  const [substituteTextures, setSubstituteTextures] = useState<
    Map<string, SubstituteLayerTexture>
  >(() => new Map());
  const textureCache = useRef(
    new Map<string, { part: Part; canvas: HTMLCanvasElement }>(),
  );

  useEffect(() => {
    textureCache.current.clear();
    if (!project) {
      setSourceImage(undefined);
      setPartTextures(new Map());
      setSubstituteTextures(new Map());
      return;
    }
    let active = true;
    loadImage(project.source)
      .then((img) => {
        if (active) setSourceImage(img);
      })
      .catch(() => {});
    return () => {
      active = false;
    };
  }, [project?.source]);

  useEffect(() => {
    if (!project || !sourceImage || sourceImage.src !== project.source) {
      if (!project) setPartTextures(new Map());
      return;
    }
    let active = true;
    const data = pixels(sourceImage, project.width, project.height);
    Promise.all(
      project.parts.map(async (part) => {
        const cached = textureCache.current.get(part.id);
        if (cached?.part === part) return [part.id, cached.canvas] as const;
        const texPixels = part.texture
          ? pixels(await loadImage(part.texture), project.width, project.height)
          : undefined;
        const result = pixelsCanvas(
          compositePartTexture(part, data, texPixels),
          project.width,
          project.height,
        );
        if (active) textureCache.current.set(part.id, { part, canvas: result });
        return [part.id, result] as const;
      }),
    ).then((entries) => {
      if (active) setPartTextures(new Map(entries));
    });
    return () => {
      active = false;
    };
  }, [project, sourceImage, project?.parts, project?.width, project?.height]);

  useEffect(() => {
    if (!project?.shapeAssets) {
      setSubstituteTextures(new Map());
      return;
    }
    let active = true;
    buildProjectSubstituteTextures(project).then((entries) => {
      if (active) setSubstituteTextures(entries);
    });
    return () => {
      active = false;
    };
  }, [project?.shapeAssets, project?.width, project?.height, project?.landmarks]);

  return { sourceImage, partTextures, substituteTextures };
}
