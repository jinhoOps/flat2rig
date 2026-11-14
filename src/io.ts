import { zip, unzipSync, strToU8, strFromU8, type Zippable } from "fflate";
import { writePsdUint8Array, readPsd } from "ag-psd";
import {
  HAIR_ROLE_META,
  PART_DEFS,
  PARAM_DEFS,
  type HairRole,
  type Project,
  type Part,
} from "./types";
import { bounds, effectiveMask } from "./engine";
import { canvas, partCanvases, pngBytes } from "./images";
import { buildRig } from "./native";

export const safeName = (name: string) =>
  name.replace(/[^\p{L}\p{N}_.-]+/gu, "_").slice(0, 90) || "character";
export function pack(files: Zippable): Promise<Uint8Array> {
  return new Promise((resolve, reject) =>
    zip(files, { level: 6 }, (error, result) =>
      error ? reject(error) : resolve(result),
    ),
  );
}
export async function saveProject(project: Project) {
  const files: Zippable = {};
  const { parts, ...meta } = project;
  const serializedParts = parts.map(({ mask: _, fillMask: __, ...p }) => p);
  files["project.json"] = strToU8(
    JSON.stringify({
      ...meta,
      schemaVersion: 2,
      parts: serializedParts,
    }),
  );
  for (const p of parts) {
    files[`masks/${p.id}.bin`] = p.mask;
    if (p.fillMask) {
      files[`masks/${p.id}.fill.bin`] = p.fillMask;
    }
  }
  return pack(files);
}
function validDataURL(v: unknown): v is string {
  return (
    typeof v === "string" &&
    /^data:image\/(png|jpeg);base64,[A-Za-z0-9+/=]+$/.test(v)
  );
}
function migrateV1ToV2(p: Record<string, unknown>, parts: Part[]): Project {
  const landmarks = p.landmarks as Project["landmarks"];
  return {
    schemaVersion: 2,
    name: p.name as string,
    width: p.width as number,
    height: p.height as number,
    originalWidth: (p.originalWidth as number) || (p.width as number),
    originalHeight: (p.originalHeight as number) || (p.height as number),
    source: p.source as string,
    original: p.original as string,
    workToOriginalTransform: {
      matrix2: [
        ((p.originalWidth as number) || (p.width as number)) / (p.width as number),
        0,
        0,
        ((p.originalHeight as number) || (p.height as number)) / (p.height as number),
        ],
      translation: [0, 0],
    },
    landmarks,
    landmarkProvenance: {
      eyeL: "manual",
      eyeR: "manual",
      mouth: "manual",
      chin: "manual",
      crown: "manual",
    },
    parts: parts.map((part, index) => ({
      ...part,
      provenance: part.provenance ?? "geometry",
      fillKind: part.fillKind ?? (part.generated ? "neighbor" : "none"),
      compositeOrder: part.compositeOrder ?? index,
    })),
    revision: 0,
    provider: (p.provider as string) || "migrated-v1",
    modelHash: typeof p.modelHash === "string" ? p.modelHash : undefined,
    analysisStatus: "idle",
    createdAt: (p.createdAt as string) || new Date().toISOString(),
  };
}
export function loadProject(bytes: Uint8Array): Project {
  if (bytes.length > 160 * 1024 * 1024)
    throw new Error("프로젝트 파일이 너무 큽니다.");
  let total = 0;
  const files = unzipSync(bytes, {
    filter: (f) => {
      total += f.originalSize;
      if (total > 450 * 1024 * 1024 || f.originalSize > 100 * 1024 * 1024)
        throw new Error("프로젝트 압축 해제 한도를 초과했습니다.");
      return true;
    },
  });
  if (!files["project.json"]) throw new Error("flat2rig 프로젝트가 아닙니다.");
  const p = JSON.parse(strFromU8(files["project.json"]));
  if (
    (p.schemaVersion !== 1 && p.schemaVersion !== 2) ||
    !Number.isInteger(p.width) ||
    !Number.isInteger(p.height) ||
    p.width < 1 ||
    p.height < 1 ||
    p.width > 2048 ||
    p.height > 2048 ||
    !validDataURL(p.source) ||
    !validDataURL(p.original)
  )
    throw new Error("지원하지 않는 프로젝트 형식입니다.");
  if (
    typeof p.name !== "string" ||
    p.name.length > 200 ||
    !Array.isArray(p.parts) ||
    p.parts.length > 64
  )
    throw new Error("잘못된 프로젝트 정보입니다.");
  const ids = new Set<string>();
  const validProvenances = ["model", "prompted", "geometry", "manual"];
  const validFillKinds = ["none", "neighbor", "external"];
  for (let idx = 0; idx < p.parts.length; idx++) {
    const part = p.parts[idx];
    if (
      typeof part.id !== "string" ||
      !/^[\w-]{1,80}$/.test(part.id) ||
      ids.has(part.id) ||
      typeof part.name !== "string" ||
      typeof part.group !== "string" ||
      !/^#[0-9a-f]{6}$/i.test(part.color) ||
      !["draft", "approved", "missing"].includes(part.status) ||
      typeof part.visible !== "boolean" ||
      (part.texture && !validDataURL(part.texture)) ||
      (part.provenance && !validProvenances.includes(part.provenance)) ||
      (part.fillKind && !validFillKinds.includes(part.fillKind))
    )
      throw new Error("잘못된 파츠 정보입니다.");
    ids.add(part.id);
    const mask = files[`masks/${part.id}.bin`];
    if (!mask || mask.length !== p.width * p.height)
      throw new Error("파츠 마스크 크기가 일치하지 않습니다.");
    part.mask = mask;
    const fillMask = files[`masks/${part.id}.fill.bin`];
    if (fillMask) {
      if (fillMask.length !== p.width * p.height)
        throw new Error("가림 보완 마스크 크기가 일치하지 않습니다.");
      part.fillMask = fillMask;
    }
    if (part.compositeOrder === undefined) {
      part.compositeOrder = idx;
    }
  }
  for (const id of ["eyeL", "eyeR", "mouth", "chin", "crown"]) {
    const pt = p.landmarks?.[id];
    if (
      !pt ||
      !Number.isFinite(pt.x) ||
      !Number.isFinite(pt.y) ||
      pt.x < 0 ||
      pt.y < 0 ||
      pt.x >= p.width ||
      pt.y >= p.height
    )
      throw new Error("잘못된 랜드마크입니다.");
  }
  if (p.schemaVersion === 1) {
    return migrateV1ToV2(p, p.parts as Part[]);
  }
  p.revision = 0;
  return p as Project;
}
function quote(value: unknown) {
  return `"${String(value ?? "").replace(/"/g, '""')}"`;
}
export function referenceCsv() {
  return (
    "\uFEFF" +
    [
      [
        "ID",
        "Name",
        "Minimum",
        "Default",
        "Maximum",
        "Preview minimum",
        "Preview maximum",
      ],
      ...PARAM_DEFS.map((p) => [
        p.id,
        p.name,
        p.min,
        p.value,
        p.max,
        p.previewMin,
        p.previewMax,
      ]),
    ]
      .map((row) => row.map(quote).join(","))
      .join("\r\n")
  );
}
export function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [],
    field = "",
    quoted = false;
  text = text.replace(/^\uFEFF/, "");
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c === '"') {
      if (quoted && text[i + 1] === '"') {
        field += '"';
        i++;
      } else quoted = !quoted;
    } else if (c === "," && !quoted) {
      row.push(field);
      field = "";
    } else if ((c === "\r" || c === "\n") && !quoted) {
      if (c === "\r" && text[i + 1] === "\n") i++;
      row.push(field);
      if (row.some(Boolean)) rows.push(row);
      row = [];
      field = "";
    } else field += c;
  }
  if (quoted) throw new Error("CSV 따옴표가 닫히지 않았습니다.");
  if (field || row.length) {
    row.push(field);
    rows.push(row);
  }
  return rows;
}
export interface CsvTemplate {
  headers: string[];
  sample: string[];
  columns: Record<string, number>;
}
export function readCsvTemplate(text: string): CsvTemplate {
  const rows = parseCsv(text);
  const aliases: Record<string, string[]> = {
    guid: ["guid"],
    type: ["type", "종류", "타입", "タイプ"],
    id: ["id"],
    name: ["name", "이름", "名前"],
    group: ["group", "그룹", "グループ"],
    min: ["minimum", "minimumvalue", "min", "최소값", "最小値"],
    value: ["default", "defaultvalue", "기본값", "デフォルト値"],
    max: ["maximum", "maximumvalue", "max", "최대값", "最大値"],
    repeat: ["repeat", "반복", "繰り返し"],
    description: ["description", "설명", "説明"],
  };
  const headerIndex = rows.findIndex(
    (row) =>
      row.some((c) => c.trim().toLowerCase() === "guid") &&
      row.some((c) => c.trim().toLowerCase() === "id"),
  );
  if (headerIndex < 0)
    throw new Error(
      "GUID/ID 헤더를 찾을 수 없습니다. Cubism 파라미터 일괄 설정 CSV를 사용하세요.",
    );
  const headers = rows[headerIndex],
    columns: Record<string, number> = {};
  for (const [key, names] of Object.entries(aliases)) {
    const i = headers.findIndex((h) =>
      names.includes(h.toLowerCase().replace(/[\s_-]/g, "")),
    );
    if (i >= 0) columns[key] = i;
  }
  if (
    ["guid", "type", "id", "name", "min", "value", "max"].some(
      (c) => columns[c] === undefined,
    )
  )
    throw new Error(
      "파라미터 CSV의 필수 열이 없습니다. 영어로 내보낸 CSV를 사용하세요.",
    );
  const sample = rows
    .slice(headerIndex + 1)
    .find((row) => /^Param/i.test(row[columns.id] || "") && row[columns.type]);
  if (!sample)
    throw new Error("Param으로 시작하는 ID의 파라미터 샘플 행이 필요합니다.");
  return { headers, sample, columns };
}
export function cubismCsv(template: CsvTemplate) {
  const { headers, sample, columns: c } = template;
  const rows = PARAM_DEFS.map((p) => {
    const r = headers.map(() => "");
    const vals: Record<string, unknown> = {
      guid: "",
      type: sample[c.type],
      id: p.id,
      name: p.name,
      group: "",
      min: p.min,
      value: p.value,
      max: p.max,
      repeat: "false",
      description: "flat2rig draft; keyforms must be created in Cubism",
    };
    for (const [key, value] of Object.entries(vals))
      if (c[key] !== undefined) r[c[key]] = String(value);
    return r;
  });
  return (
    "\uFEFF" +
    [headers, ...rows].map((row) => row.map(quote).join(",")).join("\r\n")
  );
}
function orderedEntries(parts: Part[]) {
  return parts
    .map((part, index) => ({ part, index }))
    .sort(
      (a, b) =>
        (a.part.compositeOrder ?? a.index) - (b.part.compositeOrder ?? b.index),
    );
}

async function maskPng(mask: Uint8Array, w: number, h: number) {
  const m = canvas(w, h);
  const d = m.getContext("2d")!.createImageData(w, h);
  for (let j = 0; j < mask.length; j++) {
    d.data[j * 4] = d.data[j * 4 + 1] = d.data[j * 4 + 2] = mask[j];
    d.data[j * 4 + 3] = 255;
  }
  m.getContext("2d")!.putImageData(d, 0, 0);
  return pngBytes(m);
}

export async function exportPsd(
  project: Project,
  canvases?: HTMLCanvasElement[],
) {
  const images = canvases ?? (await partCanvases(project)),
    composite = canvas(project.width, project.height),
    ctx = composite.getContext("2d")!;
  // Visibility is a viewport aid. Export all non-missing approved/draft parts visibly in compositeOrder.
  const children = orderedEntries(project.parts).flatMap(({ part: p, index: i }) => {
    const eff = effectiveMask(p);
    if (
      p.status === "missing" ||
      !bounds(eff, project.width, project.height)
    )
      return [];
    ctx.drawImage(images[i], 0, 0);
    return [
      {
        name: `${p.id} · ${p.name}`,
        left: 0,
        top: 0,
        canvas: images[i],
        blendMode: "normal" as const,
        hidden: false,
      },
    ];
  });
  if (!children.length)
    throw new Error("내보낼 파츠가 없습니다. 먼저 분리 초안을 생성하세요.");
  const bytes = writePsdUint8Array(
    {
      width: project.width,
      height: project.height,
      canvas: composite,
      children,
    },
    { trimImageData: true, noBackground: true, generateThumbnail: false },
  );
  const check = readPsd(bytes, {
    skipLayerImageData: true,
    skipCompositeImageData: true,
    skipThumbnail: true,
  });
  if (
    check.width !== project.width ||
    check.children?.length !== children.length
  )
    throw new Error("PSD 구조 검증 실패");
  return bytes;
}
export function handoff(project: Project, csv: boolean) {
  const unresolved = project.parts.filter((p) => p.status !== "approved");
  const modelHashesText =
    project.modelHashes && Object.keys(project.modelHashes).length
      ? Object.entries(project.modelHashes)
          .map(([k, v]) => `${k}=${v.slice(0, 12)}`)
          .join(", ")
      : project.modelHash || "사용하지 않음";
  return `# ${project.name} — Cubism 작업 인계\n\n- 캔버스: ${project.width} × ${project.height}px (원본 ${project.originalWidth} × ${project.originalHeight})\n- 생성 경로: ${project.provider}\n- 모델 SHA-256: ${modelHashesText}\n- 좌표: 원본 작업 캔버스 좌상단 (0,0), X 오른쪽, Y 아래. 좌우는 캐릭터 기준.\n- 미확인 파츠: ${unresolved.map((p) => p.id + " (" + p.status + ")").join(", ") || "없음"}\n\n## 가져오기\n1. character_import.psd를 Cubism Editor에서 열어 새 모델을 만듭니다. 레이어 순서와 알파를 확인합니다.\n2. ${csv ? "parameters.csv는 불러온 Cubism CSV 샘플의 형식으로 생성했습니다. 백업 후 파라미터 일괄 설정에서 UTF-8로 가져오고, 전체 삭제 옵션은 끕니다. 실제 버전 호환성은 확인이 필요합니다." : "parameters-reference.csv는 참고용 목록입니다. 공식 가져오기 파일이 아닙니다. Cubism에서 파라미터를 수동 생성하거나 flat2rig에 Editor의 CSV 샘플을 불러오세요."}\n3. guides/landmarks.png와 rig-draft.json을 참고하여 ArtMesh와 디포머를 정리합니다. 자체 JSON은 Editor로 직접 가져오지 않습니다.\n4. 목을 피벗으로 HeadRotation → HeadXY → Eyes/Mouth/Hair 디포머를 구성합니다. 눈 흰자를 클리핑 대상으로 활용하되 원화에 맞게 조정합니다.\n5. 눈 개폐 0/1, 입 개폐 0/1, 머리 X/Y -10/0/10, Z -15/0/15의 변형 키를 수동으로 작성합니다. 전체 -30~30 품질은 보장하지 않습니다.\n6. 작은 조합 입력부터 구멍, 눈꺼풀 경계, 입 안 노출을 확인하고 가림 영역을 보완합니다.\n7. 머리카락 흔들림 변형 키를 먼저 만든 뒤 Cubism에서 물리를 설정하고 .cmo3로 저장합니다.\n\n## 알려진 한계\n- 얼굴 검출(YOLOv3)·28점 랜드마크(HRNetV2)에서 crown 좌표는 기하 추정값입니다.\n- 파츠 분리는 MobileSAM + 랜드마크 하이브리드 분할 또는 기하 폴백이며, 수동 보정(브러시·클릭 프롬프트) 및 검토 상태(status)와 출처(provenance)를 분리해 기록합니다.\n- 가림 보완(neighbor)은 가시 마스크와 분리 저장되는 경계 픽셀 외삽으로, 생성형 인페인팅이 아니며 가려진 눈·입·머리카락 내부 디테일은 복원되지 않습니다.\n- PSD는 이미지만 전달합니다. 메쉬, 키폼, 클리핑, 물리는 자동 적용되지 않습니다.\n- 레이어 숨김/솔로는 편집기 표시 보조입니다. 내보내기는 미해당 파츠를 제외한 전체 파츠를 표시합니다.\n- 자동 PSD 재읽기는 통과했으나 실제 Cubism Editor에서의 가져오기는 별도 검증해야 합니다.\n\n## 파츠\n${orderedEntries(project.parts).map(({ part: p }, i) => `${i}. ${p.id} — ${p.name} [검토:${p.status}, 출처:${p.provenance ?? "geometry"}, 보완:${p.fillKind ?? "none"}]`).join("\n")}\n\n공식 참고: https://docs.live2d.com/en/cubism-editor-manual/psd-import/\n`;
}
export async function exportBundle(
  project: Project,
  template: CsvTemplate | null,
) {
  const images = await partCanvases(project),
    rig = await buildRig(
      project.parts,
      project.width,
      project.height,
      project.landmarks,
    );
  const ordered = orderedEntries(project.parts);
  const files: Zippable = {
    "character_import.psd": await exportPsd(project, images),
    "project.flat2rig": await saveProject(project),
    "handoff.md": strToU8(handoff(project, !!template)),
    [template ? "parameters.csv" : "parameters-reference.csv"]: strToU8(
      template ? cubismCsv(template) : referenceCsv(),
    ),
  };
  const guide = canvas(project.width, project.height),
    ctx = guide.getContext("2d")!;
  ordered.forEach(({ part, index }) => {
    if (part.status !== "missing") ctx.drawImage(images[index], 0, 0);
  });
  for (const [name, p] of Object.entries(project.landmarks)) {
    ctx.fillStyle = "#39d6b0";
    ctx.beginPath();
    ctx.arc(p.x, p.y, Math.max(4, project.width / 160), 0, Math.PI * 2);
    ctx.fill();
    ctx.font = `${Math.max(12, project.width / 65)}px sans-serif`;
    ctx.strokeStyle = "#132027";
    ctx.lineWidth = 3;
    ctx.strokeText(name, p.x + 10, p.y - 8);
    ctx.fillText(name, p.x + 10, p.y - 8);
  }
  files["guides/landmarks.png"] = await pngBytes(guide);
  for (let i = 0; i < project.parts.length; i++) {
    const p = project.parts[i];
    files[`parts/${p.id}.png`] = await pngBytes(images[i]);
    const visiblePng = await maskPng(p.mask, project.width, project.height);
    files[`masks/${p.id}.png`] = visiblePng;
    if (p.fillMask?.some((v) => v > 0)) {
      files[`masks/${p.id}.visible.png`] = visiblePng;
      files[`masks/${p.id}.fill.png`] = await maskPng(
        p.fillMask,
        project.width,
        project.height,
      );
    }
  }
  files["rig-draft.json"] = strToU8(
    JSON.stringify(
      {
        schemaVersion: 1,
        format: "flat2rig-rig-draft",
        directlyImportableInCubism: false,
        coordinateSystem: "pixels-top-left-y-down",
        width: project.width,
        height: project.height,
        landmarks: project.landmarks,
        parameters: PARAM_DEFS,
        deformerSuggestions: [
          "HeadRotation",
          "HeadXY",
          "Eyes",
          "Mouth",
          "Hair",
        ],
        parts: rig,
      },
      null,
      2,
    ),
  );
  const provenanceSummary = project.parts.reduce(
    (acc, p) => {
      const k = p.provenance ?? "geometry";
      acc[k] = (acc[k] ?? 0) + 1;
      return acc;
    },
    {} as Record<string, number>,
  );
  const occlusionSummary = project.parts
    .filter(
      (p) =>
        p.fillMask?.some((v) => v > 0) ||
        (p.fillKind && p.fillKind !== "none"),
    )
    .map((p) => ({
      id: p.id,
      fillKind: p.fillKind ?? "neighbor",
      filledPixels: p.fillMask ? p.fillMask.filter((v) => v > 0).length : 0,
    }));
  const meshWarnings = rig.flatMap((p) => {
    const warns: string[] = [];
    if (p.mesh.triangles.length === 0) warns.push(`${p.id}: no triangles`);
    if (p.quality?.warnings?.length) warns.push(...p.quality.warnings);
    return warns;
  });
  files["validation.json"] = strToU8(
    JSON.stringify(
      {
        revision: project.revision,
        createdAt: new Date().toISOString(),
        psdStructureRoundTrip: "passed",
        cubismEditorImport: "not-tested",
        csv: template ? "sample-derived-not-editor-tested" : "reference-only",
        unapproved: project.parts
          .filter((p) => p.status === "draft")
          .map((p) => p.id),
        missing: project.parts
          .filter((p) => p.status === "missing")
          .map((p) => p.id),
        sourceProvider: project.provider,
        engine: project.provider.includes("browser")
          ? "typescript-browser-fallback"
          : "rust-ort-delaunator-glam",
        sourceModelHash: project.modelHash,
        modelHashes: project.modelHashes ?? {},
        postprocessVersions: project.postprocessVersions ?? {},
        provenanceSummary,
        occlusionSummary,
        meshWarnings,
      },
      null,
      2,
    ),
  );
  return pack(files);
}
export function newPart(w: number, h: number): Part {
  const [, , group, color] = PART_DEFS[3];
  return {
    id: `custom_${crypto.randomUUID().slice(0, 8)}`,
    name: "새 파츠",
    group,
    color,
    role: "other",
    visible: true,
    status: "draft",
    mask: new Uint8Array(w * h),
  };
}

export function newHairPart(
  role: HairRole,
  existingParts: Part[],
  w: number,
  h: number,
): Part {
  const existingIds = new Set(existingParts.map((p) => p.id));
  let idx = role === "hair_front" || role === "hair_back" ? 2 : 1;
  let id = idx === 1 ? role : `${role}_${idx}`;
  while (existingIds.has(id)) {
    idx++;
    id = `${role}_${idx}`;
  }
  const meta = HAIR_ROLE_META[role];
  const label = idx === 1 ? meta.name : `${meta.name} ${idx}`;
  return {
    id,
    name: label,
    group: meta.group,
    color: meta.color,
    role,
    visible: true,
    status: "draft",
    provenance: "manual",
    mask: new Uint8Array(w * h),
  };
}

export function insertHairPartInOrder(parts: Part[], added: Part): Part[] {
  const role = added.role;
  const list = [...parts];
  let insertAt = list.length;
  if (role === "hair_back") {
    let lastBack = -1;
    list.forEach((p, i) => {
      if (p.id.startsWith("hair_back") || p.role === "hair_back") lastBack = i;
    });
    insertAt = lastBack >= 0 ? lastBack + 1 : 0;
  } else if (role === "hair_side_L" || role === "hair_side_R") {
    const firstFront = list.findIndex(
      (p) => p.id.startsWith("hair_front") || p.role === "hair_front",
    );
    insertAt = firstFront >= 0 ? firstFront : list.length;
  }
  list.splice(insertAt, 0, added);
  return list.map((p, idx) => ({ ...p, compositeOrder: idx }));
}
