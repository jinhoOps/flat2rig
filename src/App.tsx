import { Fragment, useCallback, useEffect, useRef, useState } from "react";
import {
  AlertCircle,
  ArrowDown,
  ArrowUp,
  Box,
  Check,
  CheckCircle2,
  ChevronRight,
  CircleHelp,
  Crosshair,
  Download,
  Eraser,
  Eye,
  EyeOff,
  FileImage,
  FolderOpen,
  GripVertical,
  ImagePlus,
  Layers,
  Loader2,
  Maximize,
  MousePointer2,
  Move,
  Pause,
  Play,
  Plus,
  Redo2,
  RefreshCw,
  Save,
  ScanFace,
  Settings,
  SlidersHorizontal,
  Smile,
  Sparkles,
  Undo2,
  Upload,
  WandSparkles,
  X,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import type {
  BlendMode,
  FillKind,
  HairRole,
  LandmarkId,
  LandmarkProvenance,
  Mesh,
  ModelStatusEntry,
  Parameters,
  Part,
  PartProvenance,
  Point,
  Project,
  PromptPoint,
  TemplateStyleSelection,
  Tool,
} from "./types";
import {
  isHairPart,
  PARAM_DEFS,
  PARAM_GROUPS,
  REST,
  TEMPLATE_STYLE_CATALOG,
} from "./types";
import { bounds, clamp, faceBoxLandmarks, paint } from "./engine";
import { loadImage, pixels, pixelsCanvas, readImage } from "./images";
import {
  analyzeFace,
  buildRig,
  cancelAnalysis,
  checkAppUpdate,
  chooseOnnxModel,
  desktop,
  detectLandmarksForFace,
  downloadModel,
  fillOcclusionNative,
  getAppVersion,
  getModelStatus,
  installAppUpdate,
  proposeParts,
  refinePartPrompt,
  relaunchApp,
  saveFile,
  setModelPath as setRegisteredModelPath,
  type FaceAnalysisResponse,
  type MeshQualityReport,
  type RefinePromptResult,
  type UpdateCheckResult,
} from "./native";
import {
  exportBundle,
  exportPsd,
  insertHairPartInOrder,
  isPreservedPart,
  loadProject,
  mergePreservedParts,
  newDynamicPart,
  newHairPart,
  newPart,
  readCsvTemplate,
  safeName,
  saveProject,
  type CsvTemplate,
} from "./io";
import Viewport from "./Viewport";
import { synthesizeShapeAssetsForProject } from "./rig/shapes";
import {
  buildAllPartMeshes,
  buildAllPartTextures,
  buildProjectSubstituteTextures,
  renderDeterministicSnapshot,
} from "./rig/render";
import { selectExpressionPreset } from "./rig/state";
import { useProjectTextures } from "./rig/useProjectTextures";
import ExpressionPanel from "./components/ExpressionPanel";
import AdjustPanel from "./components/AdjustPanel";
import PlaybackPanel from "./components/PlaybackPanel";
import {
  demoFile,
  downloadSpriteGuide,
  renderSpriteGuideCanvas,
} from "./demo";
import { BrandMark, BrandWordmark } from "./brand/Brand";
import BrandLoop from "./brand/BrandLoop";

const ROLE_LABELS: Record<string, string> = {
  face_detector: "얼굴 검출 (YOLOv3)",
  landmark_28: "28점 랜드마크 (HRNetV2)",
  background: "배경 분리 (U2NetP)",
  part_encoder: "파츠 인코더 (MobileSAM)",
  part_decoder: "파츠 디코더 (MobileSAM)",
};

function partProvenanceLabel(p?: PartProvenance): string {
  if (p === "model") return "모델";
  if (p === "prompted") return "프롬프트";
  if (p === "manual") return "수동";
  return "기하";
}

function diagnosticLabel(d: string): string {
  if (d === "empty") return "공백";
  if (d === "lr_swapped") return "좌우반전";
  if (d === "low_confidence") return "저신뢰";
  if (d.startsWith("fragmented")) return "파편화";
  if (d.startsWith("background_leak")) return "배경누출";
  return d;
}

const TEMPLATE_FEATURE_KEYS: readonly (keyof TemplateStyleSelection)[] = [
  "faceStyle",
  "hairFrontStyle",
  "hairSideStyle",
  "hairBackStyle",
  "eyeStyle",
  "mouthStyle",
  "clothesStyle",
];

function countStyleOverrides(sel?: TemplateStyleSelection): number {
  if (!sel) return 0;
  return TEMPLATE_FEATURE_KEYS.filter((k) => sel[k] && sel[k] !== "auto").length;
}

type AppTab = "prepare" | "expressions" | "adjust" | "play" | "advanced";

export default function App() {
  const [project, setProject] = useState<Project | null>(null),
    projectRef = useRef<Project | null>(null);
  const [selected, setSelected] = useState("face"),
    [tool, setTool] = useState<Tool>("landmark");
  const [tab, setTab] = useState<AppTab>("prepare");
  const [advancedPreview, setAdvancedPreview] = useState(false);
  const projectTextures = useProjectTextures(project);
  const [zoom, setZoom] = useState(1),
    [radius, setRadius] = useState(18),
    [tolerance, setTolerance] = useState(32);
  const [overlay, setOverlay] = useState(true),
    [fillOverlay, setFillOverlay] = useState(true),
    [compareDiff, setCompareDiff] = useState(false),
    [wire, setWire] = useState(false),
    [original, setOriginal] = useState(false),
    [solo, setSolo] = useState(false);
  const [detailMode, setDetailMode] = useState<"detail" | "parent">("detail");
  const [templateMode, setTemplateMode] = useState(true);
  const [hairSplit, setHairSplit] = useState(true);
  const [styleSelection, setStyleSelection] = useState<TemplateStyleSelection>({});
  const [styleMode, setStyleMode] = useState<"preserve" | "override">("preserve");
  const [subtractHairOnPrompt, setSubtractHairOnPrompt] = useState(true);
  const [parameters, setParameters] = useState<Parameters>({ ...REST }),
    [paramGroupFilter, setParamGroupFilter] = useState<string>("all"),
    [playing, setPlaying] = useState(false);
  const [meshes, setMeshes] = useState<Map<string, Mesh>>(new Map()),
    [meshWarnings, setMeshWarnings] = useState<MeshQualityReport[]>([]),
    [meshing, setMeshing] = useState(false);
  const [modelPath, setModelPath] = useState<string | null>(null),
    [modelStatus, setModelStatus] = useState<ModelStatusEntry[]>([]),
    [modelBusyRole, setModelBusyRole] = useState<string | null>(null),
    [csv, setCsv] = useState<CsvTemplate | null>(null);
  const [busy, setBusy] = useState(""),
    [message, setMessage] = useState(""),
    [error, setError] = useState("");
  const [promptResult, setPromptResult] = useState<RefinePromptResult | null>(
    null,
  );
  const [promptCandidateIdx, setPromptCandidateIdx] = useState(0);
  const [pendingReanalysis, setPendingReanalysis] = useState<{
    response: FaceAnalysisResponse;
    manualKeys: LandmarkId[];
  } | null>(null);
  const analysisJobSeq = useRef(0);
  const [dragging, setDragging] = useState(false);
  const dragDepth = useRef(0);
  useEffect(() => {
    if (desktop) {
      void getModelStatus().then(setModelStatus).catch(() => {});
    }
  }, []);
  useEffect(() => {
    const clearDrag = () => {
      dragDepth.current = 0;
      setDragging(false);
    };
    window.addEventListener("dragend", clearDrag);
    window.addEventListener("blur", clearDrag);
    return () => {
      window.removeEventListener("dragend", clearDrag);
      window.removeEventListener("blur", clearDrag);
    };
  }, []);
  const [modal, setModal] = useState<
    "export" | "regenerate" | "reanalysis" | "models" | "help" | "spriteGuide" | "updates" | null
  >(null),
    [dirty, setDirty] = useState(false);
  const [appVersion, setAppVersion] = useState("0.2.0-2");
  const [updateInfo, setUpdateInfo] = useState<UpdateCheckResult | null>(null);
  const [updateCustomUrl, setUpdateCustomUrl] = useState("");
  const [checkingUpdate, setCheckingUpdate] = useState(false);
  const [installingUpdate, setInstallingUpdate] = useState(false);
  const [updateDownloaded, setUpdateDownloaded] = useState(false);
  const [updateProgressPct, setUpdateProgressPct] = useState<number | null>(null);
  useEffect(() => {
    if (!desktop) return;
    void getAppVersion().then(setAppVersion);
    const timer = setTimeout(() => {
      void checkAppUpdate()
        .then((res) => setUpdateInfo(res))
        .catch(() => {});
    }, 1500);
    return () => clearTimeout(timer);
  }, []);
  const runUpdateCheck = async (customUrl?: string) => {
    setCheckingUpdate(true);
    try {
      const res = await checkAppUpdate(customUrl ?? updateCustomUrl);
      setUpdateInfo(res);
    } catch (e) {
      setUpdateInfo({
        status: "unavailable",
        currentVersion: appVersion,
        signed: false,
        endpointUsed: customUrl ?? updateCustomUrl,
        releaseUrl: "https://github.com/jinhoOps/flat2rig/releases",
        message: String(e),
      });
    } finally {
      setCheckingUpdate(false);
    }
  };
  const runUpdateInstall = async () => {
    setInstallingUpdate(true);
    setUpdateProgressPct(0);
    try {
      await installAppUpdate((ev) => {
        if (ev.event === "Started") {
          setUpdateProgressPct(0);
        } else if (ev.event === "Progress") {
          const total = ev.data.contentLength ?? 0;
          setUpdateProgressPct(
            total > 0 ? Math.min(99, Math.round((ev.data.downloaded / total) * 100)) : 50,
          );
        } else if (ev.event === "Finished") {
          setUpdateProgressPct(100);
        }
      });
      setUpdateDownloaded(true);
      notify("업데이트 설치 준비가 완료되었습니다. 앱을 재시작하면 적용됩니다.");
    } catch (e) {
      setError(String(e));
    } finally {
      setInstallingUpdate(false);
    }
  };
  const [guideZoom, setGuideZoom] = useState<"fit" | 100 | 150 | 200>("fit");
  const [spriteGuideUrl, setSpriteGuideUrl] = useState<string>("");
  useEffect(() => {
    setSpriteGuideUrl(renderSpriteGuideCanvas().toDataURL("image/png"));
  }, []);
  const [historyCount, setHistoryCount] = useState([0, 0]);
  const history = useRef<Project[]>([]),
    future = useRef<Project[]>([]),
    controller = useRef<AbortController | null>(null);
  const input = useRef<HTMLInputElement>(null),
    projectInput = useRef<HTMLInputElement>(null),
    partInput = useRef<HTMLInputElement>(null),
    csvInput = useRef<HTMLInputElement>(null);
  const dragSnapshot = useRef<Project | null>(null);
  const gesture = useRef<{
    before: Project;
    part?: string;
    landmark?: LandmarkId;
    start: Point;
    last: Point;
  } | null>(null);
  const update = useCallback((p: Project | null) => {
    projectRef.current = p;
    setProject(p);
  }, []);
  const commit = useCallback(
    (p: Project, before = projectRef.current) => {
      dragSnapshot.current = null;
      if (before) history.current.push(before);
      if (history.current.length > 24) history.current.shift();
      future.current = [];
      setHistoryCount([history.current.length, 0]);
      setDirty(true);
      update({ ...p, revision: (projectRef.current?.revision ?? 0) + 1 });
    },
    [update],
  );
  const beginDrag = useCallback(() => {
    if (!dragSnapshot.current && projectRef.current) {
      dragSnapshot.current = projectRef.current;
    }
  }, []);
  const previewDrag = useCallback(
    (next: Project) => {
      if (!dragSnapshot.current && projectRef.current) {
        dragSnapshot.current = projectRef.current;
      }
      update(next);
    },
    [update],
  );
  const commitDrag = useCallback(() => {
    const snap = dragSnapshot.current;
    const cur = projectRef.current;
    dragSnapshot.current = null;
    if (snap && cur && snap !== cur) {
      commit(cur, snap);
    }
  }, [commit]);
  const notify = (text: string) => {
    setMessage(text);
    setError("");
  };
  const commitRig = useCallback(
    (next: Project, note?: string) => {
      commit(next);
      if (note) notify(note);
    },
    [commit],
  );
  const regenerateShapeAssets = () => {
    const cur = projectRef.current;
    if (!cur || !cur.parts.length) return;
    void task("눈 · 입 대체 모양 파츠 생성 중", async (signal) => {
      const img = projectTextures.sourceImage ?? (await loadImage(cur.source));
      signal.throwIfAborted();
      const data = pixels(img, cur.width, cur.height);
      const next = synthesizeShapeAssetsForProject(cur, data, true);
      commit(next);
      notify("눈 · 입 대체 모양 파츠를 다시 생성했습니다.");
    });
  };
  const exportCurrentFramePng = () => {
    const p = projectRef.current;
    if (!p || !p.parts.length) return;
    setModal(null);
    void task("투명 배경 PNG 내보내는 중", async (signal) => {
      const meshesMap = meshes.size > 0 ? meshes : buildAllPartMeshes(p);
      let partTexMap = projectTextures.partTextures;
      if (partTexMap.size === 0) {
        const img = projectTextures.sourceImage ?? (await loadImage(p.source));
        signal.throwIfAborted();
        partTexMap = buildAllPartTextures(p, pixels(img, p.width, p.height));
      }
      const subTexMap =
        projectTextures.substituteTextures.size > 0 || !p.shapeAssets
          ? projectTextures.substituteTextures
          : await buildProjectSubstituteTextures(p);
      signal.throwIfAborted();
      const snap = renderDeterministicSnapshot(
        {
          project: p,
          userParameters: parameters,
        },
        partTexMap,
        meshesMap,
        subTexMap,
      );
      const blob = await new Promise<Blob | null>((resolve) =>
        snap.canvas.toBlob(resolve, "image/png"),
      );
      if (!blob) throw new Error("PNG 생성에 실패했습니다.");
      const bytes = new Uint8Array(await blob.arrayBuffer());
      const expTag = safeName(p.activeExpressionId || "neutral");
      const saved = await saveFile(
        `${safeName(p.name)}_${expTag}.png`,
        bytes,
        "image/png",
      );
      if (saved) notify("투명 배경 현재 표정 PNG를 저장했습니다.");
    });
  };
  const task = async (
    label: string,
    run: (signal: AbortSignal) => Promise<void>,
  ) => {
    if (controller.current) return;
    const abort = new AbortController();
    controller.current = abort;
    setBusy(label);
    setError("");
    setMessage("");
    try {
      await new Promise((r) => setTimeout(r, 30));
      await run(abort.signal);
    } catch (e) {
      if (abort.signal.aborted)
        setMessage("작업을 취소했습니다. 생성 결과를 적용하지 않았습니다.");
      else setError(e instanceof Error ? e.message : String(e));
    } finally {
      controller.current = null;
      setBusy("");
    }
  };
  const reset = (p: Project) => {
    dragSnapshot.current = null;
    history.current = [];
    future.current = [];
    setHistoryCount([0, 0]);
    setMeshes(new Map());
    setMeshWarnings([]);
    setPromptResult(null);
    setPromptCandidateIdx(0);
    setParameters({ ...(p.rigProfile?.neutralParameters ?? REST) });
    setPlaying(false);
    setZoom(1);
    setSolo(false);
    setOriginal(false);
    setCompareDiff(false);
    setDetailMode(p.detailMode ?? "detail");
    const loadedSel = p.styleSelection ?? {};
    setStyleSelection(loadedSel);
    setStyleMode(countStyleOverrides(loadedSel) > 0 ? "override" : "preserve");
    setTab("prepare");
    setAdvancedPreview(false);
    setTool(p.parts.length ? "select" : "face");
    setSelected(
      p.parts.find((t) => t.id === "face")?.id ?? p.parts[0]?.id ?? "face",
    );
    setDirty(false);
    update(p);
  };
  const applyAnalysisResponse = (
    base: Project,
    res: FaceAnalysisResponse,
    preserveManual: boolean,
  ): Project => {
    const prevProv = base.landmarkProvenance ?? {
      eyeL: "estimated",
      eyeR: "estimated",
      mouth: "estimated",
      chin: "estimated",
      crown: "estimated",
    };
    const nextLandmarks = { ...base.landmarks };
    const nextProvenance: LandmarkProvenance = { ...prevProv };
    if (res.rigLandmarks) {
      for (const id of ["eyeL", "eyeR", "mouth", "chin", "crown"] as LandmarkId[]) {
        if (preserveManual && prevProv[id] === "manual") continue;
        nextLandmarks[id] = res.rigLandmarks[id];
        nextProvenance[id] = res.rigProvenance?.[id] ?? "model";
      }
    }
    return {
      ...base,
      imageHash: res.imageHash || base.imageHash,
      faceCandidates: res.candidates.length
        ? res.candidates
        : base.faceCandidates,
      selectedFaceId: res.selectedFaceId ?? base.selectedFaceId,
      landmarks28: res.points28.length ? res.points28 : base.landmarks28,
      landmarks: nextLandmarks,
      landmarkProvenance: nextProvenance,
      modelHashes: { ...base.modelHashes, ...res.modelHashes },
      postprocessVersions: {
        ...base.postprocessVersions,
        ...res.postprocessVersions,
      },
      analysisStatus: res.status,
      analysisNote: res.message,
    };
  };
  const runFaceAnalysis = (
    faceIndex: number | null = null,
    bypassGuard = false,
  ) => {
    const snapshot = projectRef.current;
    if (!snapshot) return;
    const jobId = ++analysisJobSeq.current;
    void task("얼굴 · 28점 랜드마크 자동 검출 중", async (signal) => {
      const data = pixels(
        await loadImage(snapshot.source),
        snapshot.width,
        snapshot.height,
      );
      const res = await analyzeFace(
        data,
        snapshot.width,
        snapshot.height,
        faceIndex,
        jobId,
        signal,
      );
      signal.throwIfAborted();
      const current = projectRef.current;
      if (
        !current ||
        jobId !== analysisJobSeq.current ||
        current.source !== snapshot.source
      )
        return;
      if (res.status !== "detected" || !res.rigLandmarks) {
        update({
          ...current,
          imageHash: res.imageHash || current.imageHash,
          faceCandidates: res.candidates,
          modelHashes: { ...current.modelHashes, ...res.modelHashes },
          postprocessVersions: {
            ...current.postprocessVersions,
            ...res.postprocessVersions,
          },
          analysisStatus: res.status,
          analysisNote: res.message,
        });
        notify(res.message);
        return;
      }
      const manualKeys = (
        ["eyeL", "eyeR", "mouth", "chin", "crown"] as LandmarkId[]
      ).filter((k) => current.landmarkProvenance?.[k] === "manual");
      if (!bypassGuard && manualKeys.length > 0) {
        setPendingReanalysis({ response: res, manualKeys });
        setModal("reanalysis");
        return;
      }
      commit(applyAnalysisResponse(current, res, false), current);
      notify(res.message);
    });
  };
  const selectFaceCandidate = (candidateIndex: number) => {
    const current = projectRef.current;
    const cand = current?.faceCandidates?.[candidateIndex];
    if (!current || !cand) return;
    const jobId = ++analysisJobSeq.current;
    void task(`얼굴 후보 #${candidateIndex + 1} 랜드마크 추출 중`, async (signal) => {
      const data = pixels(
        await loadImage(current.source),
        current.width,
        current.height,
      );
      const res = await detectLandmarksForFace(
        data,
        current.width,
        current.height,
        cand.bbox,
        cand.cropBbox,
        candidateIndex,
        jobId,
        signal,
      );
      signal.throwIfAborted();
      const latest = projectRef.current;
      if (!latest || jobId !== analysisJobSeq.current || latest.source !== current.source)
        return;
      const manualKeys = (
        ["eyeL", "eyeR", "mouth", "chin", "crown"] as LandmarkId[]
      ).filter((k) => latest.landmarkProvenance?.[k] === "manual");
      if (manualKeys.length > 0) {
        setPendingReanalysis({ response: res, manualKeys });
        setModal("reanalysis");
        return;
      }
      commit(applyAnalysisResponse(latest, res, false), latest);
      notify(res.message);
    });
  };
  const openImage = (file?: File) => {
    if (!file) return;
    if (
      dirty &&
      !confirm("저장하지 않은 변경 사항이 있습니다. 새 이미지를 여시겠습니까?")
    )
      return;
    const jobId = ++analysisJobSeq.current;
    void task("이미지 준비 · 얼굴 검출 중", async (signal) => {
      const p = await readImage(file);
      signal.throwIfAborted();
      reset(p);
      const data = pixels(await loadImage(p.source), p.width, p.height);
      const res = await analyzeFace(data, p.width, p.height, null, jobId, signal);
      signal.throwIfAborted();
      const current = projectRef.current;
      if (!current || jobId !== analysisJobSeq.current || current.source !== p.source)
        return;
      const updated = applyAnalysisResponse(current, res, false);
      update(updated);
      notify(res.message);
    });
  };
  const generate = (
    overrideMode?: "detail" | "parent",
    overrideStyles?: TemplateStyleSelection,
    preserveApproved = false,
  ) => {
    const p = projectRef.current;
    if (!p) return;
    const mode = overrideMode ?? detailMode;
    const rawStyles = overrideStyles ?? styleSelection;
    const effectiveMode =
      overrideStyles !== undefined
        ? countStyleOverrides(overrideStyles) > 0
          ? "override"
          : "preserve"
        : styleMode;
    const styles: TemplateStyleSelection =
      effectiveMode === "preserve"
        ? { proportionMode: rawStyles.proportionMode ?? "auto" }
        : rawStyles;
    setModal(null);
    setPromptResult(null);
    const jobId = ++analysisJobSeq.current;
    void task("파츠 분리 초안 생성 중", async (signal) => {
      const data = pixels(await loadImage(p.source), p.width, p.height);
      const faceCand = p.faceCandidates?.[p.selectedFaceId ?? 0];
      const result = await proposeParts(
        data,
        p.width,
        p.height,
        p.landmarks,
        tolerance,
        {
          points28: p.landmarks28,
          faceBbox: faceCand?.bbox ?? null,
          modelPath,
          detailMode: mode,
          hairSplit,
          templateMode: templateMode && mode === "detail",
          styleOverrides: styles,
          jobId,
        },
        signal,
      );
      signal.throwIfAborted();
      const { parts: mergedParts, preservedCount } = preserveApproved
        ? mergePreservedParts(p.parts, result.parts)
        : { parts: result.parts, preservedCount: 0 };
      const nextProportion =
        result.styleMatch?.resolved.proportionMode === "sd"
          ? "sd"
          : result.styleMatch?.resolved.proportionMode === "ld"
            ? "ld"
            : p.rigProfile.proportionMode;
      const nextProject = synthesizeShapeAssetsForProject(
        {
          ...p,
          parts: mergedParts,
          provider: result.provider,
          detailMode: mode,
          styleMatch: result.styleMatch,
          styleSelection: styles,
          rigProfile: {
            ...p.rigProfile,
            proportionMode: nextProportion,
          },
          modelHash:
            result.modelHashes.part_decoder ??
            result.modelHashes.background ??
            p.modelHash,
          modelHashes: { ...p.modelHashes, ...result.modelHashes },
          postprocessVersions: {
            ...p.postprocessVersions,
            parts: result.postprocessVersion,
          },
        },
        data,
        !preserveApproved,
      );
      commit(nextProject);
      if (!mergedParts.some((t) => t.id === selected)) {
        setSelected(mergedParts[0]?.id ?? "face");
      }
      setTool("select");
      setTab("prepare");
    setAdvancedPreview(false);
      const overrideCount = countStyleOverrides(styles);
      const propLabel =
        result.styleMatch?.resolved.proportionMode === "sd" ? "SD형" : "일반 LD";
      notify(
        `${mergedParts.length}개 파츠가 ${result.styleMatch ? (overrideCount > 0 ? `명시적 스타일 변경(${overrideCount}개 부위 · ${propLabel})` : `원화 유지 + 가림 보완(${propLabel})`) : mode === "parent" ? "눈·입 통합 모드" : "세부 파츠 모드"}로 준비됐습니다.${preservedCount ? ` (수정·확인 ${preservedCount}개 보존)` : ""}`,
      );
    });
  };
  const undo = useCallback(() => {
    if (controller.current || !history.current.length || !projectRef.current)
      return;
    dragSnapshot.current = null;
    future.current.push(projectRef.current);
    const p = history.current.pop()!;
    update({ ...p, revision: projectRef.current.revision + 1 });
    setHistoryCount([history.current.length, future.current.length]);
    setDirty(true);
  }, [update]);
  const redo = useCallback(() => {
    if (controller.current || !future.current.length || !projectRef.current)
      return;
    dragSnapshot.current = null;
    history.current.push(projectRef.current);
    const p = future.current.pop()!;
    update({ ...p, revision: projectRef.current.revision + 1 });
    setHistoryCount([history.current.length, future.current.length]);
    setDirty(true);
  }, [update]);
  const save = () => {
    const p = projectRef.current;
    if (!p) return;
    void task("프로젝트 저장 중", async () => {
      const saved = await saveFile(
        `${safeName(p.name)}.flat2rig`,
        await saveProject(p),
      );
      if (saved) {
        setDirty(false);
        notify("프로젝트를 저장했습니다.");
      }
    });
  };
  useEffect(() => {
    const keyboard = (e: KeyboardEvent) => {
      const targetEl = e.target instanceof Element ? e.target : null;
      if (targetEl?.matches("input,textarea,select") || controller.current)
        return;
      if (e.ctrlKey || e.metaKey) {
        const key = e.key.toLowerCase();
        if (key === "z") {
          e.preventDefault();
          e.shiftKey ? redo() : undo();
        }
        if (key === "y") {
          e.preventDefault();
          redo();
        }
        if (key === "s") {
          e.preventDefault();
          document
            .querySelector<HTMLButtonElement>('[data-action="save"]')
            ?.click();
        }
        return;
      }
      if (
        (tab === "expressions" || tab === "adjust" || tab === "play") &&
        projectRef.current?.parts.length &&
        !modal
      ) {
        if (e.code === "Space" && !targetEl?.matches("button,a")) {
          e.preventDefault();
          setPlaying((v) => !v);
          return;
        }
        if (/^[1-9]$/.test(e.key)) {
          const cur = projectRef.current;
          const idx = Number(e.key) - 1;
          const exp =
            cur.expressions.find((item) => item.shortcutKey === e.key) ??
            cur.expressions[idx];
          if (exp) {
            e.preventDefault();
            commitRig(
              selectExpressionPreset(cur, exp.id),
              `표정 '${exp.name}' 전환`,
            );
            if (e.key === "1" || exp.id === "neutral") {
              setParameters({ ...(cur.rigProfile?.neutralParameters ?? REST) });
            }
          }
          return;
        }
      }
      if (tab === "prepare" || (tab === "advanced" && !advancedPreview)) {
        const keys: Record<string, Tool> = {
          v: "select",
          b: "add",
          e: "erase",
          p: "prompt",
          l: "landmark",
          f: "face",
        };
        if (keys[e.key]) setTool(keys[e.key]);
      }
      if (e.key === "Escape") {
        setModal(null);
        setPlaying(false);
      }
    };
    const unload = (e: BeforeUnloadEvent) => {
      if (dirty) {
        e.preventDefault();
        e.returnValue = "";
      }
    };
    window.addEventListener("keydown", keyboard);
    window.addEventListener("beforeunload", unload);
    return () => {
      window.removeEventListener("keydown", keyboard);
      window.removeEventListener("beforeunload", unload);
    };
  }, [undo, redo, dirty, tab, advancedPreview, modal, commitRig]);
  useEffect(() => {
    if (!project || !project.parts.length) return;
    let active = true;
    setMeshing(true);
    const timer = setTimeout(() => {
      void buildRig(
        project.parts,
        project.width,
        project.height,
        project.landmarks,
      )
        .then((rig) => {
          if (active) {
            setMeshes(new Map(rig.map((r) => [r.id, r.mesh])));
            setMeshWarnings(
              rig
                .map((r) => r.quality)
                .filter((q): q is MeshQualityReport => !!q && q.warnings.length > 0),
            );
          }
        })
        .catch((e) => {
          if (active) setError(String(e));
        })
        .finally(() => {
          if (active) setMeshing(false);
        });
    }, 180);
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [project?.parts, project?.landmarks, project?.width, project?.height, project?.source]);
  const patchPart = (patch: Partial<Part>) => {
    const p = projectRef.current;
    if (!p) return;
    commit({
      ...p,
      parts: p.parts.map((t) => (t.id === selected ? { ...t, ...patch } : t)),
    });
  };
  const toggleVisible = (id: string) => {
    const p = projectRef.current!;
    commit({
      ...p,
      parts: p.parts.map((t) =>
        t.id === id ? { ...t, visible: !t.visible } : t,
      ),
    });
  };
  const shiftPart = (direction: number) => {
    if (!project) return;
    const index = project.parts.findIndex((p) => p.id === selected),
      target = index + direction;
    if (index < 0 || target < 0 || target >= project.parts.length) return;
    const parts = [...project.parts];
    [parts[index], parts[target]] = [parts[target], parts[index]];
    const normalized = parts.map((item, idx) => ({
      ...item,
      compositeOrder: idx,
    }));
    commit({ ...project, parts: normalized });
  };
  const ensureProvenance = (p: Project): LandmarkProvenance =>
    p.landmarkProvenance ?? {
      eyeL: "estimated",
      eyeR: "estimated",
      mouth: "estimated",
      chin: "estimated",
      crown: "estimated",
    };
  const addDynamicLayer = (
    kind: HairRole | "hat" | "clothes_outer" | "acc_glasses" | "acc_hairpin",
  ) => {
    const p = projectRef.current;
    if (!p) return;
    const added =
      kind === "hair_front" ||
      kind === "hair_side_L" ||
      kind === "hair_side_R" ||
      kind === "hair_back"
        ? newHairPart(kind, p.parts, p.width, p.height)
        : newDynamicPart(kind, p.parts, p.width, p.height);
    const nextParts = insertHairPartInOrder(p.parts, added);
    commit({ ...p, parts: nextParts });
    setSelected(added.id);
    setTool(desktop ? "prompt" : "add");
    notify(
      `${added.name} 레이어를 추가했습니다. ${desktop ? "클릭 보정(P)으로 해당 영역을 클릭하세요." : "브러시로 영역을 칠하세요."}`,
    );
  };
  const applyPromptMaskToParts = (
    parts: Part[],
    targetId: string,
    mask: Uint8Array,
    iouScore: number,
    diagnostics: string[],
    nextPrompts?: PromptPoint[],
  ): Part[] => {
    const targetPart = parts.find((t) => t.id === targetId);
    const shouldSubtractHair =
      subtractHairOnPrompt && !!targetPart && isHairPart(targetPart);
    return parts.map((t) => {
      if (t.id === targetId) {
        let nextFill = t.fillMask;
        if (t.fillMask) {
          nextFill = t.fillMask.slice();
          for (let i = 0; i < nextFill.length; i++) {
            if (mask[i] > 0) nextFill[i] = 0;
          }
        }
        return {
          ...t,
          mask,
          ...(nextFill ? { fillMask: nextFill } : {}),
          provenance: "prompted",
          status: "draft",
          ...(nextPrompts ? { prompts: nextPrompts } : {}),
          confidence: iouScore,
          diagnostics,
        };
      }
      if (shouldSubtractHair && isHairPart(t) && t.status !== "approved") {
        let changed = false;
        const trimmed = t.mask.slice();
        for (let i = 0; i < trimmed.length; i++) {
          if (mask[i] > 128 && trimmed[i] > 0) {
            trimmed[i] = 0;
            changed = true;
          }
        }
        return changed ? { ...t, mask: trimmed } : t;
      }
      return t;
    });
  };
  const runPromptRefine = (
    p: Project,
    part: Part,
    nextPrompts: PromptPoint[],
  ) => {
    const jobId = ++analysisJobSeq.current;
    void task("클릭 프롬프트로 마스크 보정 중", async (signal) => {
      const data = pixels(await loadImage(p.source), p.width, p.height);
      const res = await refinePartPrompt(
        data,
        p.width,
        p.height,
        p.landmarks,
        part.id,
        nextPrompts,
        tolerance,
        jobId,
        signal,
      );
      signal.throwIfAborted();
      const latest = projectRef.current;
      if (!latest || jobId !== analysisJobSeq.current || latest.source !== p.source)
        return;
      const chosen = res.candidates[res.bestIndex] ?? res.candidates[0];
      if (!chosen) return;
      setPromptResult(res);
      setPromptCandidateIdx(res.bestIndex);
      commit(
        {
          ...latest,
          parts: applyPromptMaskToParts(
            latest.parts,
            part.id,
            chosen.mask,
            chosen.iouScore,
            chosen.diagnostics,
            nextPrompts,
          ),
        },
        p,
      );
      notify(
        `${part.name} 클릭 보정 완료 (IoU ${(chosen.iouScore * 100).toFixed(0)}%, ${
          res.usedFaceCrop ? "얼굴 고해상도 크롭" : "전체 캔버스"
        })`,
      );
    });
  };
  const selectPromptCandidate = (idx: number) => {
    const p = projectRef.current;
    if (!p || !promptResult) return;
    const cand = promptResult.candidates[idx];
    if (!cand) return;
    setPromptCandidateIdx(idx);
    commit({
      ...p,
      parts: applyPromptMaskToParts(
        p.parts,
        promptResult.partId,
        cand.mask,
        cand.iouScore,
        cand.diagnostics,
      ),
    });
  };
  const clearPartPrompts = () => {
    setPromptResult(null);
    setPromptCandidateIdx(0);
    patchPart({ prompts: [] });
  };
  const start = (pt: Point, exclude = false) => {
    const p = projectRef.current;
    if (!p || busy) return;
    if (tool === "landmark") {
      const nearest = (Object.keys(p.landmarks) as LandmarkId[]).sort(
        (a, b) =>
          Math.hypot(p.landmarks[a].x - pt.x, p.landmarks[a].y - pt.y) -
          Math.hypot(p.landmarks[b].x - pt.x, p.landmarks[b].y - pt.y),
      )[0];
      gesture.current = { before: p, landmark: nearest, start: pt, last: pt };
      return;
    }
    if (tool === "face") {
      gesture.current = { before: p, start: pt, last: pt };
      return;
    }
    if (tool === "select") {
      const part = [...p.parts]
        .reverse()
        .find(
          (t) =>
            t.visible &&
            t.status !== "missing" &&
            t.mask[Math.floor(pt.y) * p.width + Math.floor(pt.x)] > 8,
        );
      if (part) setSelected(part.id);
      return;
    }
    const part = p.parts.find((t) => t.id === selected);
    if (!part) return;
    if (tool === "prompt") {
      const nextPrompts: PromptPoint[] = [
        ...(part.prompts ?? []),
        { x: pt.x, y: pt.y, label: exclude ? 0 : 1 },
      ];
      runPromptRefine(p, part, nextPrompts);
      return;
    }
    gesture.current = { before: p, part: selected, start: pt, last: pt };
    const parts = p.parts.map((t) =>
      t.id === selected
        ? {
            ...t,
            status: "draft" as const,
            provenance: "manual" as const,
            mask: t.mask.slice(),
            fillMask: t.fillMask?.slice(),
          }
        : t,
    );
    const active = parts.find((t) => t.id === selected)!;
    paint(active.mask, p.width, p.height, pt, pt, radius, tool === "erase");
    if (tool === "add" && active.fillMask) {
      paint(active.fillMask, p.width, p.height, pt, pt, radius, true);
    }
    update({ ...p, parts, revision: p.revision + 1 });
  };
  const move = (pt: Point) => {
    const g = gesture.current,
      p = projectRef.current;
    if (!g || !p) return;
    if (g.landmark)
      update({
        ...p,
        landmarks: { ...p.landmarks, [g.landmark]: pt },
        landmarkProvenance: {
          ...ensureProvenance(p),
          [g.landmark]: "manual",
        },
        revision: p.revision + 1,
      });
    else if (tool === "face") {
      if (Math.abs(g.start.x - pt.x) > 8 && Math.abs(g.start.y - pt.y) > 8)
        update({
          ...p,
          landmarks: faceBoxLandmarks(g.start, pt),
          landmarkProvenance: {
            eyeL: "manual",
            eyeR: "manual",
            mouth: "manual",
            chin: "manual",
            crown: "manual",
          },
          revision: p.revision + 1,
        });
    } else if (g.part) {
      const part = p.parts.find((t) => t.id === g.part)!;
      paint(part.mask, p.width, p.height, g.last, pt, radius, tool === "erase");
      if (tool === "add" && part.fillMask) {
        paint(part.fillMask, p.width, p.height, g.last, pt, radius, true);
      }
      update({
        ...p,
        parts: p.parts.map((t) => (t.id === g.part ? { ...t } : t)),
        revision: p.revision + 1,
      });
    }
    g.last = pt;
  };
  const end = (pt: Point) => {
    const g = gesture.current;
    if (!g) return;
    move(pt);
    gesture.current = null;
    const p = projectRef.current;
    if (p && p !== g.before) {
      commit(p, g.before);
      if (g.landmark || tool === "face")
        notify(
          "랜드마크를 변경했습니다. 기존 마스크에 반영하려면 파츠 초안을 다시 생성하세요.",
        );
    }
  };
  const defaultFillRadius = (t: Part): number => {
    if (t.id === "face" || t.id === "neck" || t.id.startsWith("hair_back")) {
      return 36;
    }
    if (isHairPart(t)) {
      return 20;
    }
    return 12;
  };
  const fillPart = () => {
    const p = projectRef.current,
      part = p?.parts.find((t) => t.id === selected);
    if (!p || !part) return;
    const fillRadius = defaultFillRadius(part);
    void task("선택 파츠의 가림 영역 보완 중", async (signal) => {
      const data = pixels(
        await loadImage(part.texture ?? p.source),
        p.width,
        p.height,
      );
      const result = await fillOcclusionNative(
        data,
        part.mask,
        p.width,
        p.height,
        fillRadius,
        part.id,
      );
      signal.throwIfAborted();
      const texture = pixelsCanvas(result.data, p.width, p.height).toDataURL(
        "image/png",
      );
      commit({
        ...p,
        parts: p.parts.map((t) =>
          t.id === selected
            ? {
                ...t,
                fillMask: result.fillMask,
                fillKind: result.fillKind,
                texture,
                generated: true,
                status: "draft",
              }
            : t,
        ),
      });
      setFillOverlay(true);
      notify(
        result.warning ??
          `경계 2px 침식 내삽(${fillRadius}px, ${result.filledPixels}px)을 분리 보완 레이어에 저장했습니다.`,
      );
    });
  };
  const fillCriticalOcclusions = () => {
    const p = projectRef.current;
    if (!p) return;
    const targets = p.parts.filter(
      (t) =>
        t.status !== "missing" &&
        (t.id === "face" || t.id === "neck" || t.id.startsWith("hair_back")) &&
        t.mask.some((v) => v > 0),
    );
    if (!targets.length) return;
    void task("얼굴·목·뒷머리 가림 영역 일괄 보완 중", async (signal) => {
      const updates = new Map<
        string,
        { fillMask: Uint8Array; fillKind: FillKind; texture: string }
      >();
      for (const target of targets) {
        signal.throwIfAborted();
        const data = pixels(
          await loadImage(target.texture ?? p.source),
          p.width,
          p.height,
        );
        const result = await fillOcclusionNative(
          data,
          target.mask,
          p.width,
          p.height,
          defaultFillRadius(target),
          target.id,
        );
        const texture = pixelsCanvas(result.data, p.width, p.height).toDataURL(
          "image/png",
        );
        updates.set(target.id, {
          fillMask: result.fillMask,
          fillKind: result.fillKind,
          texture,
        });
      }
      signal.throwIfAborted();
      commit({
        ...p,
        parts: p.parts.map((t) => {
          const u = updates.get(t.id);
          return u
            ? {
                ...t,
                fillMask: u.fillMask,
                fillKind: u.fillKind,
                texture: u.texture,
                generated: true,
                status: t.status === "approved" ? "approved" : "draft",
              }
            : t;
        }),
      });
      setFillOverlay(true);
      notify(
        `${updates.size}개 주요 파츠(얼굴·목·뒷머리)의 가림 영역을 36px 침식 보완했습니다.`,
      );
    });
  };
  const clearFillMask = () => {
    patchPart({
      fillMask: undefined,
      fillKind: "none",
      texture: undefined,
      generated: false,
      status: "draft",
    });
    notify("분리된 가림 보완 영역을 제거했습니다.");
  };
  const replacePart = (file?: File) => {
    if (!file || !project) return;
    void task("파츠 이미지 교체 중", async (signal) => {
      const replacement = await readImage(file);
      const img = await loadImage(replacement.source);
      if (img.width !== project.width || img.height !== project.height)
        throw new Error(
          `파츠 PNG는 작업 캔버스와 같은 ${project.width} × ${project.height}px여야 합니다.`,
        );
      signal.throwIfAborted();
      const data = pixels(img, project.width, project.height),
        mask = Uint8Array.from(
          { length: project.width * project.height },
          (_, i) => data[i * 4 + 3],
        );
      patchPart({
        mask,
        fillMask: undefined,
        fillKind: "external",
        provenance: "manual",
        texture: replacement.source,
        generated: true,
        status: "draft",
      });
      notify("파츠 이미지를 교체했습니다.");
    });
  };
  const linkModelRole = async (role: string) => {
    try {
      const picked = await chooseOnnxModel(`${ROLE_LABELS[role] ?? role} 선택`);
      if (typeof picked !== "string") return;
      setModelBusyRole(role);
      const next = await setRegisteredModelPath(role, picked);
      setModelStatus(next);
      if (role === "background") setModelPath(picked);
      notify(`${ROLE_LABELS[role] ?? role} 모델을 연결했습니다.`);
    } catch (e) {
      setError(String(e));
    } finally {
      setModelBusyRole(null);
    }
  };
  const unlinkModelRole = async (role: string) => {
    try {
      setModelBusyRole(role);
      const next = await setRegisteredModelPath(role, null);
      setModelStatus(next);
      if (role === "background") setModelPath(null);
      notify(`${ROLE_LABELS[role] ?? role} 사용자 지정 연결을 해제했습니다.`);
    } catch (e) {
      setError(String(e));
    } finally {
      setModelBusyRole(null);
    }
  };
  const downloadModelRole = async (role: string) => {
    try {
      setModelBusyRole(role);
      const next = await downloadModel(role);
      setModelStatus(next);
      notify(`${ROLE_LABELS[role] ?? role} 모델 다운로드 및 SHA-256 검증을 완료했습니다.`);
    } catch (e) {
      setError(String(e));
    } finally {
      setModelBusyRole(null);
    }
  };
  const output = (kind: "bundle" | "psd") => {
    if (!project) return;
    setModal(null);
    void task(
      kind === "bundle" ? "Cubism 인계 패키지 생성 중" : "레이어드 PSD 생성 중",
      async (signal) => {
        const bytes =
          kind === "bundle"
            ? await exportBundle(project, csv)
            : await exportPsd(project);
        signal.throwIfAborted();
        const saved = await saveFile(
          `${safeName(project.name)}${kind === "bundle" ? "_cubism.zip" : "_import.psd"}`,
          bytes,
        );
        if (saved)
          notify(
            kind === "bundle"
              ? "PSD · 파츠 PNG · 리깅 초안 · 가이드를 내보냈습니다."
              : "레이어드 PSD를 내보냈습니다.",
          );
      },
    );
  };
  const part = project?.parts.find((p) => p.id === selected),
    approved =
      project?.parts.filter((p) => p.status === "approved").length ?? 0;
  const total = project?.parts.length ?? 0;
  const showLayers = tab === "prepare" || tab === "advanced";
  const isRigStage = tab === "expressions" || tab === "adjust" || tab === "play";
  const preview = isRigStage || (tab === "advanced" && advancedPreview);
  const changeTab = (t: AppTab) => {
    setTab(t);
    if (t === "expressions" || t === "adjust" || t === "play") {
      setOriginal(false);
      setOverlay(false);
      setFillOverlay(false);
      setCompareDiff(false);
      setSolo(false);
      if (t !== "play") setPlaying(false);
    } else if (t === "prepare") {
      setPlaying(false);
      setAdvancedPreview(false);
    } else if (t === "advanced" && !advancedPreview) {
      setPlaying(false);
    }
  };
  return (
    <div
      className={`app${dragging ? " is-dragging" : ""}`}
      onDragEnter={(e) => {
        if (!e.dataTransfer.types.includes("Files")) return;
        e.preventDefault();
        dragDepth.current++;
        if (!busy && !modal) setDragging(true);
      }}
      onDragLeave={(e) => {
        e.preventDefault();
        dragDepth.current = Math.max(0, dragDepth.current - 1);
        if (!dragDepth.current) setDragging(false);
      }}
      onDragOver={(e) => {
        if (!e.dataTransfer.types.includes("Files")) return;
        e.preventDefault();
        e.dataTransfer.dropEffect = busy || modal ? "none" : "copy";
      }}
      onDrop={(e) => {
        e.preventDefault();
        dragDepth.current = 0;
        setDragging(false);
        if (busy || modal) return;
        const files = Array.from(e.dataTransfer.files);
        if (files.length !== 1) {
          setError("PNG 또는 JPG 이미지 한 장을 놓으세요.");
          return;
        }
        const file = files[0];
        if (
          !/^image\/(png|jpeg)$/.test(file.type) &&
          !/\.(png|jpe?g)$/i.test(file.name)
        ) {
          setError("PNG 또는 JPG 파일만 불러올 수 있습니다.");
          return;
        }
        openImage(file);
      }}
    >
      {dragging && (
        <div className="drop-target" role="status" aria-live="polite">
          <div className="drop-card">
            <ImagePlus size={22} />
            <strong>일러스트 놓기</strong>
            <span>PNG · JPG 한 장</span>
          </div>
        </div>
      )}
      <input
        aria-label="이미지 파일 입력"
        ref={input}
        type="file"
        hidden
        accept="image/png,image/jpeg"
        onChange={(e) => {
          openImage(e.target.files?.[0]);
          e.target.value = "";
        }}
      />
      <input
        aria-label="프로젝트 파일 입력"
        ref={projectInput}
        type="file"
        hidden
        accept=".flat2rig"
        onChange={(e) => {
          const file = e.target.files?.[0];
          e.target.value = "";
          if (
            !file ||
            (dirty &&
              !confirm("변경 사항을 저장하지 않고 프로젝트를 여시겠습니까?"))
          )
            return;
          void task("프로젝트 불러오는 중", async (signal) => {
            const loaded = loadProject(new Uint8Array(await file.arrayBuffer()));
            const img = await loadImage(loaded.source);
            signal.throwIfAborted();
            const prepared =
              loaded.parts.length > 0
                ? synthesizeShapeAssetsForProject(
                    loaded,
                    pixels(img, loaded.width, loaded.height),
                    false,
                  )
                : loaded;
            reset(prepared);
            notify("프로젝트를 불러왔습니다.");
          });
        }}
      />
      <input
        aria-label="파츠 PNG 입력"
        ref={partInput}
        type="file"
        hidden
        accept="image/png"
        onChange={(e) => {
          replacePart(e.target.files?.[0]);
          e.target.value = "";
        }}
      />
      <input
        aria-label="Cubism CSV 샘플 입력"
        ref={csvInput}
        type="file"
        hidden
        accept=".csv"
        onChange={(e) => {
          const f = e.target.files?.[0];
          e.target.value = "";
          if (f)
            void task("CSV 형식 확인 중", async () => {
              const t = readCsvTemplate(await f.text());
              setCsv(t);
              notify(
                "CSV 샘플을 연결했습니다. 이 세션의 내보내기에 적용됩니다.",
              );
            });
        }}
      />
      <header className="topbar">
        <a className="brand" href="#" onClick={(e) => e.preventDefault()}>
          <BrandMark />
          <BrandWordmark />
        </a>
        <div className="header-divider" />
        <span className="project-title">
          {project?.name ?? "새 프로젝트"}
          {dirty && <i className="dirty-dot" />}
        </span>
        <span className="version">v{appVersion}</span>
        <div className="header-actions">
          <span className="local-status">
            <i />
            {desktop ? "로컬 처리" : "브라우저 미리보기"}
          </span>
          <button
            className={
              updateInfo?.status === "available"
                ? "compact-pill active"
                : undefined
            }
            disabled={!!busy}
            onClick={() => {
              if (!updateInfo && !checkingUpdate) void runUpdateCheck();
              setModal("updates");
            }}
            title="앱 버전 확인 및 자동 업데이트"
          >
            <RefreshCw size={14} className={checkingUpdate ? "spin" : ""} />
            {updateInfo?.status === "available" && updateInfo.latestVersion
              ? `업데이트 v${updateInfo.latestVersion}`
              : "업데이트"}
          </button>
          <button
            disabled={!!busy}
            onClick={() => {
              if (desktop) void getModelStatus().then(setModelStatus).catch(() => {});
              setModal("models");
            }}
            title="ONNX 모델 설치 · SHA-256 검증 설정"
          >
            <Settings size={15} />
            모델 {modelStatus.length ? `${modelStatus.filter((m) => m.hashMatches).length}/${modelStatus.length}` : "관리"}
          </button>
          <button
            className="icon-button"
            aria-label="사용 가이드"
            onClick={() => setModal("help")}
          >
            <CircleHelp size={17} />
          </button>
          <button
            data-action="save"
            disabled={!project || !!busy}
            onClick={save}
          >
            <Save size={15} />
            저장
          </button>
          <button
            className="primary"
            disabled={!total || !!busy}
            onClick={() => setModal("export")}
          >
            <Upload size={15} />
            내보내기
            <ChevronRight size={14} />
          </button>
        </div>
      </header>
      <div className={showLayers ? "workspace" : "workspace no-layers"}>
        {showLayers && (
        <aside className="layers-panel">
          <div className="panel-header">
            <span>
              <Layers size={15} />
              파츠 레이어
            </span>
            <span className="badge">{total.toString().padStart(2, "0")}</span>
          </div>
          <div className="layer-actions">
            <button disabled={!!busy} onClick={() => input.current?.click()}>
              <ImagePlus size={14} />
              이미지 열기
            </button>
            <button
              className="icon-button"
              aria-label="프로젝트 열기"
              disabled={!!busy}
              onClick={() => projectInput.current?.click()}
            >
              <FolderOpen size={16} />
            </button>
            <button
              className="icon-button"
              aria-label="파츠 추가"
              disabled={!project || !!busy}
              onClick={() => {
                if (project) {
                  const p = newPart(project.width, project.height);
                  commit({ ...project, parts: [...project.parts, p] });
                  setSelected(p.id);
                  setTool("add");
                }
              }}
            >
              <Plus size={16} />
            </button>
          </div>
          {total > 0 && (
            <div className="candidate-pills" style={{ padding: "6px 10px 2px", gap: 4 }}>
              {(
                [
                  ["hair_front", "+ 앞머리"],
                  ["hair_side_R", "+ 오른옆"],
                  ["hair_side_L", "+ 왼옆"],
                  ["hair_back", "+ 뒷머리"],
                  ["hat", "+ 모자"],
                  ["clothes_outer", "+ 의상"],
                  ["acc_glasses", "+ 안경"],
                  ["acc_hairpin", "+ 악세"],
                ] as const
              ).map(([kind, label]) => (
                <button
                  key={kind}
                  className="compact-pill"
                  disabled={!!busy}
                  onClick={() => addDynamicLayer(kind)}
                  title={`${label.replace("+ ", "")} 분리 레이어 추가`}
                >
                  {label}
                </button>
              ))}
            </div>
          )}
          <div className="layer-list">
            {!total ? (
              <div className="layer-empty">
                <Box size={25} />
                <span>파츠를 준비해 보세요</span>
                <p>
                  이미지를 열고 얼굴 위치를 맞추면
                  <br />
                  분리 초안을 만들 수 있습니다.
                </p>
              </div>
            ) : (
              (() => {
                const rev = [...project!.parts].reverse();
                return rev.map((p, i) => {
                  const groupName = p.group || "기타";
                  const prevGroup =
                    i > 0 ? rev[i - 1].group || "기타" : null;
                  const showHeader = groupName !== prevGroup;
                  let runCount = 0;
                  if (showHeader) {
                    for (
                      let k = i;
                      k < rev.length && (rev[k].group || "기타") === groupName;
                      k++
                    )
                      runCount++;
                  }
                  return (
                    <Fragment key={p.id}>
                      {showHeader && (
                        <div className="layer-group-header">
                          <span>{groupName}</span>
                          <span className="param-group-badge">{runCount}</span>
                        </div>
                      )}
                      <button
                        className={`layer ${selected === p.id ? "selected" : ""} ${p.status === "missing" ? "missing" : ""}`}
                        disabled={!!busy}
                        onClick={() => setSelected(p.id)}
                      >
                        <span
                          className="layer-visibility"
                          role="checkbox"
                          aria-checked={p.visible}
                          aria-label={`${p.name} 표시`}
                          tabIndex={0}
                          onKeyDown={(e) => {
                            if (e.key === " ") {
                              e.preventDefault();
                              e.stopPropagation();
                              toggleVisible(p.id);
                            }
                          }}
                          onClick={(e) => {
                            e.stopPropagation();
                            toggleVisible(p.id);
                          }}
                        >
                          {p.visible ? <Eye size={13} /> : <EyeOff size={13} />}
                        </span>
                        <span
                          className="layer-thumb"
                          style={{ "--part-color": p.color } as React.CSSProperties}
                        >
                          <Layers size={14} />
                        </span>
                        <span className="layer-label">
                          <strong>{p.name}</strong>
                          <small>
                            {p.id} · {partProvenanceLabel(p.provenance)}
                            {p.diagnostics?.length
                              ? ` · ⚠${diagnosticLabel(p.diagnostics[0])}`
                              : ""}
                          </small>
                        </span>
                        <span
                          className={`status-dot ${p.status}`}
                          title={
                            p.status === "approved"
                              ? "확인됨"
                              : p.status === "missing"
                                ? "미해당"
                                : "확인 필요"
                          }
                        >
                          {p.status === "approved" && <Check size={12} />}
                        </span>
                        <GripVertical className="grip" size={14} />
                        <span className="sr-only">{i + 1}</span>
                      </button>
                    </Fragment>
                  );
                });
              })()
            )}
          </div>
          <div className="layer-footer">
            <div>
              <span>파츠 확인</span>
              <strong>
                {approved}
                <small> / {total}</small>
              </strong>
            </div>
            <div className="progress-track">
              <div
                style={{ width: `${total ? (approved / total) * 100 : 0}%` }}
              />
            </div>
            <p>좌우는 캐릭터 기준입니다.</p>
          </div>
        </aside>
        )}
        <main className="editor">
          <div className="editor-tabs">
            <div role="tablist" aria-label="작업 단계">
              <button
                role="tab"
                aria-selected={tab === "prepare"}
                className={tab === "prepare" ? "active" : ""}
                onClick={() => changeTab("prepare")}
              >
                <ScanFace size={15} />
                1. 준비
              </button>
              <button
                role="tab"
                aria-selected={tab === "expressions"}
                disabled={!total || !!busy}
                className={tab === "expressions" ? "active" : ""}
                onClick={() => changeTab("expressions")}
              >
                <Smile size={15} />
                2. 표정
              </button>
              <button
                role="tab"
                aria-selected={tab === "adjust"}
                disabled={!total || !!busy}
                className={tab === "adjust" ? "active" : ""}
                onClick={() => changeTab("adjust")}
              >
                <SlidersHorizontal size={15} />
                3. 조정
              </button>
              <button
                role="tab"
                aria-selected={tab === "play"}
                disabled={!total || !!busy}
                className={tab === "play" ? "active" : ""}
                onClick={() => changeTab("play")}
              >
                <Play size={15} />
                4. 재생
              </button>
              <button
                role="tab"
                aria-selected={tab === "advanced"}
                disabled={!!busy}
                className={`tab-advanced${tab === "advanced" ? " active" : ""}`}
                onClick={() => changeTab("advanced")}
              >
                <Layers size={14} />
                고급 (파츠·메쉬·PSD)
              </button>
            </div>
            <span className="stage-tag">
              {tab === "prepare"
                ? "01 PREPARE"
                : tab === "expressions"
                  ? "02 EXPRESSIONS"
                  : tab === "adjust"
                    ? "03 ADJUST"
                    : tab === "play"
                      ? playing
                        ? "04 LIVE PLAYING"
                        : "04 STAGE"
                      : "ADVANCED"}
            </span>
          </div>
          {!isRigStage && (
          <div className="toolbar">
            <div className="tool-group">
              {(
                [
                  { key: "select", label: "선택 (V)", icon: MousePointer2 },
                  { key: "face", label: "얼굴 영역 지정 (F)", icon: ScanFace },
                  { key: "landmark", label: "랜드마크 이동 (L)", icon: Move },
                  { key: "prompt", label: "클릭 보정 (P)", icon: Crosshair },
                  { key: "add", label: "마스크 추가 (B)", icon: Plus },
                  { key: "erase", label: "마스크 지우기 (E)", icon: Eraser },
                ] as const
              ).map((t) => (
                <button
                  key={t.key}
                  className={`icon-button ${tool === t.key && !preview ? "active" : ""}`}
                  title={t.label}
                  aria-label={t.label}
                  disabled={
                    !project ||
                    preview ||
                    !!busy ||
                    (!total &&
                      (t.key === "add" || t.key === "erase" || t.key === "prompt"))
                  }
                  onClick={() => setTool(t.key)}
                >
                  <t.icon size={17} />
                </button>
              ))}
            </div>
            <span className="tool-divider" />
            <button
              className="icon-button"
              aria-label="실행 취소"
              title="Ctrl+Z"
              disabled={!historyCount[0] || !!busy}
              onClick={undo}
            >
              <Undo2 size={16} />
            </button>
            <button
              className="icon-button"
              aria-label="다시 실행"
              title="Ctrl+Shift+Z"
              disabled={!historyCount[1] || !!busy}
              onClick={redo}
            >
              <Redo2 size={16} />
            </button>
            <div className="toolbar-end">
              <label className="toggle-label">
                <input
                  type="checkbox"
                  checked={original}
                  disabled={!project || preview}
                  onChange={(e) => {
                    const checked = e.target.checked;
                    setOriginal(checked);
                    if (checked) setCompareDiff(false);
                  }}
                />
                원본
              </label>
              <label className="toggle-label">
                <input
                  type="checkbox"
                  checked={fillOverlay}
                  disabled={!total || preview}
                  onChange={(e) => setFillOverlay(e.target.checked)}
                />
                보완
              </label>
              <label className="toggle-label">
                <input
                  type="checkbox"
                  checked={compareDiff}
                  disabled={!total || preview}
                  onChange={(e) => {
                    const checked = e.target.checked;
                    setCompareDiff(checked);
                    if (checked) setOriginal(false);
                  }}
                />
                차이
              </label>
              <label className="toggle-label">
                <input
                  type="checkbox"
                  checked={wire}
                  disabled={!total}
                  onChange={(e) => setWire(e.target.checked)}
                />
                메쉬
              </label>
              <button
                className="icon-button"
                aria-label="화면에 맞춤"
                onClick={() => setZoom(1)}
              >
                <Maximize size={15} />
              </button>
            </div>
          </div>
          )}
          <div className={`stage ${busy ? "working" : ""}`}>
            {project ? (
              <>
                <div className="canvas-caption">
                  <span>
                    <i />
                    {original
                      ? "ORIGINAL"
                      : tab === "play"
                        ? playing
                          ? "LIVE PLAYBACK"
                          : "CLEAN STAGE"
                        : tab === "expressions"
                          ? `EXPRESSION · ${project.expressions.find((e) => e.id === project.activeExpressionId)?.name ?? "기본"}`
                          : preview
                            ? "MOTION PREVIEW"
                            : "WORKING CANVAS"}
                  </span>
                  <span>
                    {project.width} × {project.height}
                  </span>
                </div>
                <Viewport
                  project={project}
                  selected={selected}
                  tool={tool}
                  radius={radius}
                  zoom={zoom}
                  original={original}
                  overlay={overlay}
                  fillOverlay={fillOverlay}
                  compareDiff={compareDiff}
                  wire={wire}
                  solo={solo}
                  preview={preview}
                  playing={playing}
                  cleanStage={tab === "play"}
                  parameters={parameters}
                  meshes={meshes}
                  sharedTextures={projectTextures}
                  onStart={start}
                  onMove={move}
                  onEnd={end}
                />
                <div className="stage-bottom">
                  <span>
                    {tab === "play"
                      ? "Space: 재생/일시정지 · 1~9 키: 표정 빠른 전환"
                      : tab === "expressions"
                        ? "우측 표정 카드를 클릭하거나 1~9 키로 즉시 전환할 수 있습니다"
                        : tab === "adjust"
                          ? "왼쪽/오른쪽 눈·입 모양을 고르고 슬라이더로 각도를 다듬으세요"
                          : preview
                            ? "고급 메쉬/변형 확인 모드입니다"
                            : tool === "face"
                              ? "얼굴 피부 영역의 좌상단에서 우하단까지 드래그하세요"
                              : tool === "landmark"
                                ? "눈 · 입 · 턱 · 얼굴 상단의 점을 드래그하세요"
                                : tool === "prompt"
                                  ? "좌클릭: 포함(+) · 우클릭/Alt/Shift+클릭: 제외(-) 영역 지정"
                                  : tool === "add" || tool === "erase"
                                    ? "드래그하여 선택 파츠의 마스크를 수정하세요"
                                    : "파츠를 선택하여 경계를 확인하세요"}
                  </span>
                  <div>
                    {isRigStage && (
                      <>
                        <button
                          aria-label="실행 취소"
                          title="실행 취소 (Ctrl+Z)"
                          disabled={!historyCount[0] || !!busy}
                          onClick={undo}
                        >
                          <Undo2 size={14} />
                        </button>
                        <button
                          aria-label="다시 실행"
                          title="다시 실행 (Ctrl+Shift+Z)"
                          disabled={!historyCount[1] || !!busy}
                          onClick={redo}
                        >
                          <Redo2 size={14} />
                        </button>
                      </>
                    )}
                    <button
                      aria-label="축소"
                      onClick={() => setZoom((z) => clamp(z - 0.25, 0.25, 4))}
                    >
                      <ZoomOut size={14} />
                    </button>
                    <button onClick={() => setZoom(1)}>
                      {Math.round(zoom * 100)}%
                    </button>
                    <button
                      aria-label="확대"
                      onClick={() => setZoom((z) => clamp(z + 0.25, 0.25, 4))}
                    >
                      <ZoomIn size={14} />
                    </button>
                  </div>
                </div>
              </>
            ) : (
              <div
                className="welcome"
                role="region"
                aria-label="일러스트 드롭 영역"
              >
                <div className="welcome-drop-zone">
                  <BrandLoop />
                  <h1>일러스트 가져오기</h1>
                  <p>
                    PNG 또는 JPG 파일 한 장을 끌어 놓으세요.
                    <br />
                    정면 캐릭터 흉상 (일반 LD · SD 치비) · 최대 2,048px
                  </p>
                  <button
                    className="primary large"
                    disabled={!!busy}
                    onClick={() => input.current?.click()}
                  >
                    <ImagePlus size={18} />
                    파일 선택
                  </button>
                  <span className="drop-hint">
                    LD/SD 실루엣 매칭 · 모션용 21파츠 분리 · Cubism용 PSD
                  </span>
                  <button
                    className="text-button"
                    disabled={!!busy}
                    onClick={() => {
                      void demoFile().then(openImage);
                    }}
                  >
                    샘플 일러스트로 바로 체험
                    <ChevronRight size={13} />
                  </button>
                  <button
                    className="text-button guide-link-btn"
                    disabled={!!busy}
                    onClick={() => {
                      setGuideZoom("fit");
                      setModal("spriteGuide");
                    }}
                  >
                    <CircleHelp size={14} />
                    권장 일러스트 · 파츠 예시 시트 보기 (?)
                  </button>
                </div>
                <div className="welcome-features">
                  <span>
                    <Layers size={14} />
                    21개 모션 실루엣 파츠
                  </span>
                  <span>
                    <Box size={14} />
                    메쉬 · 22 파라미터 변형
                  </span>
                  <span>
                    <FileImage size={14} />
                    레이어드 PSD
                  </span>
                </div>
              </div>
            )}
            {busy && (
              <div className="busy-overlay">
                <div>
                  <Loader2 className="spin" size={26} />
                  <strong>{busy}</strong>
                  <p>이미지는 이 기기에서 처리됩니다.</p>
                  <button
                    onClick={() => {
                      controller.current?.abort();
                      void cancelAnalysis();
                      setBusy("취소 요청됨 · 처리 결과 폐기 대기");
                    }}
                  >
                    취소
                  </button>
                </div>
              </div>
            )}
          </div>
          <div className="editor-status">
            <span className={error ? "error-text" : ""}>
              {error ? <X size={13} /> : <CheckCircle2 size={13} />}
              <span role={error ? "alert" : "status"}>
                {error || message || "PNG 또는 JPG 이미지를 불러오세요."}
              </span>
            </span>
            <span>
              {meshing ? (
                <>
                  <Loader2 size={12} className="spin" />
                  메쉬 계산 중
                </>
              ) : project ? (
                `${total} PARTS · ${[...meshes.values()].reduce((n, m) => n + m.vertices.length, 0)} VERTICES`
              ) : (
                "READY"
              )}
            </span>
          </div>
        </main>
        <aside className="inspector">
          <div className="panel-header">
            <span>
              <SlidersHorizontal size={15} />
              {tab === "prepare"
                ? "준비 · 파츠 생성"
                : tab === "expressions"
                  ? "표정 선택 · 저장"
                  : tab === "adjust"
                    ? "부위 · 움직임 조정"
                    : tab === "play"
                      ? "재생 · PNG 저장"
                      : "고급 · 파츠·Cubism"}
            </span>
            <span className="tiny-label">
              {tab === "prepare"
                ? "STEP 1"
                : tab === "expressions"
                  ? "STEP 2"
                  : tab === "adjust"
                    ? "STEP 3"
                    : tab === "play"
                      ? "STEP 4"
                      : "ADVANCED"}
            </span>
          </div>
          <div className="inspector-scroll">
            {tab === "expressions" && project ? (
              <ExpressionPanel
                project={project}
                parameters={parameters}
                partTextures={projectTextures.partTextures}
                meshes={meshes}
                substituteTextures={projectTextures.substituteTextures}
                busy={!!busy}
                onCommit={commitRig}
                onBeginDrag={beginDrag}
                onPreviewDrag={previewDrag}
                onCommitDrag={commitDrag}
                onResetPoseToNeutral={() =>
                  setParameters({
                    ...(project.rigProfile?.neutralParameters ?? REST),
                  })
                }
                onOpenAdvancedParts={(partId) => {
                  if (partId) setSelected(partId);
                  setAdvancedPreview(false);
                  changeTab("advanced");
                }}
              />
            ) : tab === "adjust" && project ? (
              <AdjustPanel
                project={project}
                parameters={parameters}
                onChangeParameters={setParameters}
                partTextures={projectTextures.partTextures}
                meshes={meshes}
                substituteTextures={projectTextures.substituteTextures}
                busy={!!busy}
                onCommit={commitRig}
                onBeginDrag={beginDrag}
                onPreviewDrag={previewDrag}
                onCommitDrag={commitDrag}
                onRegenerateShapeAssets={regenerateShapeAssets}
                onOpenAdvancedParts={(partId) => {
                  if (partId) setSelected(partId);
                  setAdvancedPreview(false);
                  changeTab("advanced");
                }}
              />
            ) : tab === "play" && project ? (
              <PlaybackPanel
                project={project}
                playing={playing}
                busy={!!busy}
                onTogglePlay={() => setPlaying((v) => !v)}
                onResetToNeutral={() => {
                  setPlaying(false);
                  setParameters({
                    ...(project.rigProfile?.neutralParameters ?? REST),
                  });
                  commitRig(
                    selectExpressionPreset(
                      project,
                      project.expressions[0]?.id ?? "neutral",
                    ),
                    "기본 상태로 복귀했습니다.",
                  );
                }}
                onCommit={commitRig}
                onBeginDrag={beginDrag}
                onPreviewDrag={previewDrag}
                onCommitDrag={commitDrag}
                onExportPng={exportCurrentFramePng}
                onOpenAdvancedExport={() => setModal("export")}
              />
            ) : (
              <>
                {tab === "advanced" && (
                  <section>
                    <div className="section-title">
                      <span>00</span>
                      <h2>고급 작업 보기</h2>
                    </div>
                    <div className="two-buttons compact">
                      <button
                        className={!advancedPreview ? "active" : ""}
                        disabled={!!busy}
                        onClick={() => {
                          setAdvancedPreview(false);
                          setPlaying(false);
                        }}
                      >
                        파츠 · 마스크 보정
                      </button>
                      <button
                        className={advancedPreview ? "active" : ""}
                        disabled={!total || !!busy}
                        onClick={() => setAdvancedPreview(true)}
                      >
                        메쉬 · 파라미터 확인
                      </button>
                    </div>
                  </section>
                )}
                {tab === "prepare" && (
                <section>
                  <div className="section-title">
                    <span>01</span>
                    <h2>얼굴 · 파츠 초안</h2>
                  </div>
                  <p className="muted">
                    자동 검출 후 수동 드래그로 눈·입·crown을 보정하세요.
                  </p>
                  <button
                    className="full"
                    disabled={!project || !!busy}
                    onClick={() => runFaceAnalysis(null, false)}
                  >
                    <ScanFace size={15} />
                    얼굴 · 랜드마크 자동 검출
                  </button>
                  {project?.analysisNote && (
                    <div
                      className={`analysis-banner status-${project.analysisStatus ?? "idle"}`}
                    >
                      <span>{project.analysisNote}</span>
                    </div>
                  )}
                  {project?.faceCandidates && project.faceCandidates.length > 1 && (
                    <div className="candidate-list">
                      <span className="field-label">
                        검출된 얼굴 후보 ({project.faceCandidates.length})
                      </span>
                      <div className="candidate-pills">
                        {project.faceCandidates.map((cand) => (
                          <button
                            key={cand.id}
                            className={`compact-pill ${
                              (project.selectedFaceId ?? 0) === cand.id
                                ? "active"
                                : ""
                            }`}
                            disabled={!!busy}
                            onClick={() => selectFaceCandidate(cand.id)}
                          >
                            #{cand.id + 1} · {Math.round(cand.score * 100)}%
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                  {project?.landmarkProvenance && (
                    <div className="provenance-row">
                      {(["eyeR", "eyeL", "mouth", "chin", "crown"] as LandmarkId[]).map(
                        (k) => {
                          const src = project.landmarkProvenance![k];
                          return (
                            <span
                              key={k}
                              className={`prov-chip prov-${src}`}
                              title={`${k}: ${src}`}
                            >
                              {k}·{src === "manual" ? "수동" : src === "estimated" ? "추정" : "모델"}
                            </span>
                          );
                        },
                      )}
                    </div>
                  )}
                  <label className="field-label">
                    배경 제거
                    <span>
                      {modelPath
                        ? "ONNX"
                        : tolerance === 0
                          ? "알파 유지"
                          : "테두리 색상"}
                    </span>
                  </label>
                  <input
                    aria-label="배경 제거 허용값"
                    type="range"
                    min="0"
                    max="120"
                    value={tolerance}
                    disabled={!!busy}
                    onChange={(e) => setTolerance(+e.target.value)}
                  />
                  <div className="range-ends">
                    <span>유지</span>
                    <span>유사색 제거 {tolerance}</span>
                  </div>
                  <div className="field-label">
                    분할 단위
                    <span>{detailMode === "parent" ? "통합(눈·입)" : "세부 파츠"}</span>
                  </div>
                  <div className="two-buttons compact">
                    <button
                      className={detailMode === "detail" ? "active" : ""}
                      disabled={!!busy}
                      onClick={() => {
                        if (detailMode !== "detail") {
                          setDetailMode("detail");
                          if (total > 0) setModal("regenerate");
                        }
                      }}
                    >
                      세부 파츠
                    </button>
                    <button
                      className={detailMode === "parent" ? "active" : ""}
                      disabled={!!busy}
                      onClick={() => {
                        if (detailMode !== "parent") {
                          setDetailMode("parent");
                          if (total > 0) setModal("regenerate");
                        }
                      }}
                    >
                      눈·입 통합
                    </button>
                  </div>
                  <div className="check-options">
                    {detailMode === "detail" && (
                      <label>
                        <input
                          type="checkbox"
                          checked={templateMode}
                          disabled={!!busy}
                          onChange={(e) => {
                            setTemplateMode(e.target.checked);
                            if (total > 0) setModal("regenerate");
                          }}
                        />
                        실루엣 스타일 매칭 (LD/SD 모션 보완)
                      </label>
                    )}
                    <label>
                      <input
                        type="checkbox"
                        checked={hairSplit}
                        disabled={!!busy}
                        onChange={(e) => {
                          setHairSplit(e.target.checked);
                          if (total > 0) setModal("regenerate");
                        }}
                      />
                      다중 머리카락 분리 (앞·옆·뒷머리)
                    </label>
                  </div>
                  {detailMode === "detail" && templateMode && (
                    <div className="template-match-box">
                      <div className="template-match-header">
                        <span>분리 · 보완 방식</span>
                        {project?.styleMatch && (
                          <em>
                            {project.styleMatch.resolved.proportionMode === "sd"
                              ? "SD형"
                              : "일반 LD"}
                            {" · "}
                            {countStyleOverrides(project.styleSelection) > 0
                              ? `스타일 변경 (${countStyleOverrides(project.styleSelection)}개)`
                              : "원화 유지"}
                          </em>
                        )}
                      </div>
                      <div className="seg" style={{ marginBottom: 8 }}>
                        <button
                          className={styleMode === "preserve" ? "on" : ""}
                          disabled={!!busy}
                          onClick={() => setStyleMode("preserve")}
                        >
                          원화 유지 (기본)
                        </button>
                        <button
                          className={styleMode === "override" ? "on" : ""}
                          disabled={!!busy}
                          onClick={() => setStyleMode("override")}
                        >
                          스타일 변경
                        </button>
                      </div>
                      <div className="style-preset-grid">
                        {(styleMode === "preserve"
                          ? TEMPLATE_STYLE_CATALOG.slice(0, 1)
                          : TEMPLATE_STYLE_CATALOG
                        ).map((cat) => {
                          const curVal = styleSelection[cat.key] ?? "auto";
                          const matchedId = project?.styleMatch?.resolved?.[cat.key];
                          const matchedOpt = matchedId
                            ? cat.options.find((o) => o.id === matchedId)?.label
                            : undefined;
                          return (
                            <label key={cat.key} className="style-preset-field">
                              <span>
                                {cat.label}
                                {curVal === "auto" && matchedOpt && (
                                  <small> · {matchedOpt}</small>
                                )}
                              </span>
                              <select
                                aria-label={cat.label}
                                value={curVal}
                                disabled={!!busy}
                                onChange={(e) =>
                                  setStyleSelection((prev) => ({
                                    ...prev,
                                    [cat.key]: e.target.value,
                                  }))
                                }
                              >
                                {cat.options.map((opt) => (
                                  <option key={opt.id} value={opt.id}>
                                    {opt.label}
                                  </option>
                                ))}
                              </select>
                            </label>
                          );
                        })}
                      </div>
                      {styleMode === "preserve" && (
                        <small className="subtle-hint" style={{ display: "block", marginTop: 6 }}>
                          원화의 선과 색을 그대로 유지하고 가려진 모션 영역만 보완합니다.
                        </small>
                      )}
                      {total > 0 && (
                        <button
                          className="full"
                          disabled={!project || !!busy}
                          onClick={() =>
                            generate(
                              undefined,
                              styleMode === "preserve"
                                ? { proportionMode: styleSelection.proportionMode ?? "auto" }
                                : styleSelection,
                              true,
                            )
                          }
                        >
                          <RefreshCw size={14} />
                          {styleMode === "preserve"
                            ? "원화 유지 모드로 다시 적용 (수정·확인 파츠 유지)"
                            : "선택 스타일 적용 (수정·확인 파츠 유지)"}
                        </button>
                      )}
                    </div>
                  )}
                  <button
                    className="model-button"
                    disabled={!!busy}
                    onClick={() => {
                      if (desktop) void getModelStatus().then(setModelStatus).catch(() => {});
                      setModal("models");
                    }}
                  >
                    <Settings size={15} />
                    <span>
                      ONNX 모델 관리 ({modelStatus.filter((m) => m.hashMatches).length}/{modelStatus.length || 5})
                      <small>
                        {desktop
                          ? "MobileSAM · HRNet · YOLOv3 · U2NetP"
                          : "데스크톱 앱에서 ONNX 연결/다운로드"}
                      </small>
                    </span>
                    <ChevronRight size={14} />
                  </button>
                  <button
                    className="primary full"
                    disabled={!project || !!busy}
                    onClick={() =>
                      total ? setModal("regenerate") : generate()
                    }
                  >
                    <WandSparkles size={16} />
                    {total ? "파츠 초안 다시 생성" : "파츠 초안 생성"}
                  </button>
                  {total > 0 && (
                    <button
                      className="full"
                      style={{ marginTop: 8 }}
                      disabled={!!busy}
                      onClick={() => changeTab("expressions")}
                    >
                      <Smile size={15} />
                      다음: 2. 표정 고르기 →
                    </button>
                  )}
                  <div className="note">
                    <Sparkles size={13} />
                    <span>
                      MobileSAM + 랜드마크 분할 후 클릭 보정(P)과 브러시로
                      세부 경계를 다듬으세요.
                    </span>
                  </div>
                </section>
                )}
                {(tab === "prepare" || (tab === "advanced" && !advancedPreview)) && (
                <section>
                  <div className="section-title">
                    <span>02</span>
                    <h2>선택 파츠 보정</h2>
                  </div>
                  {part ? (
                    <>
                      <div className="selected-part-title">
                        <i style={{ background: part.color }} />
                        <strong>{part.name}</strong>
                        <span>
                          {part.status === "approved"
                            ? "확인됨"
                            : part.status === "missing"
                              ? "미해당"
                              : "초안"}
                        </span>
                      </div>
                      <div className="provenance-row">
                        <span
                          className={`prov-chip prov-${part.provenance ?? "geometry"}`}
                          title="생성 출처 (검토 상태와 별개)"
                        >
                          출처·{partProvenanceLabel(part.provenance)}
                          {part.confidence !== undefined && part.confidence > 0
                            ? ` ${(part.confidence * 100).toFixed(0)}%`
                            : ""}
                        </span>
                        {part.fillKind && part.fillKind !== "none" && (
                          <span className="prov-chip prov-estimated">
                            보완·{part.fillKind === "external" ? "외부PNG" : "인접외삽"}
                          </span>
                        )}
                        {part.diagnostics?.map((diag) => (
                          <span key={diag} className="prov-chip prov-danger">
                            {diagnosticLabel(diag)}
                          </span>
                        ))}
                      </div>
                      <label className="field-label" htmlFor="part-name">
                        레이어 이름
                      </label>
                      <input
                        id="part-name"
                        type="text"
                        key={part.id + part.name}
                        defaultValue={part.name}
                        disabled={!!busy}
                        onBlur={(e) => {
                          const name = e.target.value.trim();
                          if (name && name !== part.name) patchPart({ name });
                        }}
                      />
                      <label className="field-label">
                        브러시 크기<span>{radius}px</span>
                      </label>
                      <input
                        aria-label="브러시 크기"
                        type="range"
                        min="2"
                        max="100"
                        value={radius}
                        disabled={!!busy}
                        onChange={(e) => setRadius(+e.target.value)}
                      />
                      <div className="three-buttons compact">
                        <button
                          className={tool === "prompt" ? "active" : ""}
                          disabled={!!busy}
                          onClick={() => setTool("prompt")}
                          title="좌클릭: 포함(+), 우클릭/Alt+클릭: 제외(-)"
                        >
                          <Crosshair size={14} />
                          클릭 보정
                        </button>
                        <button
                          className={tool === "add" ? "active" : ""}
                          disabled={!!busy}
                          onClick={() => setTool("add")}
                        >
                          <Plus size={14} />
                          추가
                        </button>
                        <button
                          className={tool === "erase" ? "active" : ""}
                          disabled={!!busy}
                          onClick={() => setTool("erase")}
                        >
                          <Eraser size={14} />
                          제거
                        </button>
                      </div>
                      {isHairPart(part) && (
                        <div className="check-options">
                          <label>
                            <input
                              type="checkbox"
                              checked={subtractHairOnPrompt}
                              disabled={!!busy}
                              onChange={(e) =>
                                setSubtractHairOnPrompt(e.target.checked)
                              }
                            />
                            클릭 추출 시 기존 머리카락에서 겹침 빼기
                          </label>
                        </div>
                      )}
                      {(promptResult?.partId === part.id ||
                        (part.prompts && part.prompts.length > 0)) && (
                        <div className="prompt-card">
                          <div className="prompt-card-head">
                            <span>
                              프롬프트 점 {part.prompts?.length ?? 0}개
                            </span>
                            <button
                              className="text-button small"
                              disabled={!!busy}
                              onClick={clearPartPrompts}
                            >
                              점 초기화
                            </button>
                          </div>
                          {promptResult?.partId === part.id && (
                            <div className="candidate-pills">
                              {promptResult.candidates.map((cand) => (
                                <button
                                  key={cand.index}
                                  className={`compact-pill ${
                                    promptCandidateIdx === cand.index ? "active" : ""
                                  }`}
                                  disabled={!!busy}
                                  onClick={() => selectPromptCandidate(cand.index)}
                                >
                                  후보 #{cand.index + 1} ·{" "}
                                  {(cand.iouScore * 100).toFixed(0)}%
                                </button>
                              ))}
                            </div>
                          )}
                        </div>
                      )}
                      <div className="check-options">
                        <label>
                          <input
                            type="checkbox"
                            checked={overlay}
                            onChange={(e) => setOverlay(e.target.checked)}
                          />
                          마스크 색상 표시
                        </label>
                        <label>
                          <input
                            type="checkbox"
                            checked={fillOverlay}
                            onChange={(e) => setFillOverlay(e.target.checked)}
                          />
                          가림 보완 영역 표시
                        </label>
                        <label>
                          <input
                            type="checkbox"
                            checked={compareDiff}
                            onChange={(e) => {
                              const checked = e.target.checked;
                              setCompareDiff(checked);
                              if (checked) setOriginal(false);
                            }}
                          />
                          원본 대비 합성 차이
                        </label>
                        <label>
                          <input
                            type="checkbox"
                            checked={solo}
                            onChange={(e) => setSolo(e.target.checked)}
                          />
                          선택 파츠만 보기
                        </label>
                      </div>
                      <div className="two-buttons">
                        <button disabled={!!busy} onClick={fillPart}>
                          <Sparkles size={14} />
                          가림 보완 ({defaultFillRadius(part)}px)
                        </button>
                        <button
                          disabled={!!busy}
                          onClick={() => partInput.current?.click()}
                        >
                          <Upload size={14} />
                          PNG 교체
                        </button>
                      </div>
                      <button
                        className="text-button small"
                        disabled={!!busy}
                        onClick={fillCriticalOcclusions}
                        style={{ marginTop: 4 }}
                      >
                        얼굴·목·뒷머리 가림 일괄 보완 (36px)
                      </button>
                      {part.fillMask && (
                        <div className="analysis-banner status-model-missing">
                          <div>
                            <div>
                              <strong>가림 보완({part.fillKind ?? "neighbor"})</strong>{" "}
                              가시 마스크와 분리 저장됨
                            </div>
                            {(part.id.startsWith("eye") ||
                              part.id.startsWith("mouth") ||
                              part.id.startsWith("hair")) && (
                              <small>
                                인접 픽셀 외삽만 적용되며 가려진 눈·입·머리카락 내부
                                디테일은 복원되지 않습니다.
                              </small>
                            )}
                            <button
                              className="text-button small"
                              disabled={!!busy}
                              onClick={clearFillMask}
                            >
                              보완 영역만 제거
                            </button>
                          </div>
                        </div>
                      )}
                      <div className="two-buttons compact">
                        <button disabled={!!busy} onClick={() => shiftPart(1)}>
                          <ArrowUp size={13} />
                          앞으로
                        </button>
                        <button disabled={!!busy} onClick={() => shiftPart(-1)}>
                          <ArrowDown size={13} />
                          뒤로
                        </button>
                      </div>
                      <label className="field-row" style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8, marginTop: 6, marginBottom: 6 }}>
                        <span className="tiny">합성 모드 (normal만 PSD 지원)</span>
                        <select
                          aria-label="레이어 합성 모드"
                          disabled={!!busy}
                          value={part.blendMode ?? "normal"}
                          onChange={(e) =>
                            patchPart({
                              blendMode: e.target.value as BlendMode,
                            })
                          }
                        >
                          <option value="normal">일반 (Normal)</option>
                          <option value="multiply">곱하기 (Multiply)</option>
                          <option value="additive">가산 (Additive)</option>
                        </select>
                      </label>
                      <button
                        className={`full approve ${part.status === "approved" ? "is-approved" : ""}`}
                        disabled={
                          !!busy ||
                          !bounds(part.mask, project!.width, project!.height)
                        }
                        onClick={() =>
                          patchPart({
                            status:
                              part.status === "approved" ? "draft" : "approved",
                          })
                        }
                      >
                        <Check size={15} />
                        {part.status === "approved"
                          ? "확인 완료 · 클릭하여 해제"
                          : "이 파츠 확인 완료"}
                      </button>
                      <button
                        className="text-button small"
                        disabled={!!busy}
                        onClick={() =>
                          patchPart({
                            status:
                              part.status === "missing" ? "draft" : "missing",
                          })
                        }
                      >
                        {part.status === "missing"
                          ? "미해당 해제"
                          : "원본에 없는 파츠로 표시"}
                      </button>
                    </>
                  ) : (
                    <div className="placeholder-info">
                      <MousePointer2 size={21} />
                      <p>
                        레이어를 선택하면
                        <br />
                        보정 도구가 표시됩니다.
                      </p>
                    </div>
                  )}
                </section>
                )}
                {tab === "advanced" && advancedPreview && (
                  <>
                    <section>
                  <div className="section-title">
                    <span>01</span>
                    <h2>기본 모션 테스트</h2>
                    <button
                      className="icon-button"
                      aria-label="파라미터 초기화"
                      onClick={() => setParameters({ ...REST })}
                    >
                      <RefreshCw size={13} />
                    </button>
                  </div>
                  <p className="muted">
                    작은 범위의 변형 초안입니다.
                    <br />
                    Cubism의 최종 키폼과 다를 수 있습니다.
                  </p>
                  <div
                    className="param-group-tabs"
                    role="tablist"
                    aria-label="파라미터 그룹 필터"
                  >
                    <button
                      type="button"
                      className={`param-group-tab ${paramGroupFilter === "all" ? "active" : ""}`}
                      onClick={() => setParamGroupFilter("all")}
                    >
                      전체
                      <span className="param-group-badge">
                        {PARAM_DEFS.length}
                      </span>
                    </button>
                    {PARAM_GROUPS.map((g) => (
                      <button
                        type="button"
                        key={g.id}
                        className={`param-group-tab ${paramGroupFilter === g.id ? "active" : ""}`}
                        onClick={() => setParamGroupFilter(g.id)}
                      >
                        {g.label}
                        <span className="param-group-badge">
                          {PARAM_DEFS.filter((d) => d.groupId === g.id).length}
                        </span>
                      </button>
                    ))}
                  </div>
                  {(paramGroupFilter === "all"
                    ? PARAM_DEFS
                    : PARAM_DEFS.filter((d) => d.groupId === paramGroupFilter)
                  ).map((def) => (
                    <div className="parameter" key={def.key}>
                      <label htmlFor={def.key}>
                        {def.name}
                        <span>
                          {parameters[def.key].toFixed(def.max > 1 ? 0 : 2)}
                        </span>
                      </label>
                      <small>{def.id}</small>
                      <input
                        id={def.key}
                        type="range"
                        min={def.previewMin}
                        max={def.previewMax}
                        step={def.max > 1 ? 0.5 : 0.01}
                        value={parameters[def.key]}
                        onChange={(e) =>
                          setParameters((v) => ({
                            ...v,
                            [def.key]: +e.target.value,
                          }))
                        }
                      />
                      <div className="range-ends">
                        <span>{def.previewMin}</span>
                        <span>{def.previewMax}</span>
                      </div>
                    </div>
                  ))}
                  <button
                    className="primary full"
                    onClick={() => setPlaying((v) => !v)}
                  >
                    {playing ? <Pause size={15} /> : <Play size={15} />}자동
                    테스트 {playing ? "정지" : "재생"}
                  </button>
                </section>
                <section>
                  <h2>미리보기 확인 사항</h2>
                  <p className="muted">
                    눈을 감았을 때의 빈 영역, 입 주변 경계와 고개 회전 시 틈을
                    확인하세요. 보완한 파츠를 Cubism에서 정밀하게 마무리합니다.
                  </p>
                  {meshWarnings.length > 0 ? (
                    <div className="analysis-banner status-no-face">
                      <AlertCircle size={15} />
                      <div>
                        <strong>극단 포즈 · 메쉬 품질 경고 ({meshWarnings.length})</strong>
                        {meshWarnings.map((w) => (
                          <div key={w.partId} className="tiny">
                            {w.warnings.join(" / ")}
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : (
                    meshes.size > 0 && (
                      <div className="analysis-banner status-detected">
                        <span>
                          극단 포즈(±30°) 및 구멍 횡단 메쉬 검증을 통과했습니다.
                        </span>
                      </div>
                    )
                  )}
                  <button
                    className="full"
                    onClick={() => {
                      setAdvancedPreview(false);
                      setPlaying(false);
                    }}
                  >
                    파츠 보정으로 돌아가기
                    <ChevronRight size={14} />
                  </button>
                </section>
              </>
            )}
            {tab === "advanced" && (
            <section className="handoff-card">
              <div className="section-title">
                <span>{advancedPreview ? "02" : "03"}</span>
                <h2>Cubism 내보내기 (고급)</h2>
              </div>
              <div className="export-item">
                <FileImage size={17} />
                <span>
                  레이어드 PSD<small>기준(Rest) 파츠 이미지 · 원본 위치</small>
                </span>
                <Check size={13} />
              </div>
              <div className="export-item">
                <Box size={17} />
                <span>
                  리깅 초안 + 작업 가이드
                  <small>메쉬 · 랜드마크 · 변형 샘플</small>
                </span>
                <Check size={13} />
              </div>
              <button
                className="full"
                disabled={!!busy}
                onClick={() => csvInput.current?.click()}
              >
                <FolderOpen size={14} />
                {csv ? "Cubism CSV 샘플 연결됨" : "Cubism CSV 샘플 연결"}
              </button>
              <p className="muted tiny">
                활성 대체 눈·입 모양 및 표정·물리 설정은 .flat2rig에 저장됩니다.
                기본 PSD에는 기준 파츠만 포함되며 메쉬·키폼은 Cubism에서 적용합니다.
              </p>
            </section>
            )}
            </>
            )}
          </div>
          <div className="inspector-footer">
            <span className="local-status">
              <i />
              로컬 작업 공간
            </span>
            <span>flat2rig</span>
          </div>
        </aside>
      </div>
      {modal && (
        <div
          className="modal-backdrop"
          onMouseDown={(e) => {
            if (e.target === e.currentTarget) setModal(null);
          }}
        >
          <div
            className={`modal${modal === "spriteGuide" ? " modal-wide" : ""}`}
            role="dialog"
            aria-modal="true"
            aria-label={
              modal === "export"
                ? "내보내기"
                : modal === "help"
                  ? "사용 가이드"
                  : modal === "spriteGuide"
                    ? "권장 일러스트 및 파츠 예시 시트"
                    : modal === "reanalysis"
                      ? "랜드마크 재추출 확인"
                      : modal === "models"
                        ? "모델 관리"
                        : modal === "updates"
                          ? "앱 자동 업데이트"
                          : "파츠 재생성"
            }
          >
            <button
              className="modal-close icon-button"
              aria-label="닫기"
              onClick={() => setModal(null)}
            >
              <X size={19} />
            </button>
            {modal === "export" ? (
              <>
                <span className="eyebrow">EXPORT</span>
                <h2>내보내기</h2>
                <p>현재 표정 이미지(PNG) 또는 후반 작업용 Cubism 자료를 내보냅니다.</p>
                <div className="export-summary">
                  <span>
                    {project?.width} × {project?.height}px
                  </span>
                  <span>{total}개 파츠</span>
                  <span>{approved}개 확인됨</span>
                </div>
                {approved < total && (
                  <div className="export-warning">
                    확인하지 않은 초안과 미해당 파츠가 있습니다. 상태를 작업
                    가이드에 기록합니다.
                  </div>
                )}
                <button
                  className="export-choice"
                  onClick={exportCurrentFramePng}
                >
                  <span className="choice-icon">
                    <FileImage size={22} />
                  </span>
                  <span>
                    <strong>현재 표정 PNG</strong>
                    <small>오버레이 제외 · 투명 배경 · 현재 자세/모양 반영</small>
                  </span>
                  <Download size={18} />
                </button>
                <button
                  className="export-choice"
                  onClick={() => output("bundle")}
                >
                  <span className="choice-icon">
                    <Box size={22} />
                  </span>
                  <span>
                    <strong>Cubism 인계 패키지 (고급)</strong>
                    <small>
                      PSD · 파츠 PNG · 프로젝트 · JSON · CSV · 가이드
                    </small>
                  </span>
                  <Download size={18} />
                </button>
                <button className="export-choice" onClick={() => output("psd")}>
                  <span className="choice-icon">
                    <FileImage size={22} />
                  </span>
                  <span>
                    <strong>레이어드 PSD만 (고급)</strong>
                    <small>기준(Rest) 파츠 이미지 레이어 · normal 모드 · RGB 8bit</small>
                  </span>
                  <Download size={18} />
                </button>
                <p className="muted tiny">
                  활성 대체 눈·입 모양 및 저장한 표정·물리 설정은 .flat2rig 프로젝트에만
                  저장됩니다. 기본 PSD에는 기준 원본 파츠만 포함되며, 자체 메쉬·변형 키는
                  저장되지 않습니다. 실제 Cubism Editor 가져오기 호환성은 별도 확인이
                  필요합니다.
                </p>
              </>
            ) : modal === "reanalysis" && pendingReanalysis && project ? (
              <>
                <span className="eyebrow">PRESERVE MANUAL EDITS</span>
                <h2>수동 보정한 랜드마크가 있습니다</h2>
                <p>
                  사용자가 직접 드래그한 좌표({pendingReanalysis.manualKeys.join(", ")})를
                  유지할지, 새 자동 예측으로 덮어쓸지 선택하세요.
                </p>
                <div className="reanalysis-diffs">
                  {(["eyeR", "eyeL", "mouth", "chin", "crown"] as LandmarkId[]).map((k) => {
                    const cur = project.landmarks[k];
                    const next = pendingReanalysis.response.rigLandmarks?.[k];
                    const diffPx = next
                      ? Math.hypot(cur.x - next.x, cur.y - next.y).toFixed(1)
                      : "0.0";
                    const isManual = pendingReanalysis.manualKeys.includes(k);
                    return (
                      <div key={k} className="diff-row">
                        <span>
                          <strong>{k}</strong>
                          {isManual && <em>수동</em>}
                        </span>
                        <span>차이 {diffPx}px</span>
                      </div>
                    );
                  })}
                </div>
                <div className="two-buttons">
                  <button
                    className="primary"
                    onClick={() => {
                      const next = applyAnalysisResponse(
                        project,
                        pendingReanalysis.response,
                        true,
                      );
                      commit(next, project);
                      setPendingReanalysis(null);
                      setModal(null);
                      notify("수동 보정 좌표를 보존하고 나머지 결과를 적용했습니다.");
                    }}
                  >
                    수동 수정 보존
                  </button>
                  <button
                    onClick={() => {
                      const next = applyAnalysisResponse(
                        project,
                        pendingReanalysis.response,
                        false,
                      );
                      commit(next, project);
                      setPendingReanalysis(null);
                      setModal(null);
                      notify(pendingReanalysis.response.message);
                    }}
                  >
                    모두 덮어쓰기
                  </button>
                </div>
                <button
                  className="text-button full"
                  onClick={() => {
                    setPendingReanalysis(null);
                    setModal(null);
                  }}
                >
                  취소하고 현재 좌표 유지
                </button>
              </>
            ) : modal === "models" ? (
              <>
                <span className="eyebrow">LOCAL ONNX RUNTIME</span>
                <h2>ONNX 모델 관리 · SHA-256 검증</h2>
                <p>
                  models/manifest.json 기준으로 무결성을 검증하며 모든 추론은
                  로컬 기기에서 실행됩니다.
                </p>
                <div className="model-list">
                  {modelStatus.length === 0 ? (
                    <p className="muted tiny">
                      {desktop
                        ? "등록된 모델 상태를 불러오는 중입니다."
                        : "데스크톱 앱(tauri) 환경에서 로컬 ONNX 모델을 연결·검증할 수 있습니다."}
                    </p>
                  ) : (
                    modelStatus.map((entry) => {
                      const busySelf = modelBusyRole === entry.role;
                      const sizeMb = (entry.expectedBytes / (1024 * 1024)).toFixed(1);
                      return (
                        <div key={entry.role} className="model-row">
                          <div className="model-row-head">
                            <div>
                              <strong>{ROLE_LABELS[entry.role] ?? entry.role}</strong>
                              <small>
                                {entry.filename} · {sizeMb} MB · {entry.license}
                              </small>
                            </div>
                            <span
                              className={`prov-chip ${
                                entry.hashMatches
                                  ? "prov-manual"
                                  : entry.installed
                                    ? "prov-estimated"
                                    : "prov-danger"
                              }`}
                            >
                              {entry.hashMatches
                                ? "SHA-256 일치"
                                : entry.installed
                                  ? "해시 불일치"
                                  : "미설치"}
                            </span>
                          </div>
                          <div className="model-row-meta">
                            <code>sha256: {entry.expectedSha256.slice(0, 16)}…</code>
                            {entry.path && <span title={entry.path}>{entry.path}</span>}
                          </div>
                          {entry.error && !entry.hashMatches && (
                            <p className="tiny muted">{entry.error}</p>
                          )}
                          <div className="candidate-pills">
                            <button
                              className="compact-pill"
                              disabled={!desktop || !!busy || busySelf}
                              onClick={() => void linkModelRole(entry.role)}
                            >
                              <FolderOpen size={12} />
                              로컬 .onnx 연결
                            </button>
                            {entry.downloadUrl && !entry.hashMatches && (
                              <button
                                className="compact-pill active"
                                disabled={!desktop || !!busy || busySelf}
                                onClick={() => void downloadModelRole(entry.role)}
                              >
                                <Download size={12} />
                                {busySelf ? "다운로드 중…" : "자동 다운로드"}
                              </button>
                            )}
                            {!entry.downloadUrl && !entry.hashMatches && (
                              <span className="tiny muted">
                                python tools/model-export/export_models.py 실행 또는 파일 연결
                              </span>
                            )}
                            {entry.path && (
                              <button
                                className="compact-pill"
                                disabled={!desktop || !!busy || busySelf}
                                onClick={() => void unlinkModelRole(entry.role)}
                              >
                                연결 해제
                              </button>
                            )}
                          </div>
                        </div>
                      );
                    })
                  )}
                </div>
                {desktop && (
                  <button
                    className="full"
                    disabled={!!modelBusyRole}
                    onClick={() => {
                      void getModelStatus()
                        .then(setModelStatus)
                        .catch((e) => setError(String(e)));
                    }}
                  >
                    <RefreshCw size={14} />
                    상태 새로고침
                  </button>
                )}
              </>
            ) : modal === "regenerate" ? (
              (() => {
                const hasPreservedParts =
                  project?.parts.some((t) => isPreservedPart(t)) ?? false;
                return (
                  <>
                    <h2>파츠 초안을 다시 만들까요?</h2>
                    <p>
                      {hasPreservedParts
                        ? "수동 편집하거나 확인함으로 표시한 파츠와 추가 머리카락은 유지한 채 나머지 파츠만 다시 생성하거나, 전체를 새로 초기화할 수 있습니다."
                        : "현재 마스크와 파츠 보완 이미지가 새 초안으로 바뀝니다. 작업 후 실행 취소로 복원할 수 있습니다."}
                    </p>
                    {hasPreservedParts && (
                      <button
                        className="primary full"
                        style={{ marginBottom: 10 }}
                        onClick={() => generate(undefined, undefined, true)}
                      >
                        수정·확인 파츠 유지 후 다시 생성
                      </button>
                    )}
                    <div className="two-buttons">
                      <button onClick={() => setModal(null)}>돌아가기</button>
                      <button
                        className={hasPreservedParts ? "" : "primary"}
                        onClick={() => generate(undefined, undefined, false)}
                      >
                        {hasPreservedParts ? "전체 초기화" : "다시 생성"}
                      </button>
                    </div>
                  </>
                );
              })()
            ) : modal === "spriteGuide" ? (
              <>
                <div className="sprite-guide-head-row">
                  <div>
                    <span className="eyebrow">SPRITE PREPARATION GUIDE</span>
                    <h2>권장 일러스트 구도 &amp; LD/SD 실루엣 21파츠 예시</h2>
                  </div>
                  <div className="sprite-guide-zoom-controls" role="group" aria-label="시트 배율">
                    {(["fit", 100, 150, 200] as const).map((level) => (
                      <button
                        key={String(level)}
                        className={`compact-pill${guideZoom === level ? " active" : ""}`}
                        onClick={() => setGuideZoom(level)}
                      >
                        {level === "fit" ? "창 맞춤" : `${level}%`}
                      </button>
                    ))}
                  </div>
                </div>
                <p>
                  일반 등신(LD)과 SD 치비(2~3등신)를 자동 구분해 모션 예비 파츠(원형 동공·눈꺼풀·입안·뒷머리)까지 구성합니다. 배율 버튼이나 이미지를 클릭해 크게 확대해 확인하세요.
                </p>
                {spriteGuideUrl && (
                  <div className="sprite-guide-viewport">
                    <img
                      className="sprite-guide-modal-img"
                      style={
                        guideZoom === "fit"
                          ? { width: "100%", height: "auto" }
                          : {
                              width: `${Math.round(12.4 * guideZoom)}px`,
                              maxWidth: "none",
                              height: "auto",
                              cursor: guideZoom === 200 ? "zoom-out" : "zoom-in",
                            }
                      }
                      src={spriteGuideUrl}
                      alt="일러스트 구도 및 21개 파츠 스프라이트 분해 예시"
                      onClick={() =>
                        setGuideZoom((z) => (z === "fit" ? 150 : z === 150 ? 200 : "fit"))
                      }
                      title="클릭하여 창 맞춤 / 150% / 200% 전환"
                    />
                  </div>
                )}
                <div className="two-buttons" style={{ marginTop: 14 }}>
                  <button onClick={() => setModal("help")}>
                    ← 사용 가이드로
                  </button>
                  <button
                    onClick={() => {
                      downloadSpriteGuide();
                      notify("스프라이트 준비 가이드 PNG를 저장했습니다.");
                    }}
                  >
                    <Download size={14} />
                    고해상도 가이드 PNG 저장
                  </button>
                  <button
                    className="primary"
                    onClick={() => {
                      setModal(null);
                      void demoFile().then(openImage);
                    }}
                  >
                    샘플 일러스트로 체험
                  </button>
                </div>
              </>
            ) : modal === "updates" ? (
              <>
                <span className="eyebrow">AUTO UPDATE</span>
                <h2>앱 버전 및 자동 업데이트</h2>
                <p>
                  현재 설치된 버전은 <strong>v{appVersion}</strong>입니다. 서명 검증(Minisign)된 공식 업데이트 매니페스트만 자동 설치합니다.
                </p>
                <div className="update-box">
                  <div className="model-row-head">
                    <div>
                      <strong>
                        {updateInfo?.status === "available"
                          ? `새 버전 v${updateInfo.latestVersion ?? ""} 사용 가능`
                          : updateInfo?.status === "unsigned-release"
                            ? `릴리스 v${updateInfo.latestVersion ?? ""} (수동 다운로드 안내)`
                            : updateInfo?.status === "up-to-date"
                              ? "최신 버전 사용 중"
                              : checkingUpdate
                                ? "업데이트 확인 중…"
                                : "업데이트 상태"}
                      </strong>
                      <small>
                        {updateInfo?.message ??
                          "업데이트 확인 버튼을 눌러 최신 배포 버전을 조회하세요."}
                      </small>
                    </div>
                    <span
                      className={`prov-chip ${
                        updateInfo?.status === "available"
                          ? "prov-manual"
                          : updateInfo?.status === "up-to-date"
                            ? "prov-model"
                            : "prov-estimated"
                      }`}
                    >
                      {updateInfo?.status === "available"
                        ? "설치 가능"
                        : updateInfo?.status === "up-to-date"
                          ? "최신"
                          : updateInfo?.status === "unsigned-release"
                            ? "서명 미배포"
                            : "대기"}
                    </span>
                  </div>
                  {updateInfo?.notes && (
                    <pre className="update-notes">{updateInfo.notes}</pre>
                  )}
                  {installingUpdate && updateProgressPct !== null && (
                    <div className="update-progress">
                      <div
                        className="update-progress-bar"
                        style={{ width: `${updateProgressPct}%` }}
                      />
                    </div>
                  )}
                </div>
                <label className="style-preset-field" style={{ marginTop: 10 }}>
                  <span>업데이트 매니페스트 URL (선택 · 비우면 공식 GitHub Releases 사용)</span>
                  <input
                    type="url"
                    placeholder="https://github.com/jinhoOps/flat2rig/releases/latest/download/latest.json"
                    value={updateCustomUrl}
                    disabled={checkingUpdate || installingUpdate}
                    onChange={(e) => setUpdateCustomUrl(e.target.value)}
                  />
                </label>
                <div className="two-buttons" style={{ marginTop: 14 }}>
                  <button
                    disabled={checkingUpdate || installingUpdate}
                    onClick={() => void runUpdateCheck()}
                  >
                    <RefreshCw size={14} className={checkingUpdate ? "spin" : ""} />
                    {checkingUpdate ? "확인 중…" : "지금 다시 확인"}
                  </button>
                  {updateDownloaded ? (
                    <button className="primary" onClick={() => void relaunchApp()}>
                      앱 재시작하여 업데이트 적용
                    </button>
                  ) : updateInfo?.status === "available" ? (
                    <button
                      className="primary"
                      disabled={installingUpdate || !desktop}
                      onClick={() => void runUpdateInstall()}
                    >
                      <Download size={14} />
                      {installingUpdate
                        ? `설치 중 (${updateProgressPct ?? 0}%)`
                        : "지금 다운로드 및 설치"}
                    </button>
                  ) : (
                    <button onClick={() => setModal(null)}>닫기</button>
                  )}
                </div>
              </>
            ) : (
              <>
                <span className="eyebrow">GUIDE</span>
                <h2>작업 순서</h2>
                <div className="modal-actions-row">
                  <button
                    className="primary"
                    onClick={() => {
                      setGuideZoom("fit");
                      setModal("spriteGuide");
                    }}
                  >
                    <CircleHelp size={15} />
                    권장 일러스트 · 파츠 예시 시트 확대 보기
                  </button>
                  <button
                    onClick={() => {
                      downloadSpriteGuide();
                      notify("스프라이트 준비 가이드 PNG를 저장했습니다.");
                    }}
                  >
                    <Download size={14} />
                    가이드 PNG 저장
                  </button>
                </div>
                <ol className="help-list">
                  <li>
                    <strong>이미지 열기 (일반 LD / SD 치비)</strong>정면 흉상 PNG/JPEG를 준비합니다.
                    작업 해상도는 긴 변 최대 2,048px입니다. 일반 등신과 SD 치비는 자동 판별되며 실루엣 패널에서 수동 전환할 수 있습니다.
                  </li>
                  <li>
                    <strong>얼굴과 랜드마크 맞추기</strong>YOLOv3+HRNetV2가
                    얼굴과 28점 랜드마크를 자동 추출합니다. 다중 얼굴 후보를
                    전환하거나 F/L로 수동 수정하세요.
                  </li>
                  <li>
                    <strong>파츠 분리·클릭 보정</strong>MobileSAM과 랜드마크
                    하이브리드 분할을 실행하고 클릭 보정(P)·브러시(B/E)로
                    다듬습니다.
                  </li>
                  <li>
                    <strong>가림 보완·차이 확인</strong>가림 보완은 가시 마스크와
                    분리된 12px 인접색 외삽이며, 눈·입·머리카락 내부 디테일은
                    생성되지 않으므로 필요시 PNG로 교체하세요.
                  </li>
                  <li>
                    <strong>움직임 확인·내보내기</strong>윤곽·랜드마크 메쉬와
                    극단 포즈 경고를 점검한 뒤 Cubism용 PSD·패키지를 내보냅니다.
                  </li>
                </ol>
                <p className="muted tiny">
                  저장 Ctrl+S · 실행 취소 Ctrl+Z · 다시 실행 Ctrl+Shift+Z · 선택
                  V · 얼굴 F · 랜드마크 L · 클릭 보정 P · 브러시 B · 지우개 E
                </p>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
