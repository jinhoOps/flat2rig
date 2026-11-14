import { useCallback, useEffect, useRef, useState } from "react";
import {
  ArrowDown,
  ArrowUp,
  Box,
  Check,
  CheckCircle2,
  ChevronRight,
  CircleHelp,
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
  SlidersHorizontal,
  Sparkles,
  Undo2,
  Upload,
  WandSparkles,
  X,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import type {
  LandmarkId,
  Mesh,
  Parameters,
  Part,
  Point,
  Project,
  Tool,
} from "./types";
import { PARAM_DEFS, REST } from "./types";
import { bounds, clamp, faceBoxLandmarks, paint } from "./engine";
import { loadImage, pixels, pixelsCanvas, readImage } from "./images";
import {
  buildRig,
  chooseModel,
  desktop,
  fillNative,
  saveFile,
  separateNative,
} from "./native";
import {
  exportBundle,
  exportPsd,
  loadProject,
  newPart,
  readCsvTemplate,
  safeName,
  saveProject,
  type CsvTemplate,
} from "./io";
import Viewport from "./Viewport";
import { demoFile } from "./demo";

export default function App() {
  const [project, setProject] = useState<Project | null>(null),
    projectRef = useRef<Project | null>(null);
  const [selected, setSelected] = useState("face"),
    [tool, setTool] = useState<Tool>("landmark");
  const [tab, setTab] = useState<"prepare" | "preview">("prepare");
  const [zoom, setZoom] = useState(1),
    [radius, setRadius] = useState(18),
    [tolerance, setTolerance] = useState(32);
  const [overlay, setOverlay] = useState(true),
    [wire, setWire] = useState(false),
    [original, setOriginal] = useState(false),
    [solo, setSolo] = useState(false);
  const [parameters, setParameters] = useState<Parameters>({ ...REST }),
    [playing, setPlaying] = useState(false);
  const [meshes, setMeshes] = useState<Map<string, Mesh>>(new Map()),
    [meshing, setMeshing] = useState(false);
  const [modelPath, setModelPath] = useState<string | null>(null),
    [csv, setCsv] = useState<CsvTemplate | null>(null);
  const [busy, setBusy] = useState(""),
    [message, setMessage] = useState(""),
    [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  const dragDepth = useRef(0);
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
  const [modal, setModal] = useState<"export" | "regenerate" | "help" | null>(
      null,
    ),
    [dirty, setDirty] = useState(false);
  const [historyCount, setHistoryCount] = useState([0, 0]);
  const history = useRef<Project[]>([]),
    future = useRef<Project[]>([]),
    controller = useRef<AbortController | null>(null);
  const input = useRef<HTMLInputElement>(null),
    projectInput = useRef<HTMLInputElement>(null),
    partInput = useRef<HTMLInputElement>(null),
    csvInput = useRef<HTMLInputElement>(null);
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
      if (before) history.current.push(before);
      if (history.current.length > 24) history.current.shift();
      future.current = [];
      setHistoryCount([history.current.length, 0]);
      setDirty(true);
      update({ ...p, revision: (projectRef.current?.revision ?? 0) + 1 });
    },
    [update],
  );
  const notify = (text: string) => {
    setMessage(text);
    setError("");
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
    history.current = [];
    future.current = [];
    setHistoryCount([0, 0]);
    setMeshes(new Map());
    setParameters({ ...REST });
    setPlaying(false);
    setZoom(1);
    setSolo(false);
    setOriginal(false);
    setTab("prepare");
    setTool(p.parts.length ? "select" : "face");
    setSelected(
      p.parts.find((t) => t.id === "face")?.id ?? p.parts[0]?.id ?? "face",
    );
    setDirty(false);
    update(p);
  };
  const openImage = (file?: File) => {
    if (!file) return;
    if (
      dirty &&
      !confirm("저장하지 않은 변경 사항이 있습니다. 새 이미지를 여시겠습니까?")
    )
      return;
    void task("이미지 준비 중", async (signal) => {
      const p = await readImage(file);
      signal.throwIfAborted();
      reset(p);
      notify(
        `작업 캔버스 ${p.width} × ${p.height}px. 얼굴 영역을 드래그한 뒤 눈·입 위치를 맞추세요.`,
      );
    });
  };
  const generate = () => {
    const p = projectRef.current;
    if (!p) return;
    setModal(null);
    void task("파츠 분리 초안 생성 중", async (signal) => {
      const data = pixels(await loadImage(p.source), p.width, p.height);
      const result = await separateNative(
        data,
        p.width,
        p.height,
        p.landmarks,
        tolerance,
        modelPath,
        signal,
      );
      signal.throwIfAborted();
      commit({ ...p, ...result });
      setTool("select");
      setTab("prepare");
      notify(
        "18개 파츠 초안이 준비됐습니다. 마스크 경계와 가림 영역을 확인하세요.",
      );
    });
  };
  const undo = useCallback(() => {
    if (controller.current || !history.current.length || !projectRef.current)
      return;
    future.current.push(projectRef.current);
    const p = history.current.pop()!;
    update({ ...p, revision: projectRef.current.revision + 1 });
    setHistoryCount([history.current.length, future.current.length]);
    setDirty(true);
  }, [update]);
  const redo = useCallback(() => {
    if (controller.current || !future.current.length || !projectRef.current)
      return;
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
      if (
        (e.target as HTMLElement).matches("input,textarea,select") ||
        controller.current
      )
        return;
      if (e.ctrlKey || e.metaKey) {
        if (e.key === "z") {
          e.preventDefault();
          e.shiftKey ? redo() : undo();
        }
        if (e.key === "y") {
          e.preventDefault();
          redo();
        }
        if (e.key === "s") {
          e.preventDefault();
          document
            .querySelector<HTMLButtonElement>('[data-action="save"]')
            ?.click();
        }
        return;
      }
      const keys: Record<string, Tool> = {
        v: "select",
        b: "add",
        e: "erase",
        l: "landmark",
        f: "face",
      };
      if (keys[e.key]) setTool(keys[e.key]);
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
  }, [undo, redo, dirty]);
  useEffect(() => {
    if (!project || !project.parts.length || (!wire && tab !== "preview"))
      return;
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
          if (active) setMeshes(new Map(rig.map((r) => [r.id, r.mesh])));
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
  }, [project?.revision, project?.source, wire, tab]);
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
    commit({ ...project, parts });
  };
  const start = (pt: Point) => {
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
    gesture.current = { before: p, part: selected, start: pt, last: pt };
    const parts = p.parts.map((t) =>
      t.id === selected
        ? { ...t, status: "draft" as const, mask: t.mask.slice() }
        : t,
    );
    paint(
      parts.find((t) => t.id === selected)!.mask,
      p.width,
      p.height,
      pt,
      pt,
      radius,
      tool === "erase",
    );
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
        revision: p.revision + 1,
      });
    else if (tool === "face") {
      if (Math.abs(g.start.x - pt.x) > 8 && Math.abs(g.start.y - pt.y) > 8)
        update({
          ...p,
          landmarks: faceBoxLandmarks(g.start, pt),
          revision: p.revision + 1,
        });
    } else if (g.part) {
      const part = p.parts.find((t) => t.id === g.part)!;
      paint(part.mask, p.width, p.height, g.last, pt, radius, tool === "erase");
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
  const fillPart = () => {
    const p = projectRef.current,
      part = p?.parts.find((t) => t.id === selected);
    if (!p || !part) return;
    void task("선택 파츠의 가림 영역 보완 중", async (signal) => {
      const data = pixels(
        await loadImage(part.texture ?? p.source),
        p.width,
        p.height,
      );
      const result = await fillNative(data, part.mask, p.width, p.height, 12);
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
                mask: result.mask,
                texture,
                generated: true,
                status: "draft",
              }
            : t,
        ),
      });
      notify(
        "경계에서 12px를 인접색으로 확장했습니다. 정밀 복원은 PNG 교체로 보완하세요.",
      );
    });
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
        texture: replacement.source,
        generated: true,
        status: "draft",
      });
      notify("파츠 이미지를 교체했습니다.");
    });
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
  const total = project?.parts.length ?? 0,
    preview = tab === "preview";
  const changeTab = (t: "prepare" | "preview") => {
    setTab(t);
    if (t === "preview") {
      setOriginal(false);
      setOverlay(false);
      setSolo(false);
    } else setPlaying(false);
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
            const p = loadProject(new Uint8Array(await file.arrayBuffer()));
            await loadImage(p.source);
            signal.throwIfAborted();
            reset(p);
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
          <img src="/logo.svg" alt="" />
          <span>
            flat<span className="brand-two">2</span>rig
          </span>
          <small>STUDIO</small>
        </a>
        <div className="header-divider" />
        <span className="project-title">
          {project?.name ?? "새 프로젝트"}
          {dirty && <i className="dirty-dot" />}
        </span>
        <span className="version">MVP / 0.1</span>
        <div className="header-actions">
          <span className="local-status">
            <i />
            {desktop ? "로컬 처리" : "브라우저 미리보기"}
          </span>
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
      <div className="workspace">
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
              [...project!.parts].reverse().map((p, i) => (
                <button
                  className={`layer ${selected === p.id ? "selected" : ""} ${p.status === "missing" ? "missing" : ""}`}
                  key={p.id}
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
                    <small>{p.id}</small>
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
                    {p.status === "approved" && <Check size={10} />}
                  </span>
                  <GripVertical className="grip" size={12} />
                  <span className="sr-only">{i + 1}</span>
                </button>
              ))
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
        <main className="editor">
          <div className="editor-tabs">
            <div role="tablist">
              <button
                role="tab"
                aria-selected={!preview}
                className={!preview ? "active" : ""}
                onClick={() => changeTab("prepare")}
              >
                <ScanFace size={15} />
                파츠 준비
              </button>
              <button
                role="tab"
                aria-selected={preview}
                disabled={!total || !!busy}
                className={preview ? "active" : ""}
                onClick={() => changeTab("preview")}
              >
                <SlidersHorizontal size={15} />
                리깅 미리보기
              </button>
            </div>
            <span className="stage-tag">
              {preview ? "RIG DRAFT" : "MATERIAL SEPARATION"}
            </span>
          </div>
          <div className="toolbar">
            <div className="tool-group">
              {(
                [
                  { key: "select", label: "선택 (V)", icon: MousePointer2 },
                  { key: "face", label: "얼굴 영역 지정 (F)", icon: ScanFace },
                  { key: "landmark", label: "랜드마크 이동 (L)", icon: Move },
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
                    (!total && (t.key === "add" || t.key === "erase"))
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
                  onChange={(e) => setOriginal(e.target.checked)}
                />
                원본
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
          <div className={`stage ${busy ? "working" : ""}`}>
            {project ? (
              <>
                <div className="canvas-caption">
                  <span>
                    <i />
                    {original
                      ? "ORIGINAL"
                      : preview
                        ? "MOTION STUDY"
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
                  wire={wire}
                  solo={solo}
                  preview={preview}
                  playing={playing}
                  parameters={parameters}
                  meshes={meshes}
                  onStart={start}
                  onMove={move}
                  onEnd={end}
                />
                <div className="stage-bottom">
                  <span>
                    {preview
                      ? "작은 변형으로 파츠와 보완 영역을 확인하세요"
                      : tool === "face"
                        ? "얼굴 피부 영역의 좌상단에서 우하단까지 드래그하세요"
                        : tool === "landmark"
                          ? "눈 · 입 · 턱 · 얼굴 상단의 점을 드래그하세요"
                          : tool === "add" || tool === "erase"
                            ? "드래그하여 선택 파츠의 마스크를 수정하세요"
                            : "파츠를 선택하여 경계를 확인하세요"}
                  </span>
                  <div>
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
                  <div className="welcome-art">
                    <div className="welcome-icon">
                      <ImagePlus size={32} aria-hidden="true" />
                    </div>
                  </div>
                  <h1>일러스트 가져오기</h1>
                  <p>
                    PNG 또는 JPG 파일 한 장을 끌어 놓으세요.
                    <br />
                    정면 캐릭터 흉상 · 작업 해상도 최대 2,048px
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
                    파츠 분리 · 마스크 보정 · Cubism용 PSD 내보내기
                  </span>
                  <button
                    className="text-button"
                    disabled={!!busy}
                    onClick={() => {
                      void demoFile().then(openImage);
                    }}
                  >
                    샘플 이미지 열기
                    <ChevronRight size={13} />
                  </button>
                </div>
                <div className="welcome-features">
                  <span>
                    <Layers size={14} />
                    18개 파츠 초안
                  </span>
                  <span>
                    <Box size={14} />
                    메쉬 · 변형 확인
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
              {preview ? "파라미터" : "작업 설정"}
            </span>
            <span className="tiny-label">
              {preview ? "PREVIEW" : "INSPECTOR"}
            </span>
          </div>
          <div className="inspector-scroll">
            {!preview ? (
              <>
                <section>
                  <div className="section-title">
                    <span>01</span>
                    <h2>분리 초안 생성</h2>
                  </div>
                  <p className="muted">
                    얼굴 영역과 랜드마크를 먼저 맞추세요.
                    <br />
                    분리 결과는 직접 확인하고 보정합니다.
                  </p>
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
                  <button
                    className="model-button"
                    disabled={!desktop || !!busy}
                    onClick={() => {
                      void chooseModel()
                        .then((path) => {
                          if (typeof path === "string") setModelPath(path);
                        })
                        .catch((e) => setError(String(e)));
                    }}
                  >
                    <Box size={15} />
                    <span>
                      {modelPath
                        ? modelPath.split(/[\\/]/).pop()
                        : "U2Net ONNX 모델 연결"}
                      <small>
                        {desktop
                          ? "배경 제거용 로컬 ONNX 모델"
                          : "데스크톱 앱에서 사용 가능"}
                      </small>
                    </span>
                    <ChevronRight size={14} />
                  </button>
                  {modelPath && (
                    <button
                      className="text-button small"
                      onClick={() => setModelPath(null)}
                    >
                      모델 연결 해제
                    </button>
                  )}
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
                  <div className="note">
                    <Sparkles size={13} />
                    <span>
                      세부 파츠는 랜드마크 기반 기하 초안입니다. 복잡한 가림은
                      보정이 필요합니다.
                    </span>
                  </div>
                </section>
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
                      <div className="two-buttons">
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
                            checked={solo}
                            onChange={(e) => setSolo(e.target.checked)}
                          />
                          선택 파츠만 보기
                        </label>
                      </div>
                      <div className="two-buttons">
                        <button disabled={!!busy} onClick={fillPart}>
                          <Sparkles size={14} />
                          가림 보완
                        </button>
                        <button
                          disabled={!!busy}
                          onClick={() => partInput.current?.click()}
                        >
                          <Upload size={14} />
                          PNG 교체
                        </button>
                      </div>
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
              </>
            ) : (
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
                  {PARAM_DEFS.map((def) => (
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
                  <button className="full" onClick={() => changeTab("prepare")}>
                    파츠 보정으로 돌아가기
                    <ChevronRight size={14} />
                  </button>
                </section>
              </>
            )}
            <section className="handoff-card">
              <div className="section-title">
                <span>{preview ? "02" : "03"}</span>
                <h2>Cubism 내보내기</h2>
              </div>
              <div className="export-item">
                <FileImage size={17} />
                <span>
                  레이어드 PSD<small>파츠 이미지 · 원본 위치</small>
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
                샘플 연결 전에는 참고용 파라미터 목록을 내보냅니다. 메쉬·키폼은
                Cubism에서 적용합니다.
              </p>
            </section>
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
            className="modal"
            role="dialog"
            aria-modal="true"
            aria-label={
              modal === "export"
                ? "내보내기"
                : modal === "help"
                  ? "사용 가이드"
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
                <h2>Cubism용 파일 내보내기</h2>
                <p>파츠와 기초 리깅 자료를 함께 내보냅니다.</p>
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
                  onClick={() => output("bundle")}
                >
                  <span className="choice-icon">
                    <Box size={22} />
                  </span>
                  <span>
                    <strong>Cubism 인계 패키지</strong>
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
                    <strong>레이어드 PSD만</strong>
                    <small>이미지 레이어와 투명도 · RGB 8bit</small>
                  </span>
                  <Download size={18} />
                </button>
                <p className="muted tiny">
                  자체 메쉬·변형 키는 PSD에 저장되지 않습니다. Cubism Editor에서
                  후반 작업을 진행하세요. 실제 Editor 가져오기 호환성은 별도
                  확인이 필요합니다.
                </p>
              </>
            ) : modal === "regenerate" ? (
              <>
                <h2>파츠 초안을 다시 만들까요?</h2>
                <p>
                  현재 마스크와 파츠 보완 이미지가 새 초안으로 바뀝니다. 작업 후
                  실행 취소로 복원할 수 있습니다.
                </p>
                <div className="two-buttons">
                  <button onClick={() => setModal(null)}>돌아가기</button>
                  <button className="primary" onClick={generate}>
                    다시 생성
                  </button>
                </div>
              </>
            ) : (
              <>
                <span className="eyebrow">GUIDE</span>
                <h2>작업 순서</h2>
                <ol className="help-list">
                  <li>
                    <strong>이미지 열기</strong>정면 흉상 PNG/JPEG를 준비합니다.
                    작업 해상도는 긴 변 최대 2,048px입니다.
                  </li>
                  <li>
                    <strong>얼굴과 랜드마크 맞추기</strong>F로 얼굴 영역을
                    드래그하고 L로 눈·입·턱 위치를 보정합니다. 상단 점은 얼굴
                    피부의 상단입니다.
                  </li>
                  <li>
                    <strong>분리 초안 확인</strong>U2Net 모델을 연결하면 Rust
                    ort로 배경을 제거합니다. 세부 파츠는 기하 템플릿이므로 B/E
                    브러시로 보정하세요.
                  </li>
                  <li>
                    <strong>빈 곳 보완</strong>가림 보완은 12px 인접색
                    확장입니다. 정교한 그리기는 같은 캔버스 크기의 PNG로
                    교체합니다.
                  </li>
                  <li>
                    <strong>움직임 확인·내보내기</strong>미리보기 후 PSD와
                    가이드를 내보냅니다. Cubism에서 메쉬·디포머·변형 키를
                    마무리하세요.
                  </li>
                </ol>
                <p className="muted tiny">
                  저장 Ctrl+S · 실행 취소 Ctrl+Z · 다시 실행 Ctrl+Shift+Z · 선택
                  V · 얼굴 F · 랜드마크 L · 브러시 B · 지우개 E
                </p>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
