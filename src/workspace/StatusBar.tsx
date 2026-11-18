import { CheckCircle2, Loader2, X } from "lucide-react";
import CatProgress from "./CatProgress";
import type { CombinedProgress } from "./progress";

export interface StatusBarProps {
  zoom: number;
  width?: number;
  height?: number;
  parts: number;
  vertices: number;
  meshing: boolean;
  message: string;
  error: string;
  busy: string;
  progress: CombinedProgress;
  onFit: () => void;
}

/** Bottom status bar: zoom · canvas px · part count · message (keeps `role="status"`/`alert`). */
export default function StatusBar(props: StatusBarProps) {
  const { zoom, width, height, parts, error, message, busy } = props;
  return (
    <footer className="ws-status editor-status" aria-label="상태 표시줄">
      <button type="button" className="sb-zoom" onClick={props.onFit} title="화면에 맞춤" disabled={!width}>
        {Math.round(zoom * 100)}%
      </button>
      <span className="sb-cell">{width && height ? `${width} × ${height} px` : "— px"}</span>
      <span className="sb-cell">{parts}개 파츠</span>
      <span className="sb-cell sb-mesh">
        {props.meshing ? (
          <>
            <Loader2 size={11} className="spin" />
            메쉬 계산 중
          </>
        ) : width ? (
          `${props.vertices} VERTICES`
        ) : (
          "READY"
        )}
      </span>
      <span className={`sb-msg${error ? " error-text" : ""}`}>
        {error ? <X size={12} /> : <CheckCircle2 size={12} />}
        <span role={error ? "alert" : "status"}>
          {error || message || "PNG 또는 JPG 이미지를 불러오세요."}
        </span>
      </span>
      {busy && (
        <span className="sb-busy">
          <CatProgress
            compact
            decorative
            label={busy}
            fraction={props.progress.fraction}
            step={props.progress.step}
          />
        </span>
      )}
    </footer>
  );
}
