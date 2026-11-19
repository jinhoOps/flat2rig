import { AlertTriangle, Check, FolderOpen, Play, RotateCcw } from "lucide-react";
import {
  SEE_THROUGH_DEFAULT_SEED,
  SEE_THROUGH_INSTALL_GUIDE,
  SEE_THROUGH_LICENSE_NOTE,
  SEE_THROUGH_LIMITATIONS,
  type SeeThroughInstallStatus,
} from "../seethrough";

export interface SeeThroughModalProps {
  desktop: boolean;
  busy: boolean;
  installDir: string;
  seed: number;
  status: SeeThroughInstallStatus | null;
  error: string | null;
  onChangeInstallDir: (dir: string) => void;
  onChangeSeed: (seed: number) => void;
  onBrowseDir: () => void;
  onVerify: () => void;
  onClear: () => void;
  onRunDecompose: () => void;
  onClose: () => void;
}

export default function SeeThroughModal({
  desktop,
  busy,
  installDir,
  seed,
  status,
  error,
  onChangeInstallDir,
  onChangeSeed,
  onBrowseDir,
  onVerify,
  onClear,
  onRunDecompose,
  onClose,
}: SeeThroughModalProps) {
  const ready = desktop && Boolean(status?.installed);

  return (
    <div className="st-modal">
      <p className="st-desc">
        사용자가 로컬에 설치한 <strong>See-through(SIGGRAPH 2026)</strong> 외부 CLI를 실행해 단일 일러스트를
        레이어 PSD로 분해한 뒤 매핑 화면으로 엽니다. 모델 가중치는 앱에 번들하지 않습니다.
      </p>

      <div className="st-field">
        <label htmlFor="st-install-dir">See-through 설치 폴더</label>
        <div className="st-row">
          <input
            id="st-install-dir"
            type="text"
            value={installDir}
            placeholder="예: C:\tools\seethrough"
            disabled={busy}
            onChange={(e) => onChangeInstallDir(e.target.value)}
          />
          {desktop && (
            <button type="button" disabled={busy} onClick={onBrowseDir}>
              <FolderOpen size={13} />
              폴더 선택…
            </button>
          )}
          <button type="button" disabled={busy || !installDir.trim()} onClick={onVerify}>
            경로 확인
          </button>
          {installDir.trim() && (
            <button type="button" disabled={busy} title="설정 초기화" onClick={onClear}>
              <RotateCcw size={13} />
            </button>
          )}
        </div>
      </div>

      <div className="st-field st-seed">
        <label htmlFor="st-seed">시드 (기본 {SEE_THROUGH_DEFAULT_SEED})</label>
        <input
          id="st-seed"
          type="number"
          min={0}
          max={4294967295}
          value={seed}
          disabled={busy}
          onChange={(e) => {
            const v = Number(e.target.value);
            if (Number.isFinite(v) && v >= 0) onChangeSeed(Math.trunc(v));
          }}
        />
      </div>

      {error && (
        <div className="st-status is-error" role="alert">
          <AlertTriangle size={14} />
          <span>{error}</span>
        </div>
      )}

      {status && (
        <div className={`st-status ${status.installed ? "is-ok" : "is-warn"}`}>
          {status.installed ? <Check size={14} /> : <AlertTriangle size={14} />}
          <div>
            <strong>{status.message}</strong>
            {status.pythonPath && <small>Python: {status.pythonPath}</small>}
            {status.scriptPath && <small>스크립트: {status.scriptPath}</small>}
            {status.hfHome && <small>HF 캐시: {status.hfHome}</small>}
          </div>
        </div>
      )}

      {!status?.installed && (
        <div className="st-guide">
          <strong>외부 설치 안내 (선택 기능)</strong>
          <ul>
            {SEE_THROUGH_INSTALL_GUIDE.map((step, i) => (
              <li key={i}>{step}</li>
            ))}
          </ul>
          {!desktop && (
            <p className="st-browser-note">
              브라우저 미리보기에서는 외부 프로세스를 실행할 수 없습니다. Tauri 데스크톱 앱에서 설정·실행할 수 있습니다.
            </p>
          )}
        </div>
      )}

      <div className="st-warnings">
        <p className="st-license">
          <strong>라이선스:</strong> {SEE_THROUGH_LICENSE_NOTE}
        </p>
        <ul>
          {SEE_THROUGH_LIMITATIONS.map((lim) => (
            <li key={lim.id}>
              <strong>{lim.title}</strong>: {lim.detail}
            </li>
          ))}
        </ul>
      </div>

      <div className="modal-actions">
        <button type="button" disabled={busy} onClick={onClose}>
          닫기
        </button>
        <button
          type="button"
          className="primary"
          data-action="seethrough-decompose"
          disabled={busy || !ready}
          onClick={onRunDecompose}
        >
          <Play size={13} />
          이미지 선택 후 분해 실행…
        </button>
      </div>
    </div>
  );
}
