import {
  Download,
  FileImage,
  Pause,
  Play,
  RotateCcw,
} from "lucide-react";
import type { Project } from "../types";
import { selectExpressionPreset, updatePlayback } from "../rig/state";

interface Props {
  project: Project;
  playing: boolean;
  busy: boolean;
  onTogglePlay: () => void;
  onResetToNeutral: () => void;
  onCommit: (next: Project, message?: string) => void;
  onBeginDrag: () => void;
  onPreviewDrag: (next: Project) => void;
  onCommitDrag: () => void;
  onExportPng: () => void;
  onOpenAdvancedExport: () => void;
}

export default function PlaybackPanel({
  project,
  playing,
  busy,
  onTogglePlay,
  onResetToNeutral,
  onCommit,
  onBeginDrag,
  onPreviewDrag,
  onCommitDrag,
  onExportPng,
  onOpenAdvancedExport,
}: Props) {
  const pb = project.playback;

  return (
    <>
      <section>
        <div className="section-title">
          <span>01</span>
          <h2>재생 제어</h2>
        </div>
        <button
          className="primary full"
          disabled={busy}
          onClick={onTogglePlay}
        >
          {playing ? <Pause size={15} /> : <Play size={15} />}
          {playing ? "일시정지 (Space)" : "라이브 재생 시작 (Space)"}
        </button>
        <button
          className="full"
          style={{ marginTop: 8 }}
          disabled={busy}
          onClick={onResetToNeutral}
        >
          <RotateCcw size={14} />
          기본 상태로 복귀 (1)
        </button>
      </section>

      <section>
        <div className="section-title">
          <span>02</span>
          <h2>수동 표정 전환 (1~9 키)</h2>
        </div>
        <div className="candidate-pills" style={{ gap: 6 }}>
          {project.expressions.map((exp, idx) => {
            const shortcut =
              exp.shortcutKey ?? (idx < 9 ? String(idx + 1) : undefined);
            const active = exp.id === project.activeExpressionId;
            return (
              <button
                key={exp.id}
                type="button"
                className={`compact-pill${active ? " active" : ""}`}
                disabled={busy}
                onClick={() =>
                  onCommit(
                    selectExpressionPreset(project, exp.id),
                    `표정 '${exp.name}' 전환`,
                  )
                }
              >
                {shortcut ? `${shortcut} · ` : ""}
                {exp.name}
              </button>
            );
          })}
        </div>
      </section>

      <section>
        <div className="section-title">
          <span>03</span>
          <h2>자동 동작 설정</h2>
        </div>
        <div className="check-options">
          <label>
            <input
              type="checkbox"
              checked={pb.autoBlink}
              disabled={busy}
              onChange={(e) =>
                onCommit(
                  updatePlayback(project, { autoBlink: e.target.checked }),
                )
              }
            />
            자동 눈 깜빡임 (감김·윙크 표정 우선)
          </label>
          <label>
            <input
              type="checkbox"
              checked={pb.autoBreath}
              disabled={busy}
              onChange={(e) =>
                onCommit(
                  updatePlayback(project, { autoBreath: e.target.checked }),
                )
              }
            />
            자동 호흡 (상체 미세 상하)
          </label>
          <label>
            <input
              type="checkbox"
              checked={pb.autoIdle}
              disabled={busy}
              onChange={(e) =>
                onCommit(
                  updatePlayback(project, { autoIdle: e.target.checked }),
                )
              }
            />
            유휴 고개·시선 흔들림
          </label>
          <label>
            <input
              type="checkbox"
              checked={pb.autoMouthDemo === true}
              disabled={busy}
              onChange={(e) =>
                onCommit(
                  updatePlayback(project, { autoMouthDemo: e.target.checked }),
                )
              }
            />
            테스트용 자동 입 개폐 (데모)
          </label>
        </div>

        <div className="param-row" style={{ marginTop: 10 }}>
          <div className="param-head">
            <label htmlFor="blink-interval">깜빡임 주기</label>
            <span className="param-val">{pb.blinkIntervalSec.toFixed(1)}초</span>
          </div>
          <input
            id="blink-interval"
            aria-label="깜빡임 주기"
            type="range"
            min="1.5"
            max="8.0"
            step="0.3"
            disabled={busy || !pb.autoBlink}
            value={pb.blinkIntervalSec}
            onPointerDown={onBeginDrag}
            onChange={(e) => {
              onBeginDrag();
              onPreviewDrag(
                updatePlayback(project, { blinkIntervalSec: +e.target.value }),
              );
            }}
            onPointerUp={onCommitDrag}
            onBlur={onCommitDrag}
          />
        </div>
      </section>

      <section className="handoff-card">
        <div className="section-title">
          <span>04</span>
          <h2>현재 프레임 출력</h2>
        </div>
        <button
          className="primary full"
          disabled={busy}
          onClick={onExportPng}
        >
          <Download size={15} />
          투명 배경 PNG 저장
        </button>
        <p className="muted tiny" style={{ marginTop: 8 }}>
          핸들·메쉬·안내선 없이 현재 표정·포즈 상태를 투명 RGBA PNG로 저장합니다.
        </p>
        <button
          className="full"
          style={{ marginTop: 8 }}
          disabled={busy}
          onClick={onOpenAdvancedExport}
        >
          <FileImage size={14} />
          고급 · Cubism용 PSD 내보내기
        </button>
        <p className="muted tiny" style={{ marginTop: 6 }}>
          참고: 기본 PSD 내보내기는 원화 기반 기본 파츠를 출력하며, 앱에서 만든
          대체 모양·표정은 Cubism으로 자동 이전되지 않습니다.
        </p>
      </section>
    </>
  );
}
