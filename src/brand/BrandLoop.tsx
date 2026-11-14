import { useState } from "react";
import { Play, Square } from "lucide-react";
import { Stage } from "../../vendor/stageshow/kit/stage";
import type { Act } from "../../vendor/stageshow/kit/act";
import { RigGeometry } from "./Brand";
import "./brand.css";

type Scene = { turned: boolean };
const INITIAL: Scene = { turned: false };
const POSTER: Scene = { turned: false };

// Adapted from Stageshow's tilted-stack composition: one cell, retained layers.
// The silhouette stays still while its retained mesh gently changes pose.
async function sequence(a: Act<Scene>) {
  a.mark("rest");
  await a.wait(1200);
  a.mark("rig");
  await a.set({ turned: true });
  await a.wait(4000);
  a.mark("return");
  await a.set({ turned: false });
  await a.wait(5600);
}

export default function BrandLoop({ controls = true }: { controls?: boolean }) {
  const [stopped, setStopped] = useState(false);
  return (
    <div className="brand-loop" data-stopped={stopped || undefined}>
      <div
        role="img"
        aria-label="단순한 사람 윤곽 안에서 삼각 메쉬가 살짝 움직이는 flat2rig 로고"
      >
        <Stage<Scene>
          width={400}
          height={220}
          compact={{ below: 400, width: 358, height: 200 }}
          initial={INITIAL}
          poster={POSTER}
          script={stopped ? undefined : sequence}
          play="loop"
          theme="light"
          greys="cool"
          accent="#2456d6"
          className="flat2rig-motion"
          tokens={{
            "--ss-font": "var(--font-sans, 'Segoe UI', sans-serif)",
            "--ss-text-strong": "#161920",
            "--ss-well": "transparent",
          }}
        >
          {(s) => (
            <div
              className="rig-scene"
              data-turned={s.turned || undefined}
            >
              <RigGeometry />
            </div>
          )}
        </Stage>
      </div>
      {controls && (
        <button
          className="brand-motion-control"
          type="button"
          onClick={() => setStopped((v) => !v)}
          aria-label={
            stopped ? "로고 모션 다시 재생" : "로고 모션 정지 화면 보기"
          }
        >
          {stopped ? <Play size={12} /> : <Square size={12} />}
          <span>{stopped ? "모션 보기" : "정지 화면"}</span>
        </button>
      )}
    </div>
  );
}
