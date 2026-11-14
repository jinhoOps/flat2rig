import { useState } from "react";
import { Play, Square } from "lucide-react";
import { Stage } from "../../vendor/stageshow/kit/stage";
import type { Act } from "../../vendor/stageshow/kit/act";
import { RigGeometry } from "./Brand";
import "./brand.css";

type Scene = { separated: boolean; mesh: boolean; turned: boolean };
const INITIAL: Scene = { separated: false, mesh: false, turned: false };
const POSTER: Scene = { separated: true, mesh: true, turned: true };

// Adapted from Stageshow's tilted-stack composition: one cell, retained layers.
// The return is the logo's own mechanical motion, never fabricated product data.
async function sequence(a: Act<Scene>) {
  a.mark("flat");
  await a.wait(900);
  a.mark("separate");
  await a.set({ separated: true });
  await a.wait(1500);
  a.mark("mesh");
  await a.set({ mesh: true });
  await a.wait(1500);
  a.mark("rig");
  await a.set({ turned: true });
  await a.wait(2600);
  a.mark("return");
  await a.set({ turned: false, mesh: false });
  await a.wait(1400);
  a.mark("align");
  await a.set({ separated: false });
  await a.wait(2900);
}

export default function BrandLoop({ controls = true }: { controls?: boolean }) {
  const [stopped, setStopped] = useState(false);
  return (
    <div className="brand-loop" data-stopped={stopped || undefined}>
      <div role="img" aria-label="원화 레이어가 분리되고 메쉬와 피벗을 따라 회전한 뒤 다시 정렬되는 flat2rig 로고">
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
          tokens={{ "--ss-font": "var(--font-sans, 'Segoe UI', sans-serif)", "--ss-text-strong": "#161920", "--ss-well": "transparent" }}
        >
          {(s) => (
            <div className="rig-scene" data-separated={s.separated || undefined} data-turned={s.turned || undefined}>
              <RigGeometry mesh={s.mesh} />
            </div>
          )}
        </Stage>
      </div>
      {controls && <button className="brand-motion-control" type="button" onClick={() => setStopped((v) => !v)} aria-label={stopped ? "로고 모션 다시 재생" : "로고 모션 정지 화면 보기"}>
        {stopped ? <Play size={12} /> : <Square size={12} />}
        <span>{stopped ? "모션 보기" : "정지 화면"}</span>
      </button>}
    </div>
  );
}
