function drawBustIllustration(
  g: CanvasRenderingContext2D,
  ox: number,
  oy: number,
  scale: number,
) {
  g.save();
  g.translate(ox, oy);
  g.scale(scale, scale);
  const oval = (
    x: number,
    y: number,
    rx: number,
    ry: number,
    color: string,
  ) => {
    g.fillStyle = color;
    g.beginPath();
    g.ellipse(x, y, rx, ry, 0, 0, Math.PI * 2);
    g.fill();
  };
  oval(400, 425, 220, 330, "#566780");
  g.fillStyle = "#586e86";
  g.beginPath();
  g.moveTo(180, 960);
  g.quadraticCurveTo(145, 650, 300, 605);
  g.lineTo(500, 605);
  g.quadraticCurveTo(655, 650, 620, 960);
  g.fill();
  g.fillStyle = "#ebc7b9";
  g.fillRect(345, 430, 110, 215);
  oval(400, 610, 80, 45, "#ebc7b9");
  oval(400, 286, 154, 190, "#f4d7c8");
  oval(400, 305, 147, 165, "#f8dece");
  oval(313, 304, 41, 21, "#fff7eb");
  oval(487, 304, 41, 21, "#fff7eb");
  oval(313, 306, 17, 21, "#66898b");
  oval(487, 306, 17, 21, "#66898b");
  oval(313, 308, 8, 15, "#304950");
  oval(487, 308, 8, 15, "#304950");
  oval(308, 298, 5, 7, "#fff");
  oval(482, 298, 5, 7, "#fff");
  g.strokeStyle = "#435067";
  g.lineWidth = 6;
  g.lineCap = "round";
  for (const x of [313, 487]) {
    g.beginPath();
    g.moveTo(x - 38, 303);
    g.quadraticCurveTo(x, 278, x + 38, 303);
    g.stroke();
    g.lineWidth = 4;
    g.beginPath();
    g.moveTo(x - 29, 263);
    g.quadraticCurveTo(x, 251, x + 30, 262);
    g.stroke();
  }
  oval(400, 398, 26, 8, "#a96971");
  g.strokeStyle = "#cd9b8c";
  g.lineWidth = 3;
  g.beginPath();
  g.moveTo(400, 330);
  g.lineTo(395, 354);
  g.lineTo(404, 357);
  g.stroke();
  g.fillStyle = "#687e98";
  g.beginPath();
  g.moveTo(246, 300);
  g.bezierCurveTo(192, 107, 258, 70, 408, 80);
  g.bezierCurveTo(590, 70, 619, 207, 549, 301);
  g.quadraticCurveTo(531, 174, 440, 167);
  g.quadraticCurveTo(374, 250, 278, 250);
  g.lineTo(246, 390);
  g.closePath();
  g.fill();
  g.strokeStyle = "#8da1ba";
  g.lineWidth = 5;
  g.beginPath();
  g.moveTo(411, 99);
  g.quadraticCurveTo(298, 124, 268, 219);
  g.stroke();
  g.beginPath();
  g.moveTo(437, 101);
  g.quadraticCurveTo(548, 126, 559, 225);
  g.stroke();
  g.strokeStyle = "#a9c0c7";
  g.lineWidth = 4;
  g.beginPath();
  g.moveTo(252, 719);
  g.lineTo(548, 719);
  g.stroke();
  g.beginPath();
  g.moveTo(240, 734);
  g.lineTo(560, 734);
  g.stroke();
  g.restore();
}

export function demoFile() {
  const c = document.createElement("canvas");
  c.width = 800;
  c.height = 960;
  const g = c.getContext("2d")!;
  drawBustIllustration(g, 0, 0, 1);
  return new Promise<File>((resolve) =>
    c.toBlob((blob) =>
      resolve(new File([blob!], "character-study.png", { type: "image/png" })),
    ),
  );
}

/**
 * Renders a visual preparation sprite sheet explaining recommended input illustration layout
 * alongside the 21 motion-ready silhouette layers matched by flat2rig.
 */
export function renderSpriteGuideCanvas(width = 1860, height = 1170): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = width;
  c.height = height;
  const g = c.getContext("2d")!;
  const sx = width / 1240;
  const sy = height / 780;
  g.scale(sx, sy);

  g.fillStyle = "#f8fafd";
  g.fillRect(0, 0, 1240, 780);

  // Header strip
  g.fillStyle = "#0f172a";
  g.font = "700 24px system-ui, -apple-system, 'Malgun Gothic', sans-serif";
  g.fillText("flat2rig 일러스트 · LD / SD 실루엣 파츠 스프라이트 준비 가이드", 28, 42);
  g.fillStyle = "#475569";
  g.font = "500 14px system-ui, -apple-system, 'Malgun Gothic', sans-serif";
  g.fillText(
    "일반 등신(LD)과 SD 치비(2~3등신)를 자동 구분해 모션용 21레이어(가려진 눈동자·눈꺼풀·입안·뒷머리 포함)로 매칭합니다.",
    28,
    68,
  );

  // Left card: Recommended input illustration
  g.fillStyle = "#ffffff";
  g.strokeStyle = "#cbd5e1";
  g.lineWidth = 1.5;
  g.beginPath();
  g.roundRect(28, 90, 384, 662, 14);
  g.fill();
  g.stroke();

  g.fillStyle = "#1d4ed8";
  g.font = "700 15px system-ui, -apple-system, 'Malgun Gothic', sans-serif";
  g.fillText("1. 준비할 일러스트 (정면 흉상 PNG/JPG)", 48, 120);

  // Checker preview box inside left card
  g.fillStyle = "#f1f5f9";
  g.beginPath();
  g.roundRect(48, 136, 344, 412, 10);
  g.fill();
  g.save();
  g.beginPath();
  g.roundRect(48, 136, 344, 412, 10);
  g.clip();
  drawBustIllustration(g, 48, 136, 0.43);
  g.restore();

  // Landmark guides on left illustration
  const marks: [number, number, string][] = [
    [220, 170, "crown"],
    [182, 267, "eyeR"],
    [258, 267, "eyeL"],
    [220, 307, "mouth"],
    [220, 338, "chin"],
  ];
  for (const [mx, my, label] of marks) {
    g.fillStyle = "#0ea5e9";
    g.beginPath();
    g.arc(mx, my, 5, 0, Math.PI * 2);
    g.fill();
    g.strokeStyle = "#ffffff";
    g.lineWidth = 1.5;
    g.stroke();
    g.fillStyle = "#0f172a";
    g.font = "600 12px monospace";
    g.fillText(label, mx + 9, my + 4);
  }

  const tips = [
    "• 정면 흉상 · 수평 양눈 · 1,024~2,048px PNG/JPG",
    "• 일반 등신(LD) · SD 치비(2~3등신 큰 눈) 모두 지원",
    "• 양눈 뜬 정면 시선 (닫힌 눈꺼풀 · 입안 자동 보강)",
    "• 턱 아래 손·목·의상 넥라인이 겹치면 SD 선택 권장",
  ];
  g.fillStyle = "#1e293b";
  g.font = "500 13px system-ui, -apple-system, 'Malgun Gothic', sans-serif";
  tips.forEach((tip, idx) => {
    g.fillText(tip, 48, 582 + idx * 34);
  });

  // Right grid: 12 exploded motion-ready sprite cards
  g.fillStyle = "#ffffff";
  g.strokeStyle = "#cbd5e1";
  g.lineWidth = 1.5;
  g.beginPath();
  g.roundRect(432, 90, 780, 662, 14);
  g.fill();
  g.stroke();

  g.fillStyle = "#1d4ed8";
  g.font = "700 15px system-ui, -apple-system, 'Malgun Gothic', sans-serif";
  g.fillText("2. 실루엣 뱅크가 만들어내는 모션용 파츠 스프라이트 구성", 452, 120);

  const tiles: { title: string; sub: string; draw: (cx: number, cy: number) => void }[] = [
    {
      title: "face · 얼굴 바탕",
      sub: "구멍 없는 매끈 피부",
      draw: (cx, cy) => {
        g.fillStyle = "#f6dace";
        g.beginPath();
        g.ellipse(cx, cy + 4, 34, 42, 0, 0, Math.PI * 2);
        g.fill();
        g.strokeStyle = "#c99787";
        g.lineWidth = 1.5;
        g.stroke();
      },
    },
    {
      title: "eye_iris · 원형 동공",
      sub: "가려진 위쪽까지 복원",
      draw: (cx, cy) => {
        g.fillStyle = "#ffffff";
        g.strokeStyle = "#cbd5e1";
        g.lineWidth = 1.2;
        g.beginPath();
        g.ellipse(cx, cy + 4, 42, 22, 0, 0, Math.PI * 2);
        g.fill();
        g.stroke();
        g.fillStyle = "#5b77d4";
        g.beginPath();
        g.arc(cx, cy + 2, 18, 0, Math.PI * 2);
        g.fill();
        g.fillStyle = "#1e293b";
        g.beginPath();
        g.arc(cx, cy + 2, 7, 0, Math.PI * 2);
        g.fill();
        g.strokeStyle = "#f59e0b";
        g.setLineDash([4, 3]);
        g.beginPath();
        g.arc(cx, cy + 2, 19, Math.PI * 1.1, Math.PI * 1.9);
        g.stroke();
        g.setLineDash([]);
      },
    },
    {
      title: "eye_upper / lower",
      sub: "깜빡임 · 미소 눈꺼풀",
      draw: (cx, cy) => {
        g.fillStyle = "#f6dace";
        g.beginPath();
        g.ellipse(cx, cy - 8, 38, 16, 0, Math.PI, Math.PI * 2);
        g.fill();
        g.strokeStyle = "#1e293b";
        g.lineWidth = 5;
        g.lineCap = "round";
        g.beginPath();
        g.moveTo(cx - 38, cy - 6);
        g.quadraticCurveTo(cx, cy - 26, cx + 38, cy - 6);
        g.stroke();
        g.strokeStyle = "#475569";
        g.lineWidth = 2.5;
        g.beginPath();
        g.moveTo(cx - 28, cy + 16);
        g.quadraticCurveTo(cx, cy + 24, cx + 28, cy + 16);
        g.stroke();
      },
    },
    {
      title: "mouth_inner + lips",
      sub: "치아 · 혀 포함 입안",
      draw: (cx, cy) => {
        g.fillStyle = "#7b3040";
        g.beginPath();
        g.ellipse(cx, cy, 32, 18, 0, 0, Math.PI * 2);
        g.fill();
        g.fillStyle = "#fff8f8";
        g.fillRect(cx - 20, cy - 13, 40, 7);
        g.fillStyle = "#ee858f";
        g.beginPath();
        g.ellipse(cx, cy + 9, 20, 8, 0, 0, Math.PI * 2);
        g.fill();
      },
    },
    {
      title: "browR / L · 눈썹",
      sub: "앞머리 아래 연장",
      draw: (cx, cy) => {
        g.strokeStyle = "#4b5563";
        g.lineWidth = 4.5;
        g.lineCap = "round";
        for (const ox of [-26, 26]) {
          g.beginPath();
          g.moveTo(cx + ox - 16, cy + 4);
          g.quadraticCurveTo(cx + ox, cy - 8, cx + ox + 16, cy + 2);
          g.stroke();
        }
      },
    },
    {
      title: "neck · 목 (+턱그림자)",
      sub: "위아래 회전 여유 연장",
      draw: (cx, cy) => {
        g.fillStyle = "#ebc7b9";
        g.beginPath();
        g.roundRect(cx - 18, cy - 36, 36, 72, 10);
        g.fill();
        g.fillStyle = "#d5a798";
        g.beginPath();
        g.ellipse(cx, cy - 18, 17, 10, 0, 0, Math.PI);
        g.fill();
      },
    },
    {
      title: "body · 몸통 피부",
      sub: "의상 아래 쇄골 마네킹",
      draw: (cx, cy) => {
        g.fillStyle = "#f4d7c8";
        g.beginPath();
        g.moveTo(cx - 46, cy + 36);
        g.quadraticCurveTo(cx - 38, cy - 20, cx - 16, cy - 30);
        g.lineTo(cx + 16, cy - 30);
        g.quadraticCurveTo(cx + 38, cy - 20, cx + 46, cy + 36);
        g.closePath();
        g.fill();
      },
    },
    {
      title: "clothes · 의상",
      sub: "스웨터 · 세일러 · V넥",
      draw: (cx, cy) => {
        g.fillStyle = "#5c749e";
        g.beginPath();
        g.moveTo(cx - 48, cy + 36);
        g.quadraticCurveTo(cx - 40, cy - 22, cx - 14, cy - 28);
        g.lineTo(cx, cy - 12);
        g.lineTo(cx + 14, cy - 28);
        g.quadraticCurveTo(cx + 40, cy - 22, cx + 48, cy + 36);
        g.closePath();
        g.fill();
      },
    },
    {
      title: "hair_front · 앞머리",
      sub: "가르마 / 풀뱅 스타일",
      draw: (cx, cy) => {
        g.fillStyle = "#5b6d8a";
        g.beginPath();
        g.moveTo(cx - 42, cy + 20);
        g.quadraticCurveTo(cx - 38, cy - 36, cx, cy - 34);
        g.quadraticCurveTo(cx + 38, cy - 36, cx + 42, cy + 20);
        g.lineTo(cx + 12, cy + 8);
        g.lineTo(cx, cy - 8);
        g.lineTo(cx - 14, cy + 12);
        g.closePath();
        g.fill();
      },
    },
    {
      title: "hair_side_R/L",
      sub: "관자놀이 고정 옆머리",
      draw: (cx, cy) => {
        g.fillStyle = "#4d6180";
        for (const s of [-1, 1]) {
          g.beginPath();
          g.moveTo(cx + s * 22, cy - 34);
          g.quadraticCurveTo(cx + s * 42, cy, cx + s * 26, cy + 38);
          g.quadraticCurveTo(cx + s * 16, cy + 8, cx + s * 14, cy - 24);
          g.closePath();
          g.fill();
        }
      },
    },
    {
      title: "hair_back · 뒷머리",
      sub: "상체 뒤 풀실루엣",
      draw: (cx, cy) => {
        g.fillStyle = "#3f4f6b";
        g.beginPath();
        g.ellipse(cx, cy + 4, 48, 42, 0, 0, Math.PI * 2);
        g.fill();
      },
    },
    {
      title: "22 Live2D 파라미터",
      sub: "고개 · 눈 · 입 · 머리카락",
      draw: (cx, cy) => {
        g.fillStyle = "#e0ebff";
        g.strokeStyle = "#2456d6";
        g.lineWidth = 1.5;
        g.beginPath();
        g.roundRect(cx - 48, cy - 26, 96, 52, 10);
        g.fill();
        g.stroke();
        g.fillStyle = "#1d4ed8";
        g.font = "700 13px monospace";
        g.fillText("Angle X/Y/Z", cx - 42, cy - 4);
        g.fillText("Eye / Mouth", cx - 42, cy + 16);
      },
    },
  ];

  const cols = 4;
  const cardW = 178;
  const cardH = 190;
  const gapX = 14;
  const gapY = 14;
  const startX = 452;
  const startY = 138;

  tiles.forEach((tile, idx) => {
    const col = idx % cols;
    const row = Math.floor(idx / cols);
    const tx = startX + col * (cardW + gapX);
    const ty = startY + row * (cardH + gapY);

    g.fillStyle = "#f8fafc";
    g.strokeStyle = "#e2e8f0";
    g.lineWidth = 1.2;
    g.beginPath();
    g.roundRect(tx, ty, cardW, cardH, 10);
    g.fill();
    g.stroke();

    tile.draw(tx + cardW / 2, ty + 68);

    g.fillStyle = "#0f172a";
    g.font = "700 13px system-ui, -apple-system, 'Malgun Gothic', sans-serif";
    g.fillText(tile.title, tx + 10, ty + 150);
    g.fillStyle = "#475569";
    g.font = "500 12px system-ui, -apple-system, 'Malgun Gothic', sans-serif";
    g.fillText(tile.sub, tx + 10, ty + 172);
  });

  return c;
}

export function downloadSpriteGuide(): string {
  const url = renderSpriteGuideCanvas().toDataURL("image/png");
  const a = document.createElement("a");
  a.href = url;
  a.download = "flat2rig-sprite-guide.png";
  a.click();
  return url;
}
