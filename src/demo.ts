export function demoFile() {
  const c = document.createElement("canvas");
  c.width = 800;
  c.height = 960;
  const g = c.getContext("2d")!;
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
  // Procedural vector study used only to exercise tools without downloading assets.
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
  return new Promise<File>((resolve) =>
    c.toBlob((blob) =>
      resolve(new File([blob!], "character-study.png", { type: "image/png" })),
    ),
  );
}
