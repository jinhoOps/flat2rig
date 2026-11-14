import React, { lazy, Suspense } from "react";
import ReactDOM from "react-dom/client";
import BrandLoop from "./BrandLoop";
import { BrandMark, BrandWordmark } from "./Brand";
import "../design.css";
import "./preview.css";

const gold = new URLSearchParams(location.search).get("capture") === "gold";
const Gold = lazy(() => import("../../tools/stageshow/examples/tilted-stack"));

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    {gold ? <div id="example" className="gold-preview"><Suspense fallback={null}><Gold /></Suspense></div> : <main className="brand-preview">
      <header><a href="/" aria-label="flat2rig 편집기 열기"><BrandMark /><BrandWordmark /></a><span>브랜드 모션</span></header>
      <section className="brand-preview-feature" aria-label="로고 모션 미리보기"><BrandLoop /></section>
      <p className="brand-preview-caption">원화 레이어 · 파츠 분리 · 리깅 피벗</p>
      <div className="brand-assets">
        <a href="/brand/mark.svg" download>심볼 SVG</a>
        <a href="/logo.svg" download>앱 아이콘 SVG</a>
      </div>
    </main>}
  </React.StrictMode>,
);
