# flat2rig Design System — Daylight Acrylic Glass (`DESIGN.md`)

> **적용 대상**: `flat2rig` (Live2D Cubism 준비 스튜디오 · Tauri v2 + React + TypeScript)  
> **디자인 콘셉트**: **Daylight Drafting Acrylic** — 기존 다크·민트 네온 계열을 완전히 배제하고, 밝은 제도실 데스크 위의 반투명 아크릴 판과 제도용 코발트·그라파이트 잉크를 기준으로 설계한 라이트 글래스모피즘 디자인 시스템.

---

## 1. 디자인 방향과 탈(脫) AI 슬롭 원칙

크리에이티브 툴의 유리는 "빛나는 사이버 패널"이 아니라 **종이와 원화 위에 올려놓은 무색·반투명 아크릴 자(Drafting Acrylic)**처럼 동작해야 한다.

1. **네온 · 오로라 · 글로우 전면 배제**
   - 배경에 보라·청록 오로라 블럽, 버튼·배지의 발광 그림자(`0 0 20px`), 형광 LED 도트, 대각선 그라데이션 카드를 사용하지 않는다.
   - 깊이와 재질감은 오직 **무채색 채광 배경의 부드러운 명암 차**, **1px 헤어라인 경계(`rgba(15, 23, 42, 0.08~0.11)`)**, **상단 유리면 캐치라이트(`inset 0 1px 0 rgba(255, 255, 255, 0.85~0.96)`)**로만 만든다.
2. **원화 색채 정확도 보호**
   - `.canvas-frame` 내부의 체커보드(`#FFFFFF` / `#ECEEF2`)와 일러스트 영역에는 `backdrop-filter`, 색상 틴트, 반투명을 절대 적용하지 않는다.
   - 유리 패널의 `saturate()`는 `110%~120%`로 낮게 유지해, 캔버스 경계에 닿은 유리가 원화 색을 과장해 번지게 하지 않는다.
3. **절제된 시그널 컬러**
   - 주 포커스/선택/도구 컬러는 **Studio Cobalt(`#2456D6`)**, 완료/승인 상태는 **Botanical Sage(`#18794E`)**, 경고는 **Drafting Ochre(`#9A6700`)**, 오류는 **Studio Crimson(`#C9372C`)**으로 역할을 엄격히 분리한다.
   - 번호(`01`, `02`, `03`)나 메타 라벨(`.eyebrow`, `.stage-tag`)에 테두리 박스를 치지 않고 조용한 보조 텍스트로 처리한다.

---

## 2. 3계층 라이트 아크릴 표면 모델

| 계층 | 역할 및 대표 요소 | 표면 배경 & 블러 | 경계 & 입체감 |
| --- | --- | --- | --- |
| **Layer 0: Daylight Desk** | `.app` 루트 작업대 배경 | `linear-gradient(165deg, #F6F7FA 0%, #ECEEF3 55%, #E3E7EE 100%)` | 무채색 자연 채광 그라데이션 (색상 오라 없음) |
| **Layer 1: Frosted Chrome** | `.topbar`, `.layers-panel`, `.inspector`, `.editor-status` | `rgba(255, 255, 255, 0.72)`<br>`blur(20px) saturate(115%)` | `1px solid rgba(15, 23, 42, 0.08)`<br>`inset 0 1px 0 rgba(255, 255, 255, 0.85)` |
| **Layer 1B: Sub Chrome** | `.editor-tabs`, `.toolbar` | `rgba(255, 255, 255, 0.56)`<br>`blur(16px) saturate(110%)` | `.topbar`보다 한 톤 얇은 유리로 중앙 작업 공간 위계 구분 |
| **Layer 2: Floating HUD & Cards** | `.canvas-caption > span`, `.stage-bottom`, `.welcome-drop-zone`, `.export-choice`, `.drop-card` | `rgba(255, 255, 255, 0.84)`<br>`blur(22px) saturate(120%)` | `1px solid rgba(15, 23, 42, 0.10)`<br>`0 8px 24px rgba(15, 23, 42, 0.07)` |
| **Layer 3: Modal & Overlay** | `.modal`, `.busy-overlay > div` | `rgba(255, 255, 255, 0.90)`<br>`blur(28px) saturate(120%)` | 스크림 `rgba(15, 23, 42, 0.24) + blur(8px)`<br>`0 24px 64px rgba(15, 23, 42, 0.14)` |

---

## 3. 색상 · 잉크 · 상태 토큰

### 3.1 텍스트 (Graphite Ink Hierarchy)
- `--ink-primary`: `#161920` (제목, 활성 탭, 주요 버튼·레이어 이름 — 순수 블랙 `#000` 대신 부드러운 먹색)
- `--ink-secondary`: `#475060` (일반 라벨, 설명문, 비활성 툴 아이콘)
- `--ink-muted`: `#747E8F` (보조 힌트, 섹션 번호, 단축키, 영문 메타 라벨)
- `--ink-subtle`: `#98A2B3` (비활성 안내, 얇은 아이콘)

### 3.2 액센트 및 상태 색상
- **Primary Accent (Studio Cobalt)**
  - `--accent`: `#2456D6` / 호버 `--accent-hover`: `#1B47BA` / 버튼 텍스트 `#FFFFFF`
  - `--accent-tint`: `rgba(36, 86, 214, 0.08)` / `--accent-border`: `rgba(36, 86, 214, 0.26)` / 텍스트 `--accent-text`: `#1D47B8`
- **Approved (Botanical Sage)**
  - `--ok`: `#18794E` / `--ok-tint`: `rgba(24, 121, 78, 0.10)` / `--ok-border`: `rgba(24, 121, 78, 0.28)`
- **Draft / Warning (Drafting Ochre)**
  - `--warn`: `#9A6700` / `--warn-tint`: `rgba(191, 135, 0, 0.11)` / `--warn-border`: `rgba(191, 135, 0, 0.28)`
- **Error (Studio Crimson)**
  - `--error`: `#C9372C`

### 3.3 18개 파츠 컬러(`PART_DEFS`) 라이트 모드 보정
밝은 유리 위에서 연한 파스텔 파츠 색(`#f3b48c`, `#b3d5ff` 등)이 흐려지지 않도록 `.layer-thumb` 칩은 배경과 아이콘 명도를 분리한다.
- **배경**: `color-mix(in srgb, var(--part-color) 18%, rgba(255, 255, 255, 0.88))`
- **아이콘 전경색**: `color-mix(in srgb, var(--part-color) 62%, #161920)`
- **테두리**: `color-mix(in srgb, var(--part-color) 32%, rgba(15, 23, 42, 0.08))`

---

## 4. 타이포그래피 & 레이아웃 원칙

- **로컬 폰트 스택** (`src-tauri/tauri.conf.json` CSP `font-src 'self' data:` 준수):
  - Sans: `'Pretendard Variable', 'Pretendard', 'Inter', 'Segoe UI', 'Malgun Gothic', sans-serif`
  - Mono: `'JetBrains Mono', 'Cascadia Mono', 'Consolas', monospace` (`tabular-nums`)
- **레이아웃 지오메트리**: [src/style.css](file:///C:/Users/kjh/orca/flat2rig/src/style.css)의 3단 워크벤치(`64px` 헤더 + `240px / 1fr / 292px` 작업 공간 + `32px` 푸터 및 반응형 브레이크포인트)를 유지한다.

---

## 5. 컴포넌트별 명세

### 5.1 상단 바 (`.topbar`)
- 반투명 라이트 아크릴(`Layer 1`), 하단 헤어라인 경계.
- `.brand-mark`: 머리와 어깨의 단순한 사람 윤곽에 삼각 메쉬와 제어점 네 개를 배치한다. 눈·입·머리카락은 생략한다. 윤곽은 Graphite `#161920`, 메쉬는 Studio Cobalt `#2456D6`. `.brand-two`도 `--accent`를 사용한다. 앱 아이콘은 같은 도형을 밝은 타일 위에 배치한다.
- `.local-status`: 얇은 반투명 흰색 캡슐(`rgba(255, 255, 255, 0.65)`), 내부 인디케이터 점(`i`)은 단색 `--ok`(발광 없음).
- `.primary` (`내보내기`): 단색 `#2456D6`, 흰색 텍스트, 상단 `1px` 내부 하이라이트, 미세한 입체 그림자만 사용.

### 5.2 좌측 파츠 패널 (`.layers-panel`)
- `.layer`: 기본 투명, 호버 시 `rgba(15, 23, 42, 0.045)`.
- `.layer.selected`: 연한 코발트 아크릴 틴트(`rgba(36, 86, 214, 0.09)`), 테두리 `rgba(36, 86, 214, 0.24)`(좌측 네온 바 없음).
- `.progress-track`: `rgba(15, 23, 42, 0.08)` 트랙 위 단색 `--ok` 채움 바.

### 5.3 중앙 에디터 · 캔버스 · 웰컴 (`.editor`, `.stage`, `.welcome`)
- **`.editor-tabs`**: 활성 탭 텍스트 `--ink-primary`(`font-weight: 600`), 하단 `2px` `--accent` 인디케이터(글로우 없음).
- **`.toolbar`**: 활성 도구(`.icon-button.active`)는 `--accent-tint` + `--accent-border` + `--accent-text`.
- **`.stage`**: 배경은 투명(루트 채광 그라데이션 투과), 아주 옅은 제도판 도트(`rgba(15, 23, 42, 0.065)` / `20px`).
- **`.canvas-frame`**: 불투명 라이트 체커보드(`#FFFFFF` / `#ECEEF2`), `1px` 테두리(`rgba(15, 23, 42, 0.14)`), 코너 조준선(`.corner`)은 `rgba(15, 23, 42, 0.35)`.
- **`.canvas-caption`, `.stage-bottom`**: `Layer 2` 플로팅 프로스티드 캡슐로 캔버스 경계 위에서도 선명한 가독성 유지.
- **`.welcome-drop-zone`**: 점선(`rgba(15, 23, 42, 0.16)`)을 두른 반투명 화이트 아크릴 카드, `.welcome-icon`은 깨끗한 `#FFFFFF` 입체 타일(`border-radius: 18px`), `.welcome-features`는 과한 테두리 배지 대신 은은한 메타 필로 정돈.

### 5.4 우측 인스펙터 & 모달 (`.inspector`, `.modal`)
- **`.section-title > span`**: 박스·테두리 없는 담백한 모노스페이스 인덱스(`01`, `02`, `03`).
- **`input[type=range]`**: 옅은 회색 트랙(`rgba(15, 23, 42, 0.12)`) 위 흰색 원형 노브 + `2px` 코발트 링(`#2456D6`).
- **`.model-button`**: 밝은 반투명 점선 슬롯, 호버 시 코발트 점선 + 옅은 코발트 틴트.
- **`.approve`**: 대기 시 `--ok-tint` + `--ok` 텍스트, 완료(`.is-approved`) 시 단색 `--ok` 배경 + 흰색 텍스트.
- **`.handoff-card`**: 그라데이션을 제거하고 상단 구분선과 미묘한 밝은 배경(`rgba(255, 255, 255, 0.42)`)만 둔다.
- **`.modal`**: 스크림(`rgba(15, 23, 42, 0.24) + blur(8px)`) 위의 깔끔한 화이트 아크릴 다이얼로그(`rgba(255, 255, 255, 0.90) + blur(28px)`). `.export-choice`는 호버 시 순백(`#FFFFFF`) + 코발트 테두리로 명확하게 떠오른다.

---

## 6. 캔버스 오버레이 & 성능 · 접근성 기준

- **모달 블러 제한**: `.app:has(.modal-backdrop)` 상태에서는 배경 도킹 패널 및 스테이지 HUD의 `backdrop-filter`를 잠시 해제해 스크림+모달 2겹만 합성한다.
- **폴백 필수**:
  - `@media (prefers-reduced-transparency: reduce)`: 불투명 라이트 표면(`#F4F6F9`, `#FFFFFF`, `#EAECEF`)으로 치환, `backdrop-filter: none`.
  - `@supports not (backdrop-filter: blur(1px))`: 알파 `0.95` 이상으로 상향.
  - `@media (prefers-reduced-motion: reduce)`: 트랜지션 제거.
- **향후 `Viewport.tsx` 캔버스 스트로크 권장값 (라이트 체커보드 조화용)**:
  - 선택 메쉬 와이어 `#2456D6`, 비선택 와이어 `rgba(22, 25, 32, 0.30)`
  - 브러시 커서 추가 `#2456D6` / 지우기 `#C9372C`
  - 랜드마크 핀 `#FFFFFF` 채움 + `#2456D6` 스트로크, 라벨 `#161920` + 흰색 외곽 헤일로

---

## 7. 기준 CSS 변수 (`src/design.css`)

```css
:root {
  color-scheme: light;
  --font-sans: 'Pretendard Variable', 'Pretendard', 'Inter', 'Segoe UI', 'Malgun Gothic', sans-serif;
  --font-mono: 'JetBrains Mono', 'Cascadia Mono', 'Consolas', monospace;

  --bg-desk: #eceef3;
  --bg-desk-gradient: linear-gradient(165deg, #f6f7fa 0%, #eceef3 55%, #e3e7ee 100%);

  --glass-t1: rgba(255, 255, 255, 0.72);
  --glass-t1-sub: rgba(255, 255, 255, 0.56);
  --glass-t2: rgba(255, 255, 255, 0.84);
  --glass-t3: rgba(255, 255, 255, 0.90);
  --glass-scrim: rgba(15, 23, 42, 0.24);
  --glass-input: rgba(255, 255, 255, 0.80);

  --blur-t1: blur(20px) saturate(115%);
  --blur-t1-sub: blur(16px) saturate(110%);
  --blur-t2: blur(22px) saturate(120%);
  --blur-t3: blur(28px) saturate(120%);
  --blur-scrim: blur(8px);

  --line-subtle: rgba(15, 23, 42, 0.08);
  --line-control: rgba(15, 23, 42, 0.12);
  --line-glass: rgba(15, 23, 42, 0.10);
  --rim-top: inset 0 1px 0 rgba(255, 255, 255, 0.86);
  --rim-crystal: inset 0 1px 0 rgba(255, 255, 255, 0.96);
  --shadow-hud: 0 8px 24px rgba(15, 23, 42, 0.07), 0 1px 2px rgba(15, 23, 42, 0.04);
  --shadow-modal: 0 24px 64px rgba(15, 23, 42, 0.14), 0 4px 16px rgba(15, 23, 42, 0.06);

  --ink-primary: #161920;
  --ink-secondary: #475060;
  --ink-muted: #747e8f;
  --ink-subtle: #98a2b3;

  --accent: #2456d6;
  --accent-hover: #1b47ba;
  --accent-text: #1d47b8;
  --accent-tint: rgba(36, 86, 214, 0.08);
  --accent-border: rgba(36, 86, 214, 0.26);

  --ok: #18794e;
  --ok-tint: rgba(24, 121, 78, 0.10);
  --ok-border: rgba(24, 121, 78, 0.28);

  --warn: #9a6700;
  --warn-tint: rgba(191, 135, 0, 0.11);
  --warn-border: rgba(191, 135, 0, 0.28);

  --error: #c9372c;
}
```
