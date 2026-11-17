# 로컬 검증 기록

2026-11-15 · Windows 11 (x64) · Node 24.22.3 · Rust 1.99.1 · WebView2

- `npm test`: 18개 통과. 마스크 분할·편집·의상/몸통 정적 유지·가변 다중 머리카락(Z-순서 및 가중치 연동)·외곽 어두운 선 침식 기반 가림 보완(`fillMask` / `fillKind`), 윤곽/랜드마크 인식 Delaunay 메쉬 및 12개 극단 포즈 품질 검사, 프로젝트 v1/v2 왕복, 손상 파일 거절, 실제 RGB PSD 바이트/알파/오프셋, ZIP 산출물(`validation.json` 포함), CSV 형식 검증.
- `npx tsc --noEmit` 및 `npm run build`: 오류 없이 통과 (`vite build`).
- `cargo test --manifest-path src-tauri/Cargo.toml`: 16개 통과 (행렬 왕복 3종, 매니페스트 5종 역할 및 ONNX 로드 2종, 코어 분리·메쉬·침식 가림 보완·머리카락 흔들림 프로필·품질 9종, 실제 ONNX YOLOv3·HRNetV2·MobileSAM·U2NetP 정합성·프롬프트 캐시 재추론 2종).
- `cargo clippy --manifest-path src-tauri/Cargo.toml --all-targets -- -D warnings`: 경고/오류 0건 통과.
- `npm run desktop:build -- --debug`: `src-tauri/target/debug/flat2rig.exe` 생성.
- `npm run tauri -- build --bundles msi`: `src-tauri/target/release/bundle/msi/flat2rig_0.1.0_x64_en-US.msi` 생성 (`26,292,224` bytes, 약 `25.07 MiB`, WiX 3.14, SHA-256 `0AB3750F425505B0C2B9A8B0189BCA83EEBA4E803E69954E6BC394614A7CDC01`). `src-tauri/target/release/wix/x64/main.wxs`에 `flat2rig.exe`, `flat2rig-cli.exe`, `DirectML.dll` 포함 확인 (`models/manifest.json`은 바이너리에 내장되며 `.onnx` 모델 파일은 미포함). 설치 대상은 64비트 Windows(`x64` / `x86_64`)이며 32비트(`i686`) 빌드는 포함하지 않습니다.
- `npm run test:native`: debug 및 release(`FLAT2RIG_TEST_BINARY=src-tauri/target/release/flat2rig.exe`) 실행 파일의 실제 WebView2에서 이미지 열기·드래그 앤 드롭·잘못된 파일 거절, Rust 18파츠 생성(`mobilesam-vit-t+landmark-partition-v1`, 128×128 픽스처), `propose_parts` 부모 통합 모드(10파츠, `eyeR_all`), `build_rig` 12극단 포즈 품질 리포트(`586` 정점), `fill_part_occlusion` 분리 보완(`neighbor`, `16,384` bytes)·숨은 피처 경고(`occWarning: true`), `get_model_status`(5모델), 투명 효과 축소 폴백 검증 완료. JS 예외 없음(`exceptions: []`).
- `tools/evaluation/evaluate.py` (`2`개 로컬 일러스트 + `4`개 합성 검증, 외부 20장 홀드아웃 미포함):
  - 정합 오차: YOLOv3 `bboxMaxDiffPx` `0.0428–0.1236px`(로컬 2종 `0.0428–0.049px`), HRNet 28점 평균 오차 `0.0466–0.1472px`(최대 `0.5419px`, 로컬 2종 최대 `0.2154–0.2561px`), MobileSAM `samFacePredIoU` `0.9229–0.9838`, Rust 파츠 중복 픽셀 `0`, `pythonProcessPeakRssMb` `1886.22 MB`.
  - 릴리스 CLI P50 지연: YOLOv3 `934.61 ms`, HRNet `331.22 ms`, MobileSAM `2733.8 ms`, 전체 CLI `4062.35 ms`.
  - 메쉬 극단 포즈 검사: `sutteang_2d`(`2,426`개 삼각형) 및 `sutteang_cutout`(`2,434`개 삼각형) 모두 경고 `0`건. 합성 변환·다중 얼굴 케이스(`nonsquare_portrait`, `letterbox_wide`, `multi_face_pair`)에서는 극단 포즈 뒤집힘(`neck` 1–3건) 및 소형 파츠 유효 삼각형 미생성(`clothes`, `hair_side_L`, `hair_front`) 경고 검출 확인, `negative_blank`는 얼굴 `0`개 판별.
- `flat2rig-cli.exe sutteang/2d_.png artifacts/cli-smoke` (릴리스): `elapsedMs` `4490 ms`, `landmarkSource: "onnx-auto"`, 26개 파츠(`model` 25개, `geometry` 1개 — `body`/`clothes` 분리 및 가변 머리카락 9레이어 포함), `2,426`개 삼각형, 메쉬 경고 `0`건.
- 모델 SHA-256 (`models/manifest.json`):
  - `anime_face_yolov3.onnx`: `fdf4cd4da82664dc8a377263d96182d265b2fca7b533e826df18eebc61295d60`
  - `anime_face_hrnetv2.onnx`: `68c970acb1603991d9389d4b1be647742c7d5a9701291b81c6747fcc3305ebee`
  - `u2netp.onnx`: `309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8`
  - `mobilesam_encoder.onnx`: `dd4687f70e0508f5b261a74cceb911f2bfb32e5c6a32fcc428039345da44e9cb`
  - `mobilesam_decoder.onnx`: `e654370ef8d761959b2df0388b21cc7392d9e2fe9275ee2e170832713fbe54aa`

임시 결과와 스크린샷은 Git에서 제외한 `artifacts/`에 저장됩니다. 네이티브 UI 검사는 테스트용 WebView2 디버깅 포트를 열고 검사 후 프로세스를 종료합니다.

Cubism Editor에서 PSD/CSV를 실제 가져오는 검증(`cubismEditorImport: "not-tested"`), 클린 미설치 PC 환경의 `.msi` 설치 실행 테스트, 20장 이상 외부 원화 데이터셋에 대한 대규모 분할 품질/보정 시간 측정은 수행하지 않았습니다. 자체 메쉬/JSON을 Cubism에 직접 적용하는 기능은 없습니다.


---

## 2026-11-15 T0 기준 검증 추가 (원화 보존 · ArtMesh · SD/LD 회귀 · 보존 병합)

> 아래 항목은 `docs/product-plan.md` / `docs/development-tasks.md` T0 기준에 따라 기존 기록(상단)을 유지한 채 구분 추가한 로컬 재현 기록입니다.

- **프론트엔드·타입·빌드**:
  - `npx tsc --noEmit` — 종료 코드 `0`
  - `npm test` — `tests/pipeline.test.ts` **28개 통과** (원화/가림 분리, 보존 파츠 겹침 거절 및 동적 머리카락 병합, 역할별 Cubism ArtMesh, `normal`/`multiply`/`additive` 합성 모드 저장·불러오기 및 비일반 모드 PSD 차단 포함)
  - `npm run build` — `vite v6.4.6` 종료 코드 `0`
- **Rust 코어·분석·Lint**:
  - `cargo test --manifest-path src-tauri/Cargo.toml` — `flat2rig_lib` **26개 통과, 0 failed** (역방향 `smoothstep`, 원화 가시 픽셀 보존·동적 머리카락, 파츠 역할별 ArtMesh, ONNX 매니페스트 5종 포함)
  - `cargo clippy --manifest-path src-tauri/Cargo.toml --all-targets -- -D warnings` — 종료 코드 `0`
- **네이티브 데스크톱 패키지 & WebView2 스모크**:
  - `npx tauri build --bundles msi` — `src-tauri/target/release/bundle/msi/flat2rig_0.2.0-1_x64_en-US.msi` 생성 (`27.11 MiB`)
  - `node scripts/native-smoke.mjs` (`FLAT2RIG_TEST_BINARY=src-tauri/target/release/flat2rig.exe`, 격리된 `WEBVIEW2_USER_DATA_FOLDER`, `v0.2.0-1`) — `"passed": true`, `"exceptions": []` (`18` 파츠, `parentParts: 10`, `rigParts: 18`, `modelsCount: 5`). 기본 WebView2 데이터 폴더 동시 접근 시 `HRESULT(0x800700AA)` 발생 확인.
- **참고 PSD 및 가용 고정 입력 평가 (`flat2rig-cli.exe` 릴리스)**:
  - `Himikan LD (683×1024)` (`--proportion ld`): `22`파츠, `restDiff` `exact=699,392/699,392` (`rgb>3=0`, `missing=0`, `extra=0`), `totalFillPixels=266,477`, `meshBad: NONE`
  - `Himikan SD (683×1024)` (`--proportion sd`): `22`파츠, `restDiff` `exact=699,392/699,392` (`rgb>3=0`, `missing=0`, `extra=0`), `totalFillPixels=285,291`, `meshBad: NONE`
  - `Himikan LD (1365×2048)` / `Himikan SD (1365×2048)`: 모두 `exact=2,795,520/2,795,520` (`rgb>3=0`, `missing=0`, `extra=0`), `meshBad: NONE`
  - `user_sd_glasses (1254×1254, SD)`: `24`파츠, 전경 내 `rgb>3=0` (`maxRgbOnFg=0`), `fill=950,922px`, `meshBad: NONE`. 다만 안경 낀 SD에서 눈 감김 모션 시 홍채/눈꺼풀-렌즈 테 경계 잔상 확인 (T3 과제).
  - `nonsquare_portrait (768×1024, LD)` — **검출된 실패/제한 기록**: U2NetP가 밝은 창 배경과 맞닿은 연파랑 머리카락 외곽을 배경으로 제거해 상반신 전경 손실 발생, `hair_side_L=0px`, `neck` 극단 포즈 반전 삼각형 `8`개(`meshBad`) 검출.
  - `letterbox_wide (1280×720, auto)`: `23`파츠, 전경 내 `rgb>3=598` (`maxRgb=6`, U2NetP 알파 경계 재합성 차이), `meshBad: NONE`. 극단 입 벌림 + 기울기 조합 시 턱선 하부 이음새 미세 노출.
- **비교 이미지 & UI/모드 정리**:
  - `artifacts/himikan-review/himikan_ld_vs_sd_sheet.png`, `artifacts/himikan-review/himikan_ld_vs_sd_parts_sheet.png`, `artifacts/himikan-review/himikan_artmesh_sheet.png`, `artifacts/t0-eval/t0_inputs_sheet.png`, `artifacts/native-ui.png`
  - UI에서 `원화 유지 (기본)` vs `스타일 변경` 모드를 분리하고, 실측 근거가 아닌 `윤곽 정합 XX%` 배지 표기를 제거했습니다.
- **현재 T0 남은 제한 (미완료·미검증 구분)**:
  - 목표 평가 세트(참고 PSD 외 일반형 2장·SD형 3장) 중 SD 외부 원화 `2`장이 로컬 미확보 상태로, 현재는 참고 PSD + 일반형 1장(`nonsquare_portrait`) + SD형 1장(`user_sd_glasses`) + 합성 레터박스 1장으로 검증했습니다.
  - CLI (`clip_to_mask` 기반 `draw_deformed_part`)와 GUI (`Viewport.tsx` Canvas 2D 삼각형 워프) 간 안티앨리어싱/서브픽셀 래스터 오차 정량 비교는 `T2` 공통 렌더러 분리 시 측정 예정입니다.
  - 실제 Live2D Cubism Editor 가져오기 및 미설치 클린 PC `.msi` 실행은 `미검증`입니다.

## T1–T7 통합 검증 (v0.2.0-2 · v6 시각 개선 및 릴리스 검증)
- **검증일**: 2026-11-15
- **실행 검사 및 빌드 (v0.2.0-2 최종 반영 후 통과)**:
  - `npx tsc --noEmit` — 종료 코드 `0`
  - `npm test` (`vitest run`) — `tests/pipeline.test.ts` **34 passed (34)** (`.flat2rig` v3 ZIP 저장·재열기·손상/차원/프리셋 수 거절, T2 스테이지 비누적·물리 초기화, T3 눈 4종·T4 입 7종 픽셀 차별화, T5 8종 표정 프리셋·L/R 윙크 구분·`autoMouthDemo` 기본 OFF, WebGL 미지원 Node 환경 `enableGpu` Canvas2D 폴백·스냅샷 동일성 포함)
  - `cargo clippy --manifest-path src-tauri/Cargo.toml --all-targets -- -D warnings` — 종료 코드 `0` (`0` warnings)
  - `cargo test --manifest-path src-tauri/Cargo.toml --lib` — **27 passed, 0 failed** (`7.97s`, `ureq` `rustls` + `native-tls` 업데이터 테스트 포함)
  - `npm run desktop:build -- --debug` 및 `npm run tauri -- build --bundles msi`: `vite v6.4.6`(`3.47s`) + `src-tauri/target/debug/flat2rig.exe` + `src-tauri/target/release/flat2rig.exe` + `src-tauri/target/release/bundle/msi/flat2rig_0.2.0-2_x64_en-US.msi` 생성 (`28,692,480` bytes / `27.36 MiB`, WiX `x64`)
  - `npm run test:native` (`scripts/native-smoke.mjs`, debug 및 `FLAT2RIG_TEST_BINARY=src-tauri\target\release\flat2rig.exe`, 격리 `WEBVIEW2_USER_DATA_FOLDER`, `v0.2.0-2`) — `"passed": true`, `"exceptions": []`, `minFont: 12` (`minEl: "SPAN.version:v0.2.0-2"`), `groupHeaders: 8`, `paramTabs: 7`, `parts: 18`, `parentParts: 10`, `rigParts: 18`, `vertices: 1294`, `samples: 60`, `extremePosesTested: 12`, `modelsCount: 5`
  - `flat2rig-cli.exe` `v6` 시각 점검:
    - `Himikan LD` (`683×1024`, `--proportion ld`): `17,615 ms`, `23`파츠, `11,150`개 삼각형, `flips: 0`, `warns: 0` (`mouth_inner` 닫힌 입으로 `empty`)
    - `user_sd_glasses` (`1254×1254`, SD auto): `31,848 ms`, `25`개 CLI 파츠, `23,891`개 삼각형, `flips: 0`, `warns: 0` (`acc_glasses` `fragmented:18`, `acc_hairpin` `fragmented:4` 진단 포함)
    - *(참고: 이전 `v5` 점검에서 `nonsquare_portrait` `768×1024`/`23`파츠/`flips: 0` 및 `multi_face_pair` `1408×768`/`23`파츠/`flips: 0` 확인; `v6`에서는 미재실행)*
- **실제 Windows WebView2 E2E 검증 (`native_e2e_verify.mjs`, debug + release `flat2rig.exe v0.2.0-2`, `Himikan LD` + `user_sd_glasses` SD)**:
  - 검증 흐름: 이미지 가져오기 → Rust ONNX 파츠 생성(`Himikan LD` `23`파츠, `user_sd_glasses` `24`파츠·`acc_glasses` 분리) → `2. 표정` 8종 카드 및 좌/우 윙크(`왼윙크 (L)` ≠ `오른윙크 (R)`)·커스텀 표정 생성 → `3. 조정` 눈 모양 4종 카드·무대 푸터 `실행 취소`/`Ctrl+Shift+Z` 다시 실행 → 헤더 `.flat2rig` ZIP 저장 → `4. 재생` 라이브 재생(`테스트용 자동 입 개폐 (데모)` 기본 OFF) → `일시정지`(물리 초기화) → `투명 배경 PNG 저장` → 저장된 `.flat2rig` 재열기 후 커스텀 표정(`9`개) 복원 (`passed: true`, `exceptions: []`)
  - **릴리스 바이너리 WebView2 렌더링 성능 (`GlMeshCompositor`, `300`프레임 `rAF`)**:
    - `Himikan LD (683×1024, 23 parts)`: `mean 14.5ms / p50 14.5ms / p95 15.1ms / max 15.6ms` (`~69 FPS`)
    - `user_sd_glasses (1254×1254, 24 parts)`: `mean 14.52ms / p50 14.5ms / p95 15.1ms / max 15.5ms` (`~68.9 FPS`)
  - **투명 PNG 및 프로젝트 ZIP 왕복 검증 (릴리스)**:
    - `himikan_input_expr_custom_9.png`: `683×1024`, `318,527 B`, `cornerAlpha: 0`, `nonEmptyPixels: 302,089` / `himikan_input.flat2rig`: `1,239,095 B` → 재열기 `restoredCount: 9, hasCustom: true`
    - `user_sd_glasses_expr_custom_9.png`: `1254×1254`, `2,283,738 B`, `cornerAlpha: 0`, `nonEmptyPixels: 1,211,696` / `user_sd_glasses.flat2rig`: `8,883,892 B` → 재열기 `restoredCount: 9, hasCustom: true`
  - *(참고: CLI ↔ GUI 래스터 오차 정량 비교는 `v4` 측정 시 `Himikan LD 683×1024` 기준 `rest maeAll=0.001 / diffGt8PctAll=0`, `head_yaw_turn diffGt8PctFg=5.167%`, `eyes_closed diffGt8PctFg=1.314%`, `mouth_open diffGt8PctFg=1.262%`, `motion_preview diffGt8PctFg=4.532%` 확인)*
- **시각 개선 및 남은 시각적 결함·회귀 항목 (해결 완료로 거짓 처리하지 않음)**:
  - **v5–v6에서 해소 확인된 항목**:
    - `Himikan LD`: 감은 눈 바깥쪽 회색 눈두덩 음영 제거(`src/rig/shapes.ts`), 세일러 카라·V넥 엠블럼 줄무늬 아티팩트 제거(`analysis/templates.rs`).
    - `user_sd_glasses`: 이마 앞머리 사각 파임(`face.png`) 및 감긴 눈 위 거대 보라 홍채·별/하트 잔상 제거, 곰돌이 머리띠·리본 보존 및 좌/우 헤어밴드 외곽선·`∞` 핀 주변 헤일로·미간 위 앞머리 삼각형·우측 상단 머리띠 조각 제거, 좌측 9시 방향 안경다리 연결부 복원.
  - **남은 시각적 결함 (미해결로 유지)**:
    - `user_sd_glasses`: WebView2 감은 눈(`미소`/`왼윙크`/`오른윙크`/`재생`)에서는 T3 눈 패치가 감김을 대체해 큰 흰 구멍 없이 렌더링되나, 감긴 눈 상단·안경 렌즈 내부 영역에 `acc_glasses`(`fragmented:18`) 상단 테/앞머리 교차 미세 조각이 남고, `X` 핀 한 획 일부가 분리되어 앞머리에 옅은 핀 흔적이 남음. 회전·표정 포즈 시 앞머리/볼·소매/손 경계에 얇은 반투명 흰 이음새가 보이며, raw CLI `motion_blink.png` 경로에서는 `face` 눈구멍 가장자리에 흰 틈이 드러남.
    - `Himikan LD`: 고개 회전(`play`/`export`) 시 세일러 카라·목 접합부에 미세 톱니 틈과 턱선 아래 얇은 반투명 가장자리가 노출됨(raw CLI `motion_blink.png`는 대체 눈꺼풀 없이 단순 스쿼시 방식 유지).
    - `nonsquare_portrait`: `U2NetP` 배경 제거 시 밝은 창 배경 인접 연파랑 머리카락 외곽 손실 및 T0 극단 포즈 `neck` 반전 회귀 사례 유지.
- **T7 명시적 `미검증` 항목**:
  - **독립된 원본 6장 평가 미충족**: 로컬 보유 독립 원화는 `Himikan`, `Rritteang (user_sd_glasses)`, `Sutteang (2d_.png)` 3종이며(`KakaoTalk_...`/`nonsquare_portrait`는 `Sutteang` 파생, `letterbox_wide`/`multi_face_pair`는 합성 검증 입력), 독립 원화 `3`장 추가 확보가 필요합니다.
  - **미검증 작업**: 신규 사용자 3인 15분 사용성 테스트, 미설치 클린 Windows PC에서의 `.msi` 설치·릴리스 자동 업데이트 동작, 실제 Live2D Cubism Editor에서의 `.psd` 가져오기, `2048px` 전용 실시간 60초 WebView2 장시간 메모리/성능 측정.

## 2단계(대머리 소체 → 머리카락 입히기) 파이프라인 및 메쉬/UX 검증 (2026-11-16)
- **단위 · 정적 검증 (코드 수정 후 실행)**:
  - `npx tsc --noEmit` 통과(exit `0`)
  - `npm test` (`vitest run`): `tests/pipeline.test.ts` **36 / 36 통과** (`isHairStageLayer`, `mergeStage2HairParts` 포함)
  - `cargo clippy --manifest-path src-tauri/Cargo.toml --all-targets -- -D warnings` 경고 `0`건 통과
  - `cargo test --manifest-path src-tauri/Cargo.toml --lib`: **30 / 30 통과** (`synthesize_template_bank_stages_split_base_and_hair_cleanly` 포함)
- **CLI 파이프라인 검증 (`flat2rig-cli`)**:
  - `cli_sutteang_base` (`--stage base_body`, `KakaoTalk_20261114_132504352.png`): 머리카락·모자·악세 제외 **18개 소체 파츠** 생성(`eyeL_upper: flip=2/hole=0/long=0`, `styleMatch=0.927`), `face.png` 두개골 피부 채움 연속성 개선 확인.
  - `cli_sutteang_v3` (`--stage full`): `23`파츠 (`styleMatch=0.927`), `eyeL_upper: flip=2`, `hair_back` 가슴 전체 과덮임 축소 및 `hair_side_L` 곡선 외곽 확인.
  - `cli_ritteang_v3` (`--stage full`, `Rritteang.png`): `25`파츠 (`styleMatch=0.961`), `eyeR_upper: flip=1`, `eyeL_upper: flip=2`.
  - `cli_input_v3` (`--stage full`, `artifacts/design-review/input.png`): `23`파츠 (`styleMatch=0.988`), **23개 전 메쉬 극단 포즈 뒤집힘/구멍교차/장삼각형 `0`건**.
- **네이티브 데스크톱(`flat2rig.exe`) 빌드 · 스모크 · WebView2 E2E**:
  - `npx tauri build --debug --no-bundle` 빌드 완료 후 `npm run test:native` (`scripts/native-smoke.mjs` 최신 UI 문구/버튼 반영 후 실행) 통과: `"passed": true`, `"exceptions": []`, `modelsCount: 5`, `minFont: 12`.
  - WebView2 E2E (`native_e2e_verify.mjs`) 통과 (`exceptions: []`):
    - `himikan_ld` (`683×1024`): 그룹 눈 토글(`true`), `Space+드래그` 팬(`cursor: grabbing`, `translate(36px, 24px)`), 브러시 스트로크 `695ms`, 클릭 보정 후보 `12`개(`후보 #2 · 98%` 자동 선택), 표정 8종·커스텀 저장(`.flat2rig`)·재불러오기 복원(`9`개), 투명 PNG 출력(`305,833` non-empty px), 실시간 재생 평균 `60.59ms/frame` (`~16.5 FPS`).
    - 워크스페이스 전환 확인 다이얼로그: `취소하고 현재 작업 유지` 선택 시 `himikan_input` 유지, `저장하고 열기` 선택 후 `Rritteang.png` 로드 확인.
    - `ritteang_sd` (`1254×1254`): **1단계 대머리 소체** `18`파츠(`hairFound: false`) → **2단계 머리카락 입히기** `24`파츠 병합·표정 `8`종·PNG 출력(`1,213,423` non-empty px)·`.flat2rig` 재불러오기(`9`개) 통과, 실시간 재생 평균 `90.84ms/frame` (`~11 FPS`).
- **관찰된 남은 한계 · 솔직한 비고**:
  - 기울어진 SD/안경 일러스트(`Sutteang` `eyeL_upper` 2개, `Rritteang` `eyeR_upper` 1개 / `eyeL_upper` 2개)에서는 극단 포즈 메쉬 뒤집힘이 아직 소수 남아 있으며, `Rritteang` `슬픔` 표정 등 안경 렌즈-감은 눈 겹침 아티팩트와 `Sutteang`/`Himikan` 목(`neck`) 중앙 밝은 얼룩·띠는 수동 보정 여지가 남아 있습니다.
  - WebView2 2D 캔버스 실시간 재생은 `1254×1254` 기준 `~11 FPS`(`p95 152.4ms`) 수준이므로 고해상도 실시간 `30~60 FPS` 체감은 추후 WebGL/배치 최적화가 필요합니다.
  - 이전 `T7 명시적 미검증 항목`(독립 원본 6장 충족, 신규 사용자 사용성 테스트, 클린 Windows `.msi` 설치·업데이트, 실제 Cubism Editor `.psd` 가져오기, `2048px` 60초 측정)은 여전히 미검증 상태입니다.

## 4단계 하이브리드 실루엣 적층(`1 소체 → 2 이목구비 → 3 의상 → 4 머리카락`) 및 클리핑 검증 (`0.2.0-3`)
- **단위 · 정적 · 릴리스 버전 정합성**:
  - `npx tsc --noEmit` 통과(exit `0`)
  - `npm test` (`vitest run`): `tests/pipeline.test.ts` **38 / 38 통과** (단조 눈꺼풀 커브 `0.45` 앵커 정합, 4단계 `mergeStageParts` 병합, `face_detail` 추출·`face` 클리핑 정지/모션 렌더 검증 포함)
  - `cargo clippy --manifest-path src-tauri/Cargo.toml --all-targets -- -D warnings` 경고 `0`건 통과
  - `cargo test --manifest-path src-tauri/Cargo.toml --lib`: **30 / 30 통과**
- **CLI 파이프라인 검증 (`flat2rig-cli`)**:
  - `cli_sutteang_base_sil3` (`--stage base_silhouette`, `ref/sutteang/2d_.png`, `1024×1024`): 순수 실루엣 소체 **6개**(`body`, `neck`, `face`, `eyeR_white`, `eyeL_white`, `mouth_inner`), `styleMatch=0.925` (`proportionMode: ld`), **6개 전 메쉬 극단 포즈 뒤집힘/구멍교차/장삼각형 `0`건**. 시각 검수상 앞머리 잔선·레이스 구멍·분리 파편 없이 매끈한 소체 확인.
  - `cli_sutteang_v6` (`--stage full`, `ref/sutteang/2d_.png`, `1024×1024`): **24개** 파츠(`face_detail` 볼터치·코 `54`삼각형 포함), **24개 전 메쉬 극단 포즈 뒤집힘/구멍교차/장삼각형 `0`건**.
  - `cli_ritteang_base_sil3` (`--stage base_silhouette`, `ref/ritteang/Rritteang.png`, `1254×1254`): 순수 실루엣 소체 **6개**, `styleMatch=0.961` (`proportionMode: sd`), **6개 전 메쉬 극단 포즈 뒤집힘/구멍교차/장삼각형 `0`건**. 시각 검수상 안경다리·앞머리 잔선 없는 순수 SD 소체 확인.
- **네이티브 데스크톱(`flat2rig.exe`) 빌드 · 스모크 · 4단계 WebView2 E2E**:
  - `npx tauri build --debug --no-bundle` 빌드 완료 후 `npm run test:native` (`scripts/native-smoke.mjs`) 통과: `"passed": true`, `"exceptions": []`, `modelsCount: 5`, `minFont: 12`.
  - 동일 구현(`0.2.0-3` 버전 문자열 범프 직전 빌드) WebView2 4단계 E2E (`native_e2e_verify.mjs`, `exceptions: []`) 전 캐릭터 통과:
    - `himikan_ld` (`683×1024`): 그룹 눈 토글(`true`), `Space+드래그` 팬(`cursor: grabbing`, `translate(36px, 24px)`), 브러시 `223ms`, MobileSAM 클릭 후보 `12`개(`후보 #2 · 98%` 자동 선택), 투명 PNG(`311,458` non-empty px), `.flat2rig` 복원(`9`개), 실시간 재생 평균 `43.65ms/frame` (`~22.9 FPS`).
    - `ritteang_sd` (`1254×1254`, 4단계 순차 실행): `1단계 · 실루엣 소체`(6파츠: `mouth_inner`, `eyeL_white`, `eyeR_white`, `face`, `neck`, `body`) → `2단계 · 이목구비 적층`(16파츠, `face_detail`은 8px 미만으로 생략) → `3단계 · 의상·소품 적층`(18파츠, `acc_glasses` 포함) → `4단계 · 머리카락 적층`(24파츠), 투명 PNG(`1,233,027` non-empty px), `.flat2rig` 복원(`9`개), 실시간 재생 평균 `52.56ms/frame` (`~19 FPS`).
    - `sutteang_sd` (`1024×1024`): `24`파츠, 표정 8종·투명 PNG(`854,156` non-empty px)·`.flat2rig` 복원(`9`개), 실시간 재생 평균 `38.89ms/frame` (`~25.7 FPS`).
- **관찰된 남은 시각적 결함 · 미해결 항목**:
  - `Rritteang SD` (`미소`/`왼윙크`/`오른윙크`/`슬픔`): 기울어진 턱 아래 소매 손 경계 인접부에 미세한 반투명 흰 이음새 초승달 잔여, `슬픔` 포즈 시 감은 눈 상단 안경테 교차부에 점선형 속눈썹 미세 흔적 잔존.
  - `Sutteang` (`2d_.png`): `neck` 턱 밑 그림자 외곽·목 윗부분에 원화 셔츠 칼라·머리카락 인접 밝은 흰 얼룩 잔존, `슬픔` 포즈 시 아래로 내린 눈꺼풀 외곽선에 얇은 피부 이음새 관찰.
  - 기존 `T7 명시적 미검증 항목`(독립 원본 6장 확보, 신규 사용자 3인 테스트, 클린 Windows `.msi` 설치/자동 업데이트, 실제 Cubism Editor `.psd` 가져오기, `2048px` 60초 측정) 및 `multiply`/`additive` 합성 모드 미지원은 동일하게 유지됩니다.

## 5대 모션·이음새 검증 기준 (2026-11-16 개정 · T12 예정 게이트)

> 아래 5개 기준은 [product-plan.md](product-plan.md) · [development-tasks.md](development-tasks.md) 개정 이후 추가된 목표 검증 게이트입니다. 각도 축소로 결함을 덮지 않고 독립 메쉬·텍스처가 움직이는 상태에서 검사하는 자동 리포트(`T12`)는 아직 구현·실행되지 않았습니다(`미구현 / 미검증`).

1. **기본 포즈 자연스러움**: 원본과 동일한 기본 포즈에서 색상·실루엣이 자연스러운가? (현재 CLI `restDiff` 및 WebView2 육안 검수 수준으로 부분 확인)
2. **얼굴·목 반대 이동 틈**: 얼굴과 목을 반대 방향으로 움직여도 틈이 없는가? (`미검증` — 현재는 단일 파츠 `check_mesh_quality`만 실행되며 멀티레이어 틈 자동 검사는 없음; 육안 검수상 `Himikan`/`Sutteang`/`Rritteang` 목·턱 경계 미세 결함 잔존)
3. **눈 감김 누출**: 눈을 감을 때 안쪽 홍채·흰자가 새어 나오지 않는가? (`부분 확인` — WebView2 `미소`/`윙크`는 큰 누출 없이 동작하나 `슬픔` 포즈 미세 이음새 및 안경 교차 흔적 잔존)
4. **머리카락 이동 공백**: 머리카락을 움직일 때 피부나 배경의 빈 공간이 노출되지 않는가? (`미검증` — 현재 고정 `8~14px` 언더랩 기준이며 최대 흔들림 자동 공백 검사 없음)
5. **경계 이중 선화·흰 테두리**: 파츠가 겹치는 곳에서 선화가 이중으로 보이거나 흰 테두리가 나타나지 않는가? (`미해결` — `Rritteang` 턱/손 경계 반투명 흰 이음새, `Sutteang` 목 상단 흰 얼룩 등 잔여 결함 확인됨)

6. **(2026-11-18 추가) 노출 영역 원본감**: 허용 모션에서 드러나는 노출 밴드가 원화와 같은 팔레트·그림자 경계·선화 연속성을 갖는가? (`미구현 / 미측정` — 측정 하네스(`T14`)와 노출 맵(`T13`)이 아직 없음. 1–5번 결함이 0이어도 이 항목 수치 없이 통과로 기록하지 않음)

## T8 1차 구현 상태 (2026-11-18 문서 검수 기록)

- `79610ec` `feat(rig): scale per-part fillMask overlap margins with resolution and motion (T8)`이 반영되었다. 커밋에 `part_overlap_margins_scale_with_resolution_and_cover_opposing_motion` Rust 테스트와 `tests/pipeline.test.ts` 추가분이 포함되어 있다.
- 이번 검수는 문서·소스 검토만 했고 테스트·CLI·WebView2를 실행하지 않았다. T8의 완료 조건(`1024px+` 원화에서 반대 이동·최대 흔들림 시 공백 0)은 **`미검증`**이다.
- 소스 검수에서 확인한 제한(구조 개선은 `T13`): 변형 상수의 3중 복제(`deform` / `part_overlap_margins` / `partOverlapMargins`), 테스트 표본이 2점·단일 포즈뿐, Rust `recommended_fill_radius`의 ID 접두사 분기와 TS `role` 분기 불일치. 근거는 [template-architecture-review.md](template-architecture-review.md) 2026-11-18 항목에 있다.

## T13a 이음새 계약 · 포즈 포락선 · `deform` 수치 여유분 (2026-11-18, 리뷰 수정 반영)

- 실행: `cd src-tauri && cargo test` (lib 43/43 통과), `npm test` (42/42 통과), `npx tsc --noEmit` (통과). Windows 앱·실제 원화(`Himikan`/`Sutteang`/`Rritteang`) 모션 검증은 하지 않았다(`미검증`).
- 수치 출처: `analysis::templates::tests::t13a_seam_requirements_drive_underlaps_and_coverage_is_reported_on_1024_artwork` (`cargo test --lib t13a -- --nocapture`). 1024×1024 **합성** fixture(눈 간격 224px, 머리 높이 440px)이며 입력 레이어는 `hair_back`·`neck`·`clothes`·`face`·`hair_side_R`·`hair_side_L_2`·`hair_front`·`hair_front_2`, 템플릿 합성 후 23개 레이어·이음새 55개다.
- 포즈 포락선: `src/rig/pose-envelope.json` 94포즈, `PARAM_DEFS` 범위 축소 없음.
- 깊이 측정(v2): 경계점 b의 상위 경계가 놓이는 위치 `yu = deform(upper)(b)`에서 하위 레이어 역변환 `x_l = deform(lower)⁻¹(yu)`을 구하고 `b → x_l` 노출 띠를 1px 간격으로 걷는다. 지원 표본(전경 · 휴지 소유자가 하위이거나 그보다 위)의 하위 가시 마스크 바깥 거리 최댓값이 깊이다. 수치 여유분 = `ceil(깊이 × 1.25 + 2)`. 커버리지 검사도 같은 헬퍼(`walk_exposed_band`)로 같은 띠를 걷는다.
- v1(리뷰 전)은 끝점 `b + r` 한 점만 보고 끝점이 배경·하위 아래면 표본을 버렸다. 아래 표의 "v1 → v2"는 같은 fixture에서 측정 방식만 바꾼 결과다. "T8 px" 열은 같은 하위 역할에 T8 코드가 쓰던 `part_overlap_margins` 값이다(↑ 커짐 · ↓ 작아짐 · 신규 = T8에 해당 값 없음). 얼굴 아래 눈·입 이음새의 T8 값(40)은 참고용이며 T8 코드는 얼굴을 눈·입 아래로 채우지 않았다.

| upper (role) | lower (role) | boundary px | max rel. disp px | max exposure depth px (v1 → v2) | numeric underlap px (v1 → v2) | T8 px (vs v2) | worst pose @ point (v2) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| neck (Neck) | hair_back (HairBack) | 174 | 59.1 | 49.5 → 47.6 |  **64 → 62** | 104 ↓ | headHairBody:head--/hair+ @ (558.5,639.0) |
| clothes (Clothes) | hair_back (HairBack) | 118 | 103.1 | 100.0 → 73.0 |  **127 → 94** | 104 ↓ | opp3:head+--/breath1 @ (616.0,639.5) |
| clothes (Clothes) | neck (Neck) | 93 | 45.0 | 38.7 → 26.4 |  **51 → 36** | 79 ↓ | same3:head+--/breath1 @ (476.0,639.5) |
| face (Face) | hair_back (HairBack) | 326 | 21.3 | 23.7 → 21.4 |  **32 → 29** | 104 ↓ | headHairBody:head++/hair- @ (465.0,552.5) |
| face (Face) | neck (Neck) | 109 | 9.2 | 9.8 → 10.0 | 15 → 15 | 77 ↓ | opp3:head+--/breath1 @ (558.0,552.5) |
| eyeR_white (Eye) | face (Face) | 18 | 14.7 | 11.3 → 11.3 | 17 → 17 | 40 ↓ | eyeR=0 @ (449.0,375.5) |
| eyeR_iris (Eye) | face (Face) | 40 | 34.7 | 35.0 → 35.0 | 46 → 46 | 40 ↑ | eyesClosedSmile @ (398.0,393.5) |
| eyeR_iris (Eye) | eyeR_white (Eye) | 57 | 11.5 | 13.5 → 13.5 | 19 → 19 | 18 ↑ | gaze:-- @ (420.5,337.0) |
| eyeR_lower (Eye) | face (Face) | 96 | 31.1 | 32.0 → 32.0 | 42 → 42 | 40 ↑ | eyesClosedSmile @ (415.0,391.5) |
| eyeR_lower (Eye) | eyeR_white (Eye) | 28 | 4.6 | 2.4 → 11.0 |  **6 → 16** | 18 ↓ | eyeR=0 @ (424.0,379.5) |
| eyeR_lower (Eye) | eyeR_iris (Eye) | 22 | 11.5 | 13.5 → 13.5 | 19 → 19 | 18 ↑ | gaze:-+ @ (423.5,381.0) |
| eyeR_upper (Eye) | face (Face) | 56 | 38.6 | 39.0 → 39.0 | 51 → 51 | 40 ↑ | eyeR=0 @ (401.0,315.5) |
| eyeR_upper (Eye) | eyeR_white (Eye) | 14 | 10.7 | 11.0 → 11.0 | 16 → 16 | 18 ↓ | eyesClosedSmile @ (421.0,335.5) |
| eyeR_upper (Eye) | eyeR_iris (Eye) | 43 | 12.0 | 13.5 → 13.5 | 19 → 19 | 18 ↑ | gaze:-- @ (420.5,335.0) |
| eyeL_white (Eye) | face (Face) | 65 | 17.5 | 15.6 → 15.6 | 22 → 22 | 40 ↓ | eyeL=0 @ (580.5,341.0) |
| eyeL_iris (Eye) | face (Face) | 40 | 34.7 | 35.0 → 35.0 | 46 → 46 | 40 ↑ | eyesClosedSmile @ (622.0,393.5) |
| eyeL_iris (Eye) | eyeL_white (Eye) | 57 | 11.5 | 13.5 → 13.5 | 19 → 19 | 18 ↑ | gaze:++ @ (603.5,336.0) |
| eyeL_lower (Eye) | face (Face) | 96 | 31.1 | 32.0 → 32.0 | 42 → 42 | 40 ↑ | eyesClosedSmile @ (609.0,391.5) |
| eyeL_lower (Eye) | eyeL_white (Eye) | 28 | 4.6 | 2.0 → 11.0 |  **5 → 16** | 18 ↓ | eyeL=0 @ (600.0,379.5) |
| eyeL_lower (Eye) | eyeL_iris (Eye) | 22 | 11.5 | 13.5 → 13.5 | 19 → 19 | 18 ↑ | gaze:++ @ (600.5,381.0) |
| eyeL_upper (Eye) | face (Face) | 162 | 38.6 | 39.0 → 39.0 | 51 → 51 | 40 ↑ | eyeL=0 @ (625.0,315.5) |
| eyeL_upper (Eye) | eyeL_white (Eye) | 28 | 10.7 | 11.0 → 11.0 | 16 → 16 | 18 ↓ | eyesClosedSmile @ (603.0,335.5) |
| eyeL_upper (Eye) | eyeL_iris (Eye) | 43 | 12.0 | 13.5 → 13.5 | 19 → 19 | 18 ↑ | gaze:+- @ (603.5,335.0) |
| browR (Brow) | face (Face) | 22 | 22.7 | 23.0 → 23.0 | 31 → 31 | 40 ↓ | brows:y+form+ @ (401.0,306.5) |
| browL (Brow) | face (Face) | 22 | 22.7 | 23.4 → 23.4 | 32 → 32 | 40 ↓ | brows:y+form+ @ (623.5,307.0) |
| mouth_inner (Mouth) | face (Face) | 86 | 21.9 | 3.0 → 3.0 | 6 → 6 | 40 ↓ | mouthForm=-1 @ (531.5,484.0) |
| mouth_lower (Mouth) | face (Face) | 58 | 28.3 | 5.0 → 5.0 | 9 → 9 | 40 ↓ | mouthForm=1 @ (478.0,499.5) |
| mouth_lower (Mouth) | mouth_inner (Mouth) | 39 | 7.9 | 0.0 → 0.0 | 2 → 2 | - 신규 | - @ - |
| mouth_upper (Mouth) | face (Face) | 64 | 14.4 | 8.0 → 8.0 | 12 → 12 | 40 ↓ | mouthForm=-1 @ (471.0,484.5) |
| mouth_upper (Mouth) | mouth_inner (Mouth) | 39 | 12.8 | 8.0 → 8.0 | 12 → 12 | - 신규 | mouth=1 @ (493.0,484.5) |
| mouth_upper (Mouth) | mouth_lower (Mouth) | 85 | 25.1 | 0.0 → 9.0 |  **2 → 14** | - 신규 | mouth=1 @ (470.0,493.5) |
| hair_side_R (HairSideR) | hair_back (HairBack) | 191 | 33.2 | 34.0 → 34.0 | 45 → 45 | 104 ↓ | hairSign:--+ @ (406.5,637.0) |
| hair_side_R (HairSideR) | clothes (Clothes) | 186 | 170.9 | 160.8 → 155.4 |  **203 → 197** | 62 ↑ | headHairBody:head+-/hair- @ (312.0,720.5) |
| hair_side_R (HairSideR) | face (Face) | 84 | 23.1 | 25.1 → 23.0 |  **34 → 31** | 40 ↓ | headHairBody:head++/hair- @ (380.5,463.0) |
| hair_side_R (HairSideR) | eyeR_iris (Eye) | 53 | 30.5 | 18.9 → 17.0 |  **26 → 24** | 18 ↑ | headHairBody:head++/hair- @ (380.5,381.0) |
| hair_side_L (HairSideL) | hair_back (HairBack) | 175 | 33.2 | 34.0 → 34.0 | 45 → 45 | 104 ↓ | hairSign:++- @ (617.5,637.0) |
| hair_side_L (HairSideL) | clothes (Clothes) | 26 | 114.7 | 122.5 → 121.4 |  **156 → 154** | 62 ↑ | opp3:head+--/breath1 @ (643.0,639.5) |
| hair_side_L_2 (HairSideL) | hair_back (HairBack) | 16 | 24.5 | 25.0 → 25.0 | 34 → 34 | 104 ↓ | hairSign:++- @ (643.5,485.0) |
| hair_side_L_2 (HairSideL) | clothes (Clothes) | 200 | 186.6 | 134.9 → 124.9 |  **171 → 159** | 62 ↑ | opp3:head+--/breath1 @ (678.0,760.5) |
| hair_side_L_2 (HairSideL) | face (Face) | 84 | 25.4 | 27.6 → 26.0 |  **37 → 35** | 40 ↓ | headHairBody:head++/hair+ @ (643.5,473.0) |
| hair_side_L_2 (HairSideL) | eyeL_iris (Eye) | 53 | 30.5 | 30.2 → 30.2 | 40 → 40 | 18 ↑ | eyesClosedSmile @ (643.5,390.0) |
| hair_side_L_2 (HairSideL) | hair_side_L (HairSideL) | 450 | 3.8 | 4.8 → 4.0 |  **9 → 7** | 41 ↓ | hairSign:+++ @ (643.5,596.0) |
| hair_front (HairFront) | face (Face) | 190 | 9.3 | 10.7 → 9.4 |  **16 → 14** | 40 ↓ | headHairBody:head-+/hair- @ (510.0,324.5) |
| hair_front (HairFront) | eyeR_iris (Eye) | 13 | 29.7 | 30.0 → 30.0 | 40 → 40 | 18 ↑ | eyeR=0 @ (380.0,334.5) |
| hair_front (HairFront) | eyeR_upper (Eye) | 17 | 36.6 | 37.0 → 41.0 |  **49 → 54** | - 신규 | eyeR=0 @ (380.5,324.0) |
| hair_front (HairFront) | eyeL_iris (Eye) | 13 | 29.7 | 30.0 → 30.0 | 40 → 40 | 18 ↑ | eyeL=0 @ (644.0,334.5) |
| hair_front (HairFront) | eyeL_upper (Eye) | 17 | 36.6 | 37.4 → 41.4 |  **49 → 54** | - 신규 | eyeL=0 @ (643.5,324.0) |
| hair_front (HairFront) | browR (Brow) | 42 | 22.8 | 23.2 → 23.2 | 32 → 32 | - 신규 | brows:y-form- @ (399.5,305.0) |
| hair_front (HairFront) | browL (Brow) | 42 | 22.8 | 22.6 → 22.6 | 31 → 31 | - 신규 | brows:y-form- @ (625.0,304.5) |
| hair_front (HairFront) | hair_side_R (HairSideR) | 161 | 10.3 | 10.0 → 10.0 | 15 → 15 | 41 ↓ | hairSign:-++ @ (372.5,347.0) |
| hair_front (HairFront) | hair_side_L_2 (HairSideL) | 161 | 10.3 | 10.0 → 10.0 | 15 → 15 | 41 ↓ | hairSide=-1 @ (653.5,355.0) |
| hair_front_2 (HairFront) | face (Face) | 108 | 12.2 | 12.7 → 11.4 |  **18 → 17** | 40 ↓ | headHairBody:head-+/hair- @ (508.0,335.5) |
| hair_front_2 (HairFront) | eyeR_white (Eye) | 55 | 20.2 | 20.4 → 20.4 | 28 → 28 | 18 ↑ | eyeR=0 @ (432.5,338.0) |
| hair_front_2 (HairFront) | eyeR_upper (Eye) | 20 | 34.7 | 35.0 → 44.4 |  **46 → 58** | - 신규 | eyeR=0 @ (430.5,324.0) |
| hair_front_2 (HairFront) | hair_front (HairFront) | 181 | 0.9 | 1.4 → 1.4 | 4 → 4 | 41 ↓ | hair=-1 @ (510.0,319.5) |

v1 → v2 변화의 원인:

- **커진 곳**(띠 내부를 보게 되어): 아래 눈꺼풀 → 흰자 5–6 → 16, `mouth_upper` → `mouth_lower` 2 → 14, 앞머리 → 윗눈꺼풀 46–49 → 54–58. 끝점은 하위 안쪽이나 배경에 떨어지지만 띠 중간은 상위가 비운 영역이다.
- **작아진 곳**(끝점 1차 근사 대신 하위 역변환을 써서): `clothes` → `hair_back` 127 → 94, `clothes` → `neck` 51 → 36, 옆머리 → 의상 2–12px 감소 등. 하위 레이어가 회전·비균일 변형을 받으면 `b + r`은 하위의 실제 휴지 점이 아니다. 예: `clothes`/`hair_back` (617.0,639.5), `same3:head+--/breath1`에서 `hair_back`은 고개와 함께 30° 돈다. `b + r` = (589.9,739.0)이지만 `deform(hair_back)`의 역상은 (543.5,705.0)(잔차 0.36px)이다. v1 최댓값 100.0px는 이 점의 `b + r`에서 나왔고, 역상 쪽 띠는 `hair_back` 가시 영역에 더 가깝다. v2 최댓값은 73.0px(다른 경계점 (616.0,639.5), `opp3:head+--/breath1`)다. `clothes`/`neck` (466.0,639.5)은 역상과 `b + r` 차이가 작고(잔차 0.01px), 최댓값이 다른 경계점·포즈((476.0,639.5) `same3:head+--/breath1`)로 옮겨 38.7 → 26.4px가 됐다. 감소는 표본을 버려서가 아니라 하위의 실제 휴지 점을 쓴 결과다. v2에서 띠 전체가 배경·하위 아래뿐인 경우만 `unsupportedSamples`로 센다.

T8 대비 커진 곳과 이유:

- **옆머리 → 의상**(`hair_side_*`/`clothes`, 154–197 vs 62): 가슴 위로 늘어진 머리카락은 머리 부품이라 고개 기울기 30°의 턱 피벗 회전과 흔들림을 모두 받고, 의상은 반대 방향 몸 회전을 받는다. 경계점 상대 변위가 최대 187px이다. T8 폐형식은 흔들림 항만 넣었다.
- **눈꺼풀·홍채 → 얼굴**(42–51 vs 40), **앞머리/옆머리 → 홍채·흰자**(24–40 vs 18), **앞머리 → 윗눈꺼풀**(54–58, T8 해당 없음): 눈 감김 수축과 시선 이동, 머리카락 흔들림이 실제 `deform`에서 T8 근사보다 크다.

T8 대비 작아진 곳과 근거(최악 포즈·경계점은 표의 마지막 열):

- **얼굴 → 목**(15 vs 77): 목 위쪽 가중치가 턱에서 1이라 턱선 경계는 머리와 함께 움직이고, 남는 상대 변위는 Z 회전의 22%(목 회전 가중 0.78)뿐이다. 최대 상대 변위가 9.2px이다. T8은 머리 절대 변위와 `hh·0.04` 상수를 더했다.
- **앞머리 → 얼굴**(14 vs 40), **앞머리 → 옆머리**(15 vs 41), **`hair_front_2` → `hair_front`**(4 vs 41), **옆머리끼리**(7 vs 41): 두 레이어가 같은 머리 변형을 받아 상대 변위가 흔들림 진폭 차이뿐이다(최대 상대 변위 0.9–10.3px).
- **뒷머리 아래 이음새**(`face` 29, `hair_side_*` 34–45, `neck` 62, `clothes` 94 vs 104): `face`·`hair_side`·`neck` 위쪽은 `hair_back`과 같은 머리 변형을 받아 최대 상대 변위가 21–59px이다. 깊이는 그보다 작거나 비슷하다. T8 값은 머리 절대 변위·흔들림·`hh·0.06`을 모두 더한 상한이다. `clothes`/`hair_back`은 상대 변위가 103px이지만 위의 역변환 예처럼 `hair_back`이 함께 돌아 실제 노출 띠의 끝이 `hair_back` 가시 영역 쪽에 있다.
- **눈 아래 눈꺼풀 → 흰자**(16 vs 18), **입 → 얼굴**(6–12 vs 40): 상위가 하위 쪽으로 덮어 들어가는 방향이 대부분이라 드러나는 깊이가 작다.

커버리지 검사(`check_seam_coverage`, 경계점 1/2 표본 × 94포즈, v2):

| 항목 | v1 | v2 |
| --- | ---: | ---: |
| 검사 수(경계점 × 포즈) | 221,370 | 221,370 |
| 배경 노출(`holes`) | 0 | 0 |
| 하위 레이어 계약 위반(`lowerMisses`, 다른 레이어가 덮음) | 3,748 | 3,742 |

- `lowerMisses`는 모두 다른 레이어의 변형된 유효 마스크가 덮는다(`holes` 0). 덮는 레이어는 개별 기록하지 않았고 괄호는 위치상 추정이다. 주요 위치: `hair_side_R`/`clothes` 1,577 · `hair_side_L_2`/`clothes` 1,347 · `hair_side_L`/`clothes` 299(어깨 밖으로 늘어진 옆머리 아래, `clothes` 실루엣 밖 → `body`/`hair_back` 추정), `mouth_upper`/`mouth_lower` 129(`mouth_inner` 추정), `hair_front`/눈썹 63·81 · `hair_front*`/윗눈꺼풀 26·29·44(`face`), `clothes`/`neck` 13(옷깃 바깥 x≈444/577, `neck` 블록의 폭 제한 밖 → `body` 추정).
- **미해결(T13b)**: 앞머리 아래 눈썹·윗눈꺼풀 `lowerMisses`는 구멍은 아니지만, 그 자리에 눈썹·눈꺼풀 대신 얼굴 피부가 보이는 상태다. 측정 요구량(31–58px)은 있으나 `Brow`/눈꺼풀을 하위로 받는 확장 블록이 없어 쓰이지 않는다. [development-tasks.md](development-tasks.md) T13 절 참조.
- 얼굴 기준 레이어가 눈·입 아래를 채우기 전(v1 초기)에는 같은 fixture에서 `holes` 42건(눈 아래 테두리, 깊이 1.5–13.5px)이 나왔다.
- 제한: 경계점마다 하위 역변환 띠 하나로 검사하며 ArtMesh 삼각형 보간으로 렌더링해 비교하지 않는다. 합성 fixture 결과이므로 5대 기준 ②·④의 원화 검증을 대신하지 않는다. 노출 영역 원본감(⑥)은 측정하지 않았다(T14).
