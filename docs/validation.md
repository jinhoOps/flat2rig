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

6. **(2026-11-18 추가) 노출 영역 원본감**: 허용 모션에서 드러나는 노출 밴드가 원화와 같은 팔레트·그림자 경계·선화 연속성을 갖는가? (`기준선만 측정 · 판정 없음` — T14 하네스(`flat2rig-cli --eval-holdout` / `--eval-truth`)가 노출 밴드 안 ΔE2000·팔레트·이음새 단차·선화 끊김·국소 분산을 기록한다(아래 T14 절). 임계값은 정하지 않았으므로 통과/미통과를 판정하지 않는다. 1–5번 결함이 0이어도 이 항목을 통과로 기록하지 않음)

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

## T13b 노출 맵(`ExposureMap`) · 이음새 결합 변형 · CLI/오버레이 (2026-11-18, 리뷰 수정 반영)

- 실행: `cd src-tauri && cargo test` (lib 49/49 통과), `cargo clippy --all-targets -- -D warnings` (통과), `npm test` (43/43 통과), `npx tsc --noEmit` (통과). Windows 앱에서 결합 변형 모션과 `노출 빈도 히트맵 (T13b)` 오버레이를 실제로 확인하지 않았다(`미검증`).
- 수치 출처: 표 1·2는 `rig::seams::tests::t13b_coupling_reduces_seam_relative_displacement_and_keeps_far_points_unchanged`와 `analysis::templates::tests::t13b_coupling_and_exposure_map_on_1024_artwork` (`cargo test --lib t13b_ -- --nocapture`, 출력 접두사 `T13B_*`). 표 3·4는 릴리스 `flat2rig-cli`(`--uncoupled` = 결합 전, 기본 = 결합 후)의 `stats.json`. 결과 폴더 `artifacts/t13b/<원화>-{uncoupled,coupled}/` (Git 제외, `exposure.png` 포함).
- "결합 전"은 스레드 로컬 스위치(`core::with_seam_coupling(false, …)`, CLI `--uncoupled`)로 숨은 영역 합성·이음새 요구량·커버리지·노출 맵을 **모두** 비결합 `deform`(T13a와 같은 수식, `core::deform_uncoupled`)으로 돌린 결과다. "결합 후"는 런타임 `deform`이 결합이고, 숨은 영역은 이음새마다 결합/비결합 요구량 중 큰 값(`hidden_area_seam_requirements`)으로 만든다. 두 경우 모두 T13b 눈썹·윗눈꺼풀 블록(5b)은 켜져 있다.

### 리뷰 수정 요약

1. **숨은 영역은 결합으로 줄지 않는다**: 언더랩 생성은 결합/비결합 요구량의 max. 1024 fixture 모든 레이어에서 결합 후 숨은 텍셀 집합이 결합 전 집합을 포함한다(테스트 고정, 목 32,441 → 32,441px). 처음 구현(결합 요구량만 사용)은 목 숨은 영역이 줄었다(fixture 32,441 → 31,766, Rritteang 43,309 → 30,028, Sutteang 42,341 → 22,270px).
2. **Rritteang `face`/`clothes` 구멍 0 → 12 회귀 해소**: 원인은 옷깃 열이 결합된 목 장 전체(목 상단↔머리 항 포함)를 따라가 턱 바로 아래 옷깃이 머리 회전을 물려받은 것. 옷깃은 이제 목 하단 장(목 상단↔머리 항 없음) 쪽으로 `column · neck_body` 가중치로만 섞인다. 수정 후 `face`/`clothes`: 상대 변위 101.1 → 101.1px, 구멍 0 → 0.
3. **결합 분류 역할 기반**: Rust `coupling_class`·TS `couplingClass`가 `infer_part_role`/`inferPartRole`의 `neck`으로 목을 판정한다. 동등성 fixture를 다시 생성했다.
4. 목 상단↔머리 감쇠 0.12·hh → 0.06·hh (아래 근거).

### 결합 가중치 (Rust `core.rs` / TS `engine.ts` 같은 수식)

`coupled(x) = own(x) + c(x)·(partner(x) − own(x))`, `ss`는 smoothstep(3t²−2t³). `hh`는 머리 높이, `v`·`u`는 휴지 위치의 머리 좌표.

| 이음새 | 결합되는 쪽 → 상대장 | `c(x)` | 감쇠 거리 (1024 fixture) |
| --- | --- | --- | --- |
| 목 상단 ↔ 머리 (`NeckHead`) | `neck` 역할 → `face` 장 | `1 − ss(chin_v, chin_v + 0.06·hh, v)` | 0.06·hh = 26px |
| 목 하단 ↔ 몸통 (`Collar`) | `neck` 역할 → `body` 장 | `ss(chin_v, chin_v + 0.42·hh, v)` (0.42는 기존 목 가중치 상수 `NECK_SPAN_HH`) | 0.42·hh = 185px |
| 옷깃 ↔ 목 하단 (`Collar`) | `body`·`clothes` 역할 → 목 하단 장 (`neck_raw + ss(…)·(body − neck_raw)`, 목 상단 항 없음) | `column(|u|) · ss(chin_v, chin_v + 0.42·hh, v)`, `column = 1 − ss(0.35·eye_dist, 0.60·eye_dist, |u|)` | 열 폭 78–134px, 세로 185px |
| 머리카락 뿌리 ↔ 두피 (`HairRoot`) | `hair*` → `face` 장 | `1 − ss(0, 0.25, (y − root_y)/hh)` (`root_y`는 `hair_sway_profile` 루트) | 0.25·hh = 110px |

- 목 상단 감쇠를 0.06·hh로 줄인 근거(원화, 옷깃 결합 수정 후 `clothes`/`neck` 최대 상대 변위, 결합 전 → 후): 0.12·hh에서 Rritteang 71.8 → 81.4, Sutteang 63.7 → 73.3px(옷깃 선이 턱 아래 ≈0.09·hh, 감쇠 안쪽이라 목이 옷깃 높이까지 머리를 따라감). 0.06·hh에서 71.8 → 72.3, 63.7 → 65.8px. 대가로 Rritteang `face`/`neck` 감소폭이 21.4 → 6.8에서 21.4 → 16.2px로 작아졌다. 1024 fixture `face`/`neck`은 두 경우 모두 9.2 → 0.0px.
- Rust/TS 동등성: `tests/fixtures/seam-envelope-parity.json`(Rust가 `FLAT2RIG_BLESS=1`로 생성)에 18개 점 집합의 결합/비결합 최대 상대 변위, 먼 기준점 17개 최대 변위, 결합 가중치 10개 표본을 넣었고 Rust는 `1e-9`, TS는 변위 `1e-6`·가중치 `1e-9` 안에서 같다(통과). TS는 명시 역할 `neck` 레이어(`neck_inner`)가 턱선에서 `neckHead = 1`을 받는지, 턱 바로 아래 옷깃(`y = 566`)이 `neckHead = 0`·`collar < 0.01`로 비결합 위치에서 0.5px 안에 머무는지도 검사한다.

### 표 1 — 이음새별 결합 전/후 최대 상대 변위 · 필요 언더랩 (1024px 합성 fixture, 94포즈)

언더랩 열은 "결합 전 → 결합 후 채택값 (결합 요구량 / 비결합 요구량, 채택)". 결합 대상 이음새와 결합으로 값이 바뀐 이음새만 적었다. 나머지(눈·입·눈썹 ↔ 얼굴 등)는 전/후 값이 같다.

| upper / lower | 결합 (up/low 평균 가중치) | 최대 상대 변위 px | 최대 노출 깊이 px | 필요 언더랩 px | `lowerMisses` | `holes` |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| face / neck | NeckHead (0.00 / 1.00) | 9.2 → **0.0** | 10.0 → 0.0 | 15 → 15 (2 / 15, 비결합) | 3 → 0 | 0 → 0 |
| clothes / neck | Collar (0.40 / 0.40) | 45.0 → **16.4** | 26.4 → 13.4 | 36 → 36 (19 / 36, 비결합) | 13 → 0 | 0 → 0 |
| clothes / hair_back | 없음 (옷깃 부수 효과) | 103.1 → 97.1 | 73.0 → 68.1 | 94 → 94 (88 / 94, 비결합) | 0 → 0 | 0 → 0 |
| neck / hair_back | 없음 (목 하단 부수 효과) | 59.1 → **61.2** | 47.6 → 57.8 | 62 → **75** (75 / 62, 결합) | 0 → 0 | 0 → 0 |
| hair_side_R / clothes | 없음 | 170.9 → 170.9 | 155.4 → 155.4 | 197 → 197 (같음) | 1,577 → 1,582 | 0 → 0 |
| hair_side_L / clothes | 없음 | 114.7 → 114.6 | 121.4 → 121.4 | 154 → 154 (같음) | 299 → 304 | 0 → 0 |
| hair_side_L_2 / clothes | 없음 | 186.6 → 186.6 | 124.9 → 124.9 | 159 → 159 (같음) | 1,347 → 1,345 | 0 → 0 |
| 머리카락 이음새 10개 (face/hair_back, hair_front/face 등) | HairRoot (0.00 / 0.00) | 변화 없음 | 변화 없음 | 같음 | 변화 없음 | 0 → 0 |
| 표본 face / neck (470,560.5)(512,562.5)(556,548.5) | NeckHead | 9.21 → 0.12 | – | – | – | – |
| 표본 clothes / neck (470,639.5)(512,639.5)(558.5,650) | Collar | 44.55 → 16.26 | – | – | – | – |
| 표본 body / neck (486,660.5)(538,660.5) | Collar | 39.90 → 7.53 | – | – | – | – |
| 뿌리 띠 표본 hair_front / face (512,120.5)(452,168.5)(512,182.5)(574,200.5) | HairRoot | 4.22 → 3.47 | – | – | – | – |
| 뿌리 띠 표본 hair_side_L_2 / hair_back (650.5,240)(690.5,280) | HairRoot | 7.57 → 7.14 | – | – | – | – |
| 표본 hat / hair_front (512,150.5) | 앞머리 쪽 HairRoot | 2.17 → 0.41 | – | – | – | – |

- 커버리지(경계점 1/2 × 94포즈 = 221,370 검사): `lowerMisses` 3,742 → 3,734, `holes` 0 → 0.
- 결합 요구량이 비결합보다 작은 이음새(face/neck, clothes/neck, clothes/hair_back)는 비결합 값이 채택되어 숨은 영역이 그대로이고, 결합 요구량이 큰 `neck`/`hair_back`은 결합 값이 채택됐다. 요구량은 결합/비결합 `deform`에서 각각 자동으로 측정된다(상수 복제 없음).
- 머리카락 뿌리 결합이 fixture 이음새 표에서 0인 이유: 이음새 경계(앞머리 끝 t≈0.47, 옆머리·뒷머리 경계 t≥0.32)가 모두 뿌리 띠(t<0.25) 밖이다. 뿌리 띠 안 표본에서만 줄어든다.

### 표 2 — 이음새에서 먼 기준점의 최대 변위 (결합 전 = 결합 후)

94포즈 **모든 포즈**에서 결합/비결합 위치 차가 `1e-9`px 미만이다(최댓값만 같은 것이 아님). 모션 축소 없음. 감쇠 0.06·hh·옷깃 수정 후 다시 측정했고 값은 같다.

| 기준점 | 레이어 | 위치 | 최대 변위 px (전) | 최대 변위 px (후) |
| --- | --- | --- | ---: | ---: |
| 정수리 | face | (512,120) | 341.296 | 341.296 |
| 얼굴 중심 | face | (512,400) | 181.997 | 181.997 |
| 코끝 | face | (512,440) | 157.075 | 157.075 |
| 앞머리 끝 | hair_front | (510,324) | 227.580 | 227.580 |
| 앞머리 끝 2 | hair_front_2 | (470,368) | 208.781 | 208.781 |
| 옆머리 끝 R | hair_side_R | (330,715) | 173.568 | 173.568 |
| 옆머리 끝 L | hair_side_L_2 | (694,755) | 193.741 | 193.741 |
| 뒷머리 끝 | hair_back | (512,855) | 202.119 | 202.119 |
| 몸통 중심 | clothes | (512,800) | 26.038 | 26.038 |
| 몸통 중심 | body | (512,820) | 24.080 | 24.080 |
| 어깨 R | clothes | (330,700) | 41.299 | 41.299 |
| 어깨 L | clothes | (694,700) | 41.299 | 41.299 |
| 홍채 중심 R | eyeR_iris | (400,360) | 230.537 | 230.537 |
| 홍채 중심 L | eyeL_iris | (624,360) | 230.537 | 230.537 |
| 눈썹 R | browR | (400,300) | 261.923 | 261.923 |
| 입 | mouth_upper | (512,488) | 127.491 | 127.491 |

- 머리카락 레이어 자신의 정수리 점은 기준점에서 뺐다. 그 점은 뿌리↔두피 이음새 자체이며, 결합 전에는 `angleY`로 머리가 내려갈 때 월드 고정 `root_y` 때문에 정수리 머리카락이 두피에서 최대 ≈1px 흔들렸다(결합 후 두피를 따름). 같은 위치의 `face` 정수리는 위 표대로 그대로다.

### 노출 맵 (1024px 합성 fixture, 결합 전 → 후)

렌더: 레이어별 ArtMesh(`core::art_mesh_for_part`, mask∪fillMask, 밀도 14)를 `deform`으로 변형해 아래에서 위로 무게중심 보간 래스터화, `render.ts` 클리핑 적용, 불투명 기준 alpha>127. 점 역변환 근사는 쓰지 않았다(커버리지 검사는 T13a의 경계점 띠 방식 그대로).

| 레이어 | 숨은 px | 노출 px | 노출 빈도 평균 | 최대 | 최대 노출 깊이 px |
| --- | ---: | ---: | ---: | ---: | ---: |
| neck | 32,441 → **32,441** | 3,587 → **986** | 0.088 → 0.105 | 0.851 → 0.947 | 36.5 → **23.7** |
| hair_back | 230,097 → 230,097 | 63,230 → 63,255 | 0.069 → 0.070 | 0.904 → 0.904 | 409.6 → 357.9 |
| body (가시 픽셀 없음) | 135,688 → 135,688 | 11,898 → 10,908 | 0.148 → 0.153 | 0.926 → 0.926 | n/a |
| clothes | 12,501 → 12,501 | 12,431 → 12,431 | 0.242 → 0.241 | 0.840 → 0.840 | 78.9 → 78.9 |
| face | 89,560 → 89,560 | 14,761 → 14,761 | 0.067 → 0.067 | 1.000 → 1.000 | 103.3 → 103.3 |
| 합계 | 591,686 → 591,686 | 120,610 → 117,044 | – | – | 409.6 → 357.9 |

- 구멍(래스터, 갇힌 배경): 전체 포즈 합 1,851 → 1,851px, 포즈당 최대 585 → 585px, 구멍 있는 포즈 37 → 37, 성분 합 554 → 554. 휴지 갇힌 배경 0px, 휴지 미피복 전경 226px(메쉬 삼각형 제거로 생기는 휴지 틈).
- 숨은 면적은 모든 레이어에서 결합 전과 같다(목 32,441px 포함). 결합 후 숨은 텍셀 ⊇ 결합 전 숨은 텍셀을 테스트가 확인한다.

### 앞머리 아래 눈썹·윗눈꺼풀 판정 (T13a 미해결 항목) → 숨은 영역 추가

같은 결합 파이프라인에서 5b 블록을 끈 경우(T13a 상태)와 켠 경우. "추가 영역" = 블록이 더한 숨은 텍셀. 리뷰 수정 후 다시 측정했고 값은 같다.

| 레이어 | 숨은 px (끔 → 켬) | 추가 영역 px | 끔: 추가 영역에서 얼굴 피부 노출 px / 최대 빈도 / 최대 깊이 | 켬: 추가 영역에서 획 노출 px / 평균 빈도 / 최대 빈도 / 최대 깊이 | 레이어 최대 노출 깊이 (끔 → 켬) | 머리카락→획 `lowerMisses` (끔 → 켬) |
| --- | --- | ---: | --- | --- | --- | --- |
| eyeR_upper | 1,089 → 2,165 | 1,076 | 34 / 0.255 / 68.9 | 152 / 0.093 / 0.723 / 68.4 | 14.2 → 68.4 | 70 → 70 |
| eyeL_upper | 640 → 1,062 | 422 | 10 / 0.160 / 9.0 | 12 / 0.066 / 0.160 / 14.7 | 14.2 → 14.7 | 29 → 29 |
| browR | 1,437 → 1,839 | 402 | 17 / 0.681 / 23.5 | 136 / 0.079 / 0.585 / 33.1 | 34.0 → 34.0 | 63 → 63 |
| browL | 1,437 → 1,816 | 379 | 44 / 0.691 / 23.5 | 199 / 0.065 / 0.521 / 44.0 | 70.7 → 70.7 | 81 → 81 |

- 판정: 추가 영역이 포락선에서 실제로 드러나고(최대 빈도 0.16–0.72, 깊이 최대 68px) 블록이 없으면 그 자리에 피부가 보이므로 **추가한다**. 블록은 역할 기반(`Brow`, `Eye`의 `_upper`)이고 이음새 측정 요구량(`table.need`)만 쓰며 폴백 상수는 없다. 텍스처는 기존 경로(눈썹 = 팔레트 눈썹색, 윗눈꺼풀 = 해석적 속눈썹 호 + 피부)가 칠한다.
- `lowerMisses`가 그대로인 이유(진단 확인): 남은 위반 표본은 획 중심선에서 8–20px 떨어진 위치(`browRY=-1`에서 눈썹 위 이마, `eyeR=0`에서 감기는 눈꺼풀 위)이고 피부가 맞는 자리다. 가는 획 레이어는 T13a 계약("하위가 노출 띠 전체를 덮음")으로 판정할 수 없으므로 T12에서 획 레이어 판정을 따로 둔다.
- 한계: 끔/켬 비교는 레이어별 휴지 좌표 텍셀 기준이라 눈썹 이동(`browY`)과 얼굴 이동의 차이를 무시한 근사다.

### 표 3 — 원화 3종 결합 전/후 (릴리스 CLI, 공유 포락선 94포즈)

입력은 기존 검증과 같다: Himikan = `artifacts/design-review/input.png`(PSD 합성 683×1024), Rritteang = `ref/ritteang/Rritteang.png`(1254×1254), Sutteang = `ref/sutteang/2d_.png`(1024×1024). 세 원화 모두 ONNX 모델로 자동 랜드마크(`onnx-auto`), 24파츠, 결합 전/후 파츠 ID 동일. 실행 시간 7–30초.

| 원화 | 숨은 px (전 → 후) | 노출 px | 최대 노출 깊이 px | 갇힌 구멍 px (포즈당 최대) | 갇힌 구멍 px (94포즈 합) | 갇힌 구멍 성분 합 | 이음새 `holes` | 이음새 `lowerMisses` | `neck` 숨은 px | `neck` 노출 px / 최대 깊이 px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Himikan | 408,423 → 408,423 | 85,754 → 83,266 | 407.7 → 407.7 | 775 → 775 | 17,615 → 17,600 | 687 → 692 | 52 → 52 | 7,187 → 7,177 | 8,618 → 8,618 | 964 → 127 / 45.0 → 36.0 |
| Rritteang | 1,588,056 → 1,591,612 | 415,951 → 412,271 | 840.4 → 840.4 | 26,510 → 26,510 | 856,006 → 856,047 | 12,875 → 12,895 | 11,907 → 11,907 | 79,173 → 78,905 | 43,309 → 46,865 | 6,275 → 4,406 / 46.3 → 35.2 |
| Sutteang | 1,125,101 → 1,128,338 | 349,086 → 335,176 | 370.5 → 370.5 | 521 → 521 | 7,255 → 7,255 | 229 → 229 | 1,974 → 1,974 | 46,395 → 46,261 | 42,341 → 45,578 | 8,170 → 2,905 / 57.6 → 41.0 |

- 숨은 영역은 세 원화 모두 결합 전보다 줄지 않는다(Rritteang·Sutteang 목은 결합 쪽 요구량이 더 큰 이음새 때문에 +3,556·+3,237px). 최대 노출 깊이는 `hair_back`/`body`의 넓은 템플릿 숨은 영역에서 나오며 결합과 무관하다(전후 동일). 휴지 기준: 갇힌 배경 Himikan 310 / Rritteang 35,386 / Sutteang 36px, 휴지 미피복 전경 84 / 505 / 49px.

결합 대상·관련 이음새(원화, 결합 전 → 후; 언더랩 = 결합 전 → 채택 (결합 / 비결합, 채택)):

| 원화 | upper / lower | 결합 (up/low) | 최대 상대 변위 px | 필요 언더랩 px | `lowerMisses` | `holes` |
| --- | --- | --- | ---: | --- | ---: | ---: |
| Himikan | clothes / neck | Collar (0.85 / 0.85) | 17.1 → 0.6 | 15 → 15 (4 / 15, 비결합) | 2 → 0 | 0 → 0 |
| Himikan | neck / body | Collar (0.50 / 0.50) | 28.6 → 22.1 | 35 → 35 (25 / 35, 비결합) | 0 → 0 | 0 → 0 |
| Himikan | body / hair_back | 없음 | 54.4 → 53.0 | 58 → 58 (56 / 58, 비결합) | 20 → 11 | 0 → 0 |
| Himikan | neck / hair_back | 없음 | 22.7 → 23.1 | 20 → 22 (22 / 20, 결합) | 0 → 0 | 0 → 0 |
| Rritteang | face / neck | NeckHead (0.00 / 0.36) | 21.4 → 16.2 | 25 → 40 (40 / 25, 결합) | 93 → 40 | 0 → 0 |
| Rritteang | clothes / neck | Collar (0.12 / 0.12) | 71.8 → **72.3** | 106 → 117 (117 / 106, 결합) | 625 → 537 | 0 → 0 |
| Rritteang | face / clothes | 없음 | 101.1 → 101.1 | 133 → 133 (같음) | 1,116 → 1,113 | **0 → 0** (수정 전 0 → 12) |
| Rritteang | neck / hair_back | 없음 | 38.0 → 36.5 | 52 → 52 (44 / 52, 비결합) | 0 → 0 | 0 → 0 |
| Sutteang | neck / body | Collar (0.49 / 0.39) | 60.4 → 51.6 | 79 → 79 (71 / 79, 비결합) | 454 → 397 | 0 → 0 |
| Sutteang | clothes / neck | Collar (0.51 / 0.52) | 63.7 → **65.8** | 94 → 102 (102 / 94, 결합) | 540 → 484 | 0 → 0 |
| Sutteang | face / clothes | 없음 | 69.6 → 69.6 | 80 → 80 (같음) | 550 → 550 | 0 → 0 |
| Sutteang | neck / hair_back | 없음 | 38.6 → 37.6 | 32 → 35 (35 / 32, 결합) | 0 → 0 | 0 → 0 |

- **미해결**: Rritteang·Sutteang `clothes`/`neck`은 결합 후 최대 상대 변위가 0.5·2.1px **늘었다**(구멍 0 유지, `lowerMisses`는 감소, 요구량은 결합 쪽 채택으로 늘어 숨은 영역도 늘어남). 이 원화들은 옷깃 선이 턱 아래 ≈0.09·hh라 목 상단 감쇠(0.06·hh)와 목 하단 범위(0.42·hh) 사이 간격이 좁고, 옷깃 선 위치를 랜드마크로 알 수 없다. 옷깃 선을 이음새 기하에서 읽어 목 하단 범위를 정하는 것은 후속 과제다.
- 머리카락 뿌리 결합(`HairRoot`) 이음새는 원화에서도 경계 평균 가중치가 대부분 0–0.06이라 값이 바뀌지 않았다(예외: `hair_front`/`hair_back` Himikan 0.82/0.86, Sutteang 0.42/0.53 — 두 레이어가 함께 머리 장 쪽으로 결합되어 상대 변위 4.2px, 17.0px 그대로).
- 원화의 눈썹·윗눈꺼풀 숨은 영역도 포락선에서 드러난다: 최대 노출 빈도 Himikan 0.29–0.97, Rritteang 0.99–1.00, Sutteang 0.80–0.95, 최대 깊이 11.1–72.3px.
- 래스터 구멍(갇힌 배경)은 세 원화 모두 결합 전후가 거의 같고 머리카락 끝 사이 자연 배경이 흔들림 중 갇히는 오탐이 섞여 있다(히트맵 자홍색 가시). 결함 판정에는 이음새 `holes`를 쓴다.
- 이 표는 CLI 래스터 기록이며 Windows 앱 모션·Cubism 가져오기·노출 영역 원본감(⑥, T14)을 검증하지 않았다.

### 표 4 — 원화별 이음새 구멍 상위 이음새 (T9/T11 입력, 기록만)

결합 후 런타임 기준(결합 전과 이음새별 구멍 수 동일 — 결합 전부터 있던 구멍). 구멍이 있는 이음새가 5개 미만이면 있는 것만 적었다(Himikan 3개). "대표 포즈 그룹"은 이음새의 구멍을 포즈 그룹별로 센 상위 항목, 괄호 안은 구멍이 가장 많은 단일 포즈. 추정 원인은 수치(깊이·포즈 그룹·레이어 관계)로 추정한 것이며 원화 육안 확인은 하지 않았다.

| 원화 | upper / lower | 경계 px | 구멍 (전 → 후) | `lowerMisses` | 대표 포즈 그룹 (최다 포즈) | 구멍 최대 깊이 px / 필요 언더랩 px | 추정 원인 |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| Himikan | hair_side_R / clothes | 590 | 28 → 28 | 2,279 | hairPhase 10, headBodyOpposite 7, axis 6 (`angleX=30` 6) | 84.5 / 112 | 숨은 영역 부족: 어깨 밖으로 늘어진 옆머리 아래를 의상 블록이 어깨 폭·옷깃 높이 제한(`shoulder_hw·1.05`, `local_top_v`) 때문에 요구량만큼 채우지 못함 |
| Himikan | hair_side_L / clothes | 545 | 21 → 21 | 2,266 | hairPhase 12, headHairBody 7 (`hairPhase3:120` 12) | 84.9 / 127 | 숨은 영역 부족(위와 같음) |
| Himikan | hair_front / eyeL_upper | 98 | 3 → 3 | 324 | eye 2, axis 1 (`eyesClosedSmile` 2) | 1.5 / 39 | 경계 표본 오탐 추정(깊이 1.5px, 감는 눈꺼풀 가장자리 반올림) |
| Rritteang | hair_side_R / clothes | 2,796 | 7,388 → 7,388 | 21,799 | headBodySame 3,095, headBodyOpposite 2,086, headHairBody 1,627 (`same3:head+-+/breath1` 1,337) | 177.5 / 348 | 숨은 영역 부족: 머리+몸 같은 방향 회전에서 옆머리 띠가 의상 실루엣 밖까지 크게 이동(요구량 348px), 의상 블록 제한으로 못 채움 |
| Rritteang | hair_side_L / clothes | 5,658 | 3,295 → 3,295 | 24,758 | headBodySame 2,889, headHairBody 297 (`same3:head---/breath1` 1,364) | 318.5 / 378 | 숨은 영역 부족(위와 같음, 깊이 318px) |
| Rritteang | eyeR_lower / hair_back | 375 | 765 → 765 | 3,581 | hairPhase 455, headHairBody 156, axis 154 (`hairPhase3:180` 68) | 17.5 / 112 | 분할 오류 추정: 아래 눈꺼풀이 뒷머리와 직접 경계를 공유(기울어진 얼굴·안경 주변에서 얼굴 안에 뒷머리 가시 조각) — 뒷머리 흔들림 때 그 조각 아래가 빔 |
| Rritteang | eyeL_lower / hair_back | 125 | 357 → 357 | 942 | hairPhase 187, axis 96 (`hairPhase3:0` 54) | 7.5 / 150 | 분할 오류 추정(위와 같음) |
| Rritteang | clothes / hair_back | 888 | 82 → 82 | 1,053 | headBodySame 54, headHairBody 24 (`same3:head+--/breath1` 17) | 125.5 / 374 | 숨은 영역 부족: 뒷머리 블록이 `effective_back_max_v`·브리지 폭 제한으로 의상 아래 깊은 띠를 못 채움 |
| Sutteang | hair_side_L / body | 869 | 827 → 827 | 6,885 | headBodySame 738, headBodyOpposite 59 (`same3:head---/breath1` 421) | 124.5 / 287 | 숨은 영역 부족: 몸통 블록이 `shoulder_hw·0.88` 폭·`face_sdf` 조건으로 옆머리 아래를 제한 |
| Sutteang | hair_side_R / clothes | 4,247 | 614 → 614 | 15,620 | headBodySame 614 (`same3:head+-+/breath1` 373) | 131.5 / 220 | 숨은 영역 부족(의상 블록 제한) |
| Sutteang | hair_side_R / body | 602 | 328 → 328 | 10,439 | headBodySame 328 (`same3:head+-+/breath1` 117) | 31.5 / 227 | 숨은 영역 부족(몸통 블록 제한) |
| Sutteang | face / hair_back | 748 | 204 → 204 | 541 | hairPhase 84, headHairBody 68, axis 52 (`hair=1` 18) | 18.8 / 27 | 숨은 영역 부족: 뒷머리 블록의 얼굴 중심 제외(`is_center_face`)로 얼굴 가장자리 아래 뒷머리 띠가 빔 |
| Sutteang | hair_side_L / hair_back | 585 | 1 → 1 | 11 | hairPhase 1 (`hairPhase3:300` 1) | 9.5 / 55 | 숨은 영역 부족(단일 표본, 미미) |

- 원화 전체 구멍: Himikan 52 / 351,842 검사, Rritteang 11,907 / 1,446,754, Sutteang 1,974 / 1,310,548. 상위 이음새 대부분은 "요구량은 측정되지만 템플릿 블록의 실루엣·폭 조건이 요구량만큼 채우지 못함" 유형이라 T9(합성)·T11(수동 보정)에서 블록 조건을 노출 띠 기준으로 바꾸는 것이 먼저다. 이음새 `holes`는 역변환 기반 검사라 래스터 오탐 유형(머리카락 끝 사이 배경)은 여기에 해당하지 않는다.

## T14 숨은 영역 원본감 평가 하네스 — 현 합성기 기준선 (2026-11-18)

- 실행: `cd src-tauri && cargo test` (lib 56/56 통과 — 신규 7: `rig::fidelity` 6, `analysis::templates::tests::t14_eval_holdout_is_inert_on_the_product_path_and_keeps_the_partition`), `cargo clippy --all-targets -- -D warnings` (통과), `npm test` (43/43 통과), `npx tsc --noEmit` (통과). Windows 앱·Cubism 검증 없음(이 절은 CLI 수치 기록).
- **측정만 했다.** 합성기(`templates.rs` 숨은 텍스처 생성)·숨은 영역 크기·`deform`·`PARAM_DEFS`·모션 범위는 바꾸지 않았다. 홀드아웃·정답 PSD 결과는 결과 프로젝트·숨은 영역 생성에 되먹이지 않는다(평가 전용 CLI 플래그에서만 계산).
- **임계값은 정하지 않았다.** 아래 수치는 기준선이며 품질 보증이 아니다. 단일 종합 점수도 만들지 않았다(근거: 아래 "육안 대조"에서 ΔE 평균만으로는 선화 손실이 과소평가되는 사례 확인).

### 실행 방법

```sh
# 1) 정답 레이어 추출 (평가 전용 venv: psd-tools·numpy·pillow; tools/**/.venv 는 Git 제외)
uv venv tools/evaluation/.venv && uv pip install --python tools/evaluation/.venv/Scripts/python.exe psd-tools numpy pillow
tools/evaluation/.venv/Scripts/python.exe -I tools/evaluation/himikan_truth.py          # → artifacts/t14/himikan-truth/
# 2) 평가 실행 (릴리스 CLI)
flat2rig-cli --eval-holdout --eval-truth artifacts/t14/himikan-truth artifacts/design-review/input.png artifacts/t14/himikan
flat2rig-cli --eval-holdout ref/ritteang/Rritteang.png artifacts/t14/ritteang
flat2rig-cli --eval-holdout ref/sutteang/2d_.png       artifacts/t14/sutteang
# 3) 표 생성 (리포트 전용)
tools/evaluation/.venv/Scripts/python.exe -I tools/evaluation/t14_report.py artifacts/t14/*/stats.json
```

입력은 T13b 표 3과 같다(Himikan 683×1024 `artifacts/design-review/input.png`, Rritteang 1254×1254, Sutteang 1024×1024, 모두 `onnx-auto` 랜드마크). 실행 시간(평가 포함) 11 / 26 / 21초. 결과: `stats.json` `fidelity` 키(`spec`·`holdout`·`groundTruth`), 비교 이미지 `artifacts/t14/<원화>/fidelity/{holdout,truth}_<part>.png`(왼쪽 정답 · 가운데 합성 · 오른쪽 ΔE00 히트맵: 남색 0 → 노랑 10 → 빨강 ≥20, 회색 = 가시 맥락). 전체 표는 `artifacts/t14/report.md`(Git 제외).

### 기본 실행 출력 불변 확인

- 테스트 `t14_eval_holdout_is_inert_on_the_product_path_and_keeps_the_partition`: 래퍼 없이 / 빈 홀드아웃 래퍼 안 / 래퍼 종료 후 세 번의 `synthesize_template_bank` 출력(파츠 ID·`masks`·`fillMasks`·텍스처 PNG 바이트)이 같고, 실제 홀드아웃은 분할(`masks`·`fillMasks`)과 홀드아웃 밖 가시 픽셀 텍스처를 바꾸지 않으며 홀드아웃 픽셀은 덮는 파츠 색을 복사하지 않는다.
- 해시 비교: 작업 전 HEAD(`2c53c3e`) 릴리스 CLI와 T14 릴리스 CLI를 평가 플래그 없이 원화 3종에 실행해 출력 파일을 비교(`tools/evaluation/compare_cli_outputs.py`; `stats.json`은 `elapsedMs`만 제외). 세 원화 모두 **34/34 파일 동일**. 비-stats 트리 sha256: Himikan `37b6e08e…f4ba5c3`, Rritteang `8a6b0305…5ff94e`, Sutteang `96f98475…675442f`.
- 홀드아웃 실행에서도 분할이 같다: 세 원화 모두 `masksIdentical=true`, `fillMasksIdentical=true`.

### 하네스 정의 (`src-tauri/src/rig/fidelity.rs`, 버전 `t14-v1`)

- **노출 밴드 안에서만 계산**한다. 모든 지표 블록 옆에 표본 수(`samples`), 노출 면적, 노출 빈도(평균/최대)를 기록하고, 노출되지 않는 숨은 영역은 `unexposedHiddenPx`로 보고만 한다. 숨은 영역을 줄여서 좋아지는 지표는 두지 않았다(평균은 표본당 값이며 표본 수와 함께만 읽는다 — 단위 테스트가 밴드 일부만 쓰면 평균이 달라짐을 고정).
- **정답 비교(Himikan)**: T13b `ExposureMap`이 포락선 94포즈 중 1번 이상 드러낸 숨은 텍셀 × 정답 레이어 알파 ≥128. 가중치 = 그 텍셀의 노출 빈도. 그룹 = 휴지 상태에서 그 텍셀을 덮는 상위 파츠(이음새). 정답 레이어가 없는 노출 텍셀은 `noTruthPx`(합성 숨은 영역이 PSD 파츠 밖까지 간 형태 불일치)로 따로 세고, 그 자리를 이 파츠 이하(flat2rig 순서) 파츠 정답의 합성과 비교한 값을 `noTruthVisualMetrics`로 둔다. 아래 정답도 없으면 `noTruthBackgroundPx`.
- **자기 지도 홀드아웃(원화 3종)**: T13 이음새 요구량(`SeamRequirement`, CLI `seams[]`)마다 하위 파츠 가시 픽셀 중 상위 파츠 가시 마스크로부터 chamfer 거리 ≤ 그 이음새 `maxExposureDepthPx`인 밴드(여러 이음새가 겹치면 가까운 상위 파츠). 깊이 < 1px 이음새는 건너뛴다(`skipped`). 밴드는 `templates::with_eval_holdout`(스레드 로컬, 평가 전용)으로 **숨은 텍셀과 같은 분기**를 탄다: 색 필드 시드에서 제외, 텍스처 단계에서 `fillMask` 텍셀과 같은 분기, 원본 픽셀 자리는 덮는 상위 파츠의 가장 가까운 픽셀 색으로 치환, 팔레트·홍채 적합은 치환된 원본에서 다시 추정(처음 구현은 눈썹 팔레트가 홀드아웃 픽셀에서 추정돼 눈썹 ΔE 0.41로 누출이 보였고 수정했다). 분할(피부 필드를 쓰는 `face_detail` 분리 포함)은 제품 경로 그대로다(처음 구현은 피부 필드 시드를 바꿔 `face_detail` 분할이 달라졌고 `masksIdentical=false`가 나와 수정했다). 가중치 = 같은 상위 파츠 아래 하위 파츠 숨은 텍셀의 깊이별 평균 노출 빈도를, 밴드 픽셀의 같은 깊이(상위 파츠로부터 거리)에 대응(`mirror-profile`; 숨은 텍셀이 없으면 `none` → 가중값 없음).
- **지표**: ΔE2000(sRGB D65→CIELAB) 평균·p95, 팔레트 일치율(파츠 가시 픽셀 k-means k≤8, 맥락 64px 미만이면 `null`), 이음새 단차(4-인접 밴드/가시 쌍의 ΔE00 단차를 합성·정답으로 각각 구한 뒤 |차|의 평균 = 그래디언트 불연속), 선화 끊김(정답 기준 L* < 가시 중앙값 − 25인 선 성분 중 가시 쪽 선 픽셀에서 이어져 3px 고리로 들어오는 획; 합성에서 50% 미만만 선으로 남으면 끊김), 질감(5×5 L* 국소 표준편차의 합성/정답 차). 각 지표에 노출 빈도 가중값을 같이 둔다.
- ΔE2000 검증: Sharma·Wu·Dalal(2005) 표 1의 21쌍과 1e-4 안에서 일치(대칭 포함). Python 쪽(정답 추출의 왕복 검증) 벡터 구현도 같은 수식이다.
- **전 층 홀드아웃(`wholeLayerHeldOut`)**: 밴드가 가시 픽셀의 95% 이상이면 가시 맥락이 남지 않아 이음새 밴드의 대리가 아니므로 표 3으로 분리한다. Rritteang·Sutteang은 이음새 최대 노출 깊이가 커서(예: Rritteang `hair_side_L`/`clothes` 300.7px, `clothes`/`hair_back` 297.3px) `clothes`·`face`·`neck`·`hair_back`이 통째로 홀드아웃된다(제한 (a)).

### 정답 세트 (Himikan) — 대응표 `tools/evaluation/himikan-layer-map.v1.json`

- PSD 2000×3000, sha256 `d5fc8507…0924ff` 고정. 대응 파츠 **21개**(정답 이미지 21장): `hair_back`, `neck`, `clothes`(카라 뒤·몸통·카라 앞·리본 묶음), `clothes_inner`(리본아래<·>·리본위), `face`(귀<·>·얼굴·홍조 곱하기), `eyeR/L_white`, `eyeR/L_iris`(눈동자·동공·하이라이트, 자체 알파), `eyeR/L_upper`(눈꼬리·눈꺼풀2·눈꺼풀·속눈썹2·속눈썹1), `eyeR/L_lower`(눈언더), `browR/L`, `mouth_inner`(입안·혀·윗니), `mouth_upper`, `mouth_lower`, `hair_side_R/L`, `hair_front`(앞머리<·>·중앙·더듬이).
- 정답 없는 파츠 3개: `body`(PSD에 맨 몸통이 없음), `clothes_hands`(170px 분할 조각), `face_detail`(숨은 영역 없음).
- 제외 레이어와 사유: 입 가이드 6장(숨김 가이드), `쌍커풀<`·`쌍커풀>`(별도 쌍꺼풀 획, flat2rig 대응 파츠가 애매), `입술옆 가리개<`·`>`(입꼬리를 가리는 피부색 조각, 얼굴/입술 어느 쪽도 아님), `코`(4×15px, 숨은 영역 없는 `face_detail` 안).
- L/R: `<` = 화면 왼쪽 = 캐릭터 R. 스크립트가 기호가 아니라 레이어 중심 x로 확인(오류 0).
- 좌표 변환과 왕복 검증: 면적 매핑 `x' + 0.5 = (x + 0.5)·683/2000`, `y' + 0.5 = (y + 0.5)·1024/3000`(PIL BOX, 사전곱 RGBA). (a) 스크립트 재합성(클리핑·곱하기 포함) vs psd-tools 합성 2000×3000: 불투명 2,562,504px ΔE00 평균 0.018, p95 0.00, 최대 1.75, 알파 최대차 1. (b) 레이어 알파 무게중심을 전방 사상한 값 vs 리샘플 레이어 무게중심: 최대 0.17px, 평균 0.06px. (c) 리샘플 레이어 재합성 vs CLI 입력 `input.png`: 불투명 297,881px ΔE00 평균 0.30, p95 1.38(최대 18.6은 가장자리 반투명 픽셀), 알파 차 평균 0.13, p99 1. PSD·추출 레이어는 `artifacts/t14/himikan-truth/`(Git 제외)에만 둔다.

### 표 1 — Himikan 정답 비교 (노출된 숨은 텍셀, 파츠 전체)

ΔE·σ 열의 괄호는 노출 빈도 가중값. 단차 = 합성 / 정답 → |Δ| 평균. σ = 국소 L* 표준편차 합성 / 정답 → |Δ|.

| 파츠 | 숨은 / 노출 / 정답 없음(bg) px | 표본 | 평균 빈도 | ΔE00 평균 / p95 (가중) | 팔레트 일치 (가중) | 이음새 단차 | 선화 끊김/교차 | σL* (가중 \|Δ\|) |
| --- | --- | ---: | ---: | --- | --- | --- | ---: | --- |
| hair_back | 155,730 / 60,791 / 50,009 (50,009) | 10,782 | 0.09 | 6.67 / 26.28 (7.90 / 29.56) | 0.77 (0.72) | 26.73 / 19.33 → 12.30 | 129/129 | 3.43 / 6.61 → 3.95 (4.29) |
| neck | 8,618 / 127 / 1 (0) | 126 | 0.23 | 6.40 / 13.88 (7.79 / 25.42) | 0.52 (0.39) | 3.35 / 1.15 → 2.24 | 0/0 | 1.58 / 1.61 → 1.42 (2.26) |
| clothes | 36,363 / 5,718 / 10 (10) | 5,708 | 0.41 | 2.98 / 14.38 (3.30 / 15.95) | 0.78 (0.79) | 12.84 / 12.97 → 3.35 | 4/4 | 2.93 / 3.43 → 0.87 (1.03) |
| face | 26,094 / 7,083 / 554 (0) | 6,529 | 0.06 | 2.34 / 5.29 (2.61 / 6.99) | 0.91 (0.91) | 1.28 / 3.03 → 2.23 | 0/0 | 0.25 / 1.05 → 0.87 (1.56) |
| eyeR_white | 3,875 / 967 / 331 (0) | 636 | 0.05 | 1.02 / 4.82 (1.04 / 3.52) | 0.92 (0.96) | 17.22 / 17.83 → 0.89 | 0/0 | 2.61 / 2.63 → 0.21 (0.36) |
| eyeL_white | 3,192 / 818 / 205 (0) | 613 | 0.05 | 1.07 / 4.78 (0.98 / 4.15) | 0.88 (0.91) | 13.57 / 13.84 → 0.88 | 0/0 | 2.16 / 2.16 → 0.20 (0.37) |
| eyeR_iris | 1,711 / 845 / 526 (0) | 319 | 0.06 | 13.42 / 16.99 (14.16 / 25.69) | 0.97 (0.93) | 16.12 / 2.42 → 13.70 | 0/1 | 2.48 / 1.68 → 1.05 (1.01) |
| eyeL_iris | 1,731 / 939 / 617 (0) | 322 | 0.06 | 10.66 / 15.28 (10.22 / 15.65) | 0.98 (0.97) | 11.39 / 2.68 → 8.71 | 0/1 | 2.85 / 1.32 → 1.72 (1.93) |
| eyeR_upper | 528 / 227 / 170 (0) | 57 | 0.12 | 38.57 / 53.13 (31.73 / 53.13) | 0.04 (0.06) | 27.90 / 4.24 → 23.66 | 0/0 | 17.39 / 2.37 → 15.02 (13.34) |
| eyeL_upper | 504 / 228 / 148 (0) | 80 | 0.28 | 33.82 / 53.13 (19.73 / 53.13) | 0.11 (0.18) | 33.36 / 6.21 → 27.71 | 0/0 | 19.64 / 2.57 → 17.07 (13.64) |
| browR / browL | 715 / 310 / 310 · 879 / 475 / 475 | 0 | – | 표본 0 (노출 텍셀 전부 PSD 눈썹 밖) | – | – | – | – |
| mouth_inner | 62 / 62 / 0 | 62 | 0.18 | 14.48 / 27.87 (15.32 / 28.73) | 0.90 (0.83) | 27.80 / 31.73 → 10.10 | 0/0 | 17.80 / 18.43 → 4.10 (4.18) |
| hair_side_R | 1,187 / 383 / 228 (0) | 155 | 0.19 | 5.21 / 32.05 (5.95 / 31.83) | 0.70 (0.74) | 1.52 / 1.42 → 1.13 | 0/0 | 1.99 / 6.69 → 4.95 (5.66) |
| hair_side_L | 1,521 / 540 / 213 (0) | 327 | 0.18 | 6.59 / 43.57 (8.79 / 49.56) | 0.74 (0.66) | 2.73 / 2.17 → 1.88 | 1/1 | 1.58 / 7.87 → 6.57 (6.33) |

숨은 텍셀이 없어 측정 대상이 없는 파츠: `clothes_inner`, `eyeR/L_lower`, `mouth_upper`, `mouth_lower`, `hair_front`.

정답 없는 노출 텍셀(`noTruthVisualMetrics`, ΔE00 평균 / p95): `face` 554px 22.75 / 26.94(PSD 얼굴 밖까지 피부를 칠함), `eyeR_iris` 526px 46.73 / 53.52, `eyeL_iris` 617px 61.81 / 69.00(PSD 홍채 밖까지 홍채를 그림 — PSD는 그 자리가 흰자/얼굴), `eyeR_upper` 170px 18.35 / 62.17, `eyeL_upper` 148px 14.42 / 57.27, `hair_side_R/L` 228 / 213px 7.57 / 30.85, 5.84 / 33.81, `browR/L` 310 / 475px 0.41 / 0.41, 0.98 / 0.41(해석 보류, 제한 (f)). `hair_back` 50,009px는 PSD 어느 레이어도 없는 자리(배경)라 비교 대상이 없다 — flat2rig 뒷머리 숨은 영역이 PSD 뒷머리(683 기준 y≈571에서 끝남)보다 훨씬 아래·옆까지 있다.

이음새별(정답, 표본 ≥50) 주요 값: `hair_back` ← `hair_front` 985px ΔE 11.15 / 42.77, 단차 |Δ| 39.55, 끊김 41/41 · ← `hair_side_R` 3,817px 5.89 / 24.52, 끊김 44/44 · ← `hair_side_L` 3,779px 4.87 / 23.21, 끊김 44/44 · ← `neck` 597px 13.83 / 24.72, 팔레트 0.03(목 뒤 뒷머리를 어둡게 칠하지만 정답은 밝은 머리색). `clothes` ← `hair_side_R` 2,704px 2.34 / 12.50, ← `hair_side_L` 2,056px 2.65 / 16.47, ← `clothes_inner` 903px 4.43 / 13.49. `face` ← `hair_side_R/L` 341 / 304px 7.27 / 35.76, 8.26 / 37.48(단차 합성 1.02 vs 정답 8.84 — 정답에는 머리카락 그림자·선이 있음), 눈·입 아래 0.00–3.58 / 0.00–6.68.

### 표 2 — 자기 지도 홀드아웃: 이음새 밴드 (가시 맥락이 남는 파츠, 홀드아웃 < 95%)

| 원화 | 파츠 | 홀드아웃 px (가시 대비) | 평균 빈도 | ΔE00 평균 / p95 (가중) | 팔레트 일치 (가중) | 이음새 단차 | 선화 끊김/교차 | σL* (가중 \|Δ\|) |
| --- | --- | --- | ---: | --- | --- | --- | ---: | --- |
| Himikan | face | 5,532 (32%) | 0.05 | 0.81 / 2.69 (1.14 / 3.57) | 0.90 (0.83) | 0.20 / 0.10 → 0.18 | 0/0 | 0.15 / 0.35 → 0.25 (0.55) |
| Himikan | clothes | 55,977 (43%) | 0.07 | 11.29 / 23.31 (14.92 / 24.21) | 0.59 (0.15) | 1.75 / 0.91 → 1.22 | 2/2 | 0.09 / 4.24 → 4.19 (5.41) |
| Himikan | clothes_inner | 1,171 (4%) | – (`none`) | 36.87 / 41.41 | 0.02 | 36.67 / 0.46 → 36.21 | 0/0 | 4.26 / 1.46 → 4.44 |
| Himikan | neck | 88 (2%) | 0.21 | 3.25 / 10.23 (3.25 / 10.23) | 0.69 (0.69) | 3.13 / 0.98 → 2.58 | 0/0 | 2.05 / 1.30 → 1.14 (1.14) |
| Himikan | eyeR_iris | 790 (61%) | 0.06 | 30.39 / 50.56 (31.74 / 51.10) | 0.26 (0.29) | 28.32 / 2.09 → 26.58 | 0/1 | 8.92 / 5.98 → 8.09 (7.44) |
| Himikan | eyeL_iris | 824 (64%) | 0.04 | 35.02 / 60.60 (35.14 / 60.88) | 0.25 (0.33) | 30.57 / 1.64 → 29.26 | 0/0 | 9.60 / 5.96 → 8.47 (6.84) |
| Himikan | mouth_inner | 103 (86%) | 0.18 | 28.41 / 59.65 | – (맥락 < 64px) | 2.16 / 0.00 → 2.16 | 0/0 | 16.13 / 0.47 → 15.76 |
| Himikan | hair_side_R | 385 (3%) | 0.14 | 4.90 / 30.34 (5.52 / 40.13) | 0.77 (0.74) | 3.81 / 1.43 → 3.11 | 4/4 | 2.04 / 5.50 → 3.85 (4.65) |
| Himikan | hair_side_L | 667 (5%) | 0.18 | 4.75 / 27.38 (5.48 / 30.20) | 0.73 (0.71) | 3.60 / 1.17 → 3.13 | 3/3 | 1.76 / 5.40 → 3.99 (4.83) |
| Rritteang | hair_front | 18,928 (6%) | 0.07 | 7.64 / 31.86 (10.44 / 37.57) | 0.52 (0.39) | 3.19 / 2.85 → 2.82 | 1/1 | 1.07 / 4.71 → 3.95 (4.41) |
| Rritteang | hair_side_R | 881 (1%) | 0.14 | 4.20 / 17.25 (4.21 / 18.65) | 0.65 (0.64) | 3.14 / 2.21 → 2.87 | 1/1 | 2.15 / 3.93 → 2.25 (2.48) |
| Rritteang | hair_side_L | 1,081 (1%) | 0.12 | 5.82 / 31.11 (9.45 / 35.58) | 0.70 (0.51) | 3.79 / 2.08 → 2.74 | 0/0 | 2.59 / 5.95 → 3.71 (6.02) |
| Rritteang | eyeR_lower | 7,224 (74%) | – (`none`) | 7.09 / 26.83 | 0.76 | 11.63 / 4.40 → 8.09 | 2/2 | 2.81 / 3.72 → 3.42 |
| Rritteang | acc_glasses | 6,545 (75%) | – (`none`) | 23.02 / 63.07 | 0.27 | 23.52 / 7.19 → 18.95 | 4/8 | 6.70 / 13.71 → 10.60 |
| Rritteang | mouth_inner | 10 (1%) | 0.11 | 14.99 / 30.94 | 0.20 | 15.27 / 9.70 → 10.35 | 1/1 | 10.37 / 10.09 → 3.73 |
| Sutteang | face | 24,783 (30%) | 0.05 | 2.77 / 7.68 (4.50 / 10.43) | 0.72 (0.47) | 1.01 / 0.55 → 0.68 | 1/1 | 0.17 / 1.11 → 0.96 (1.60) |
| Sutteang | clothes | 200,615 (73%) | 0.09 | 7.66 / 20.35 (10.59 / 22.97) | 0.36 (0.27) | 3.27 / 1.63 → 2.48 | 2/2 | 0.15 / 3.34 → 3.20 (4.47) |
| Sutteang | eyeR_iris | 1,399 (41%) | 0.05 | 24.29 / 46.08 (27.31 / 45.42) | 0.06 (0.12) | 29.74 / 3.29 → 26.95 | 2/5 | 6.62 / 6.51 → 6.22 (5.15) |
| Sutteang | eyeL_iris | 1,330 (39%) | 0.05 | 29.21 / 54.15 (32.49 / 55.06) | 0.06 (0.04) | 32.34 / 2.67 → 30.31 | 2/2 | 7.10 / 6.14 → 6.48 (5.44) |
| Sutteang | eyeR_lower | 46 (17%) | 0.20 | 8.79 / 15.69 (4.79 / 14.48) | 0.15 (0.36) | 10.75 / 0.83 → 9.92 | 0/0 | 1.55 / 5.20 → 4.19 (2.88) |
| Sutteang | eyeL_lower | 54 (22%) | 0.26 | 7.65 / 20.20 (7.07 / 19.65) | 0.00 (0.00) | 8.08 / 1.66 → 6.42 | 0/0 | 3.10 / 4.90 → 4.52 (4.30) |
| Sutteang | browR | 2,790 (83%) | 0.05 | 10.26 / 13.04 (9.19 / 12.92) | 0.07 (0.22) | 11.32 / 1.03 → 10.33 | 0/0 | 0.67 / 2.47 → 2.75 (4.91) |
| Sutteang | browL | 2,258 (72%) | 0.02 | 10.82 / 13.57 (9.54 / 12.69) | 0.07 (0.25) | 10.73 / 0.99 → 9.93 | 0/0 | 0.68 / 1.58 → 1.95 (3.13) |
| Sutteang | mouth_inner | 168 (37%) | 0.20 | 29.68 / 52.14 (29.71 / 52.14) | 0.04 (0.04) | 28.72 / 4.38 → 24.34 | 1/2 | 13.85 / 5.54 → 8.94 (9.17) |
| Sutteang | mouth_lower | 20 (26%) | – (`none`) | 4.98 / 6.63 | – (맥락 < 64px) | 4.56 / 1.04 → 3.53 | 0/0 | 0.84 / 0.98 → 0.75 |
| Sutteang | hair_side_R | 5,341 (9%) | 0.07 | 5.59 / 15.83 (5.89 / 17.91) | 0.43 (0.43) | 3.01 / 2.14 → 2.23 | 3/3 | 1.33 / 4.29 → 3.19 (3.60) |
| Sutteang | hair_side_L | 2,816 (4%) | 0.09 | 4.34 / 12.64 (4.39 / 12.89) | 0.60 (0.58) | 3.06 / 1.99 → 2.14 | 1/1 | 1.67 / 4.05 → 2.60 (2.92) |

이음새별 주요 값(표본 ≥50, `stats.json` `fidelity.holdout.parts[].groups`): Himikan `clothes` ← `hair_side_R` 27,698px 11.87 / 19.85, ← `hair_side_L` 28,279px 10.71 / 23.49 · `face` ← `hair_front` 768px 0.78 / 2.45, ← 아래 눈꺼풀 1,142 / 1,209px 0.69 / 1.94, 1.13 / 3.39. Sutteang `clothes` ← `hair_side_R` 128,559px 6.75 / 16.82, ← `hair_side_L` 70,586px 9.19 / 22.88 · `face` ← `hair_front` 10,273px 2.11 / 5.67, ← `eyeR_upper` 262px 17.17 / 40.23. Rritteang `hair_front` ← `hat` 17,746px 7.63 / 31.70.

### 표 3 — 전 층 홀드아웃 (가시 맥락 없음, 이음새 밴드 대리 아님 — 따로 기록)

| 원화 | 파츠 (홀드아웃 px) — ΔE00 평균 / p95 |
| --- | --- |
| Himikan | hair_back 4,718 — 15.86 / 62.37 · body 3,229 — 14.19 / 37.52 · eyeR_white 230 — 10.33 / 39.32 · eyeL_white 327 — 7.85 / 36.74 · eyeR_upper 756 — 29.78 / 52.85 · eyeL_upper 738 — 28.31 / 52.99 · browR 281 — 7.44 / 7.44 · browL 342 — 7.43 / 7.44 · mouth_lower 17 — 4.50 / 4.91 |
| Rritteang | clothes 200,931 — 8.49 / 36.01 · face 44,824 (96%) — 5.10 / 13.98 · hair_back 37,360 (97%) — 18.22 / 47.28 · neck 2,455 — 23.82 / 62.37 · body 169 — 43.56 / 73.81 · eyeR/L_white 8,095 / 7,067 — 18.75 / 54.38, 22.94 / 56.52 · eyeR/L_upper 6,783 / 8,289 — 19.24 / 61.43, 21.31 / 65.22 · eyeL_lower 11,371 — 9.00 / 34.68 · browR/L 2,926 / 7,401 — 32.14 / 51.31, 41.50 / 50.99 |
| Sutteang | hair_back 100,673 (98%) — 10.28 / 25.52 · body 45,024 — 12.33 / 23.93 · neck 25,801 — 9.18 / 15.93 · eyeR/L_white 1,411 / 1,449 — 25.47 / 47.06, 30.64 / 47.12 · eyeR/L_upper 2,273 / 2,209 — 37.14 / 70.31, 34.22 / 66.72 |

### 육안 대조 (비교 이미지)

육안으로 가장 나빠 보이는 3개와 해당 수치:

1. **Himikan 홀드아웃 `clothes`** (`artifacts/t14/himikan/fidelity/holdout_clothes.png`): 옆머리 아래 밴드(가시의 43%)에서 세일러 칼라의 남색·흰 줄·엠블럼·흰 앞판이 회색 단색으로 칠해진다. ΔE00 11.29 / 23.31(가중 14.92 / 24.21), 팔레트 0.59(가중 0.15), σL* 0.09 vs 4.24. 수치와 육안이 맞는다(무늬·구조 손실이 σ와 가중 팔레트에 드러남).
2. **Himikan 홀드아웃 `clothes_inner`(빨간 스카프)** (`holdout_clothes_inner.png`): 덮는 파츠(머리카락) 색을 그대로 복사해 스카프에 머리색 덩어리가 생긴다(착용물 `_` 분기 = 그 위치 원본 픽셀 복사). ΔE00 36.87 / 41.41, 단차 |Δ| 36.21. 맞는다. (이 파츠는 현재 숨은 영역이 0px이라 제품 출력에 바로 나타나지는 않지만, 같은 분기를 타는 착용물 숨은 영역의 결함 유형이다.)
3. **Himikan 정답 `hair_back`** (`truth_hair_back.png`): 앞머리·옆머리 아래 뒷머리가 결 없는 평면이 되고 머리카락 선이 모두 사라진다. ΔE00 평균은 6.67 / 26.28로 크지 않지만 선화 끊김 129/129, 단차 |Δ| 12.30(앞머리 아래 39.55), σL* 3.43 vs 6.61. **ΔE 평균만으로는 육안 결함이 과소평가**되고 선화·질감 지표가 이를 보완한다 → 단일 ΔE 임계값이나 종합 점수로 판정하지 않는 근거.

그 밖에 `eyeR/L_iris`(정답): 윗눈꺼풀 아래 홍채 윗테두리를 갈색 속눈썹 그림자로 칠해 PSD의 짙은 보라 테두리와 다르다(ΔE00 13.42 / 10.66, 단차 |Δ| 13.70 / 8.71), 홍채를 PSD 홍채 밖까지 그린 텍셀 526 / 617px(PSD 화면 대비 ΔE 46.73 / 61.81). `face`·`eye*_white`·`clothes`(정답)는 육안 차이가 작고 ΔE 평균 1.0–3.0이다.

### T16 착수 조건 판단 (사실 기록)

- 조건(development-tasks.md T16): "T14 수치에서 `texture`/`semantic` 노출 밴드가 **T9·T15 이후에도** 육안 판정상 원화와 구분될 때만 착수".
- 현재: **미충족(판단 불가)**. T9·T15가 아직 없고, 영역 유형 분류(`fillKind`, T9)가 없어 노출 밴드를 `texture`/`semantic`으로 나눌 수 없다. 현 조화 보간 기준선에서는 질감·무늬 밴드가 원화와 육안으로 구분된다(육안 대조 1·3번; σL* 합성/정답: Himikan 홀드아웃 `clothes` 0.09 / 4.24, Sutteang 홀드아웃 `clothes` 0.15 / 3.34, Himikan 정답 `hair_back` 3.43 / 6.61; 선화 끊김 `hair_back` 129/129). T9·T15 후 같은 단위(ΔE00 평균·p95, σL* 합성/정답, 선화 끊김/교차, 노출 빈도 가중값)로 다시 재서 판단한다.

### 남은 제한

- (a) 이음새 최대 노출 깊이를 밴드 폭으로 쓰면(과제 정의) 깊은 이음새가 있는 원화(Rritteang `clothes` 100%·`face` 96%, Sutteang `hair_back`·`neck`·`body`)는 하위 파츠 전체가 홀드아웃되어 이음새 밴드 대리가 아니다. 표 3으로 분리했다. 폭 상한(예: p95 깊이)을 두는 변형은 정의를 바꾸는 것이라 하지 않았다.
- (b) 모든 이음새 밴드를 한 번에 홀드아웃한다(합성 1회). 같은 색 필드를 쓰는 파츠(피부: `face`·`neck`·`body`·눈꺼풀·입)끼리 시드가 함께 줄어, 이음새 하나만 가린 경우보다 조건이 나쁘다.
- (c) 홀드아웃에서 분할·증거 마스크(세그멘테이션)는 원본 그대로이고 텍스처 단계만 치환된 원본을 본다. 실제 가림과 달리 증거 마스크(`ev_face` 등)는 홀드아웃 픽셀을 여전히 해당 파츠로 표시한다.
- (d) 정답 비교는 683×1024 리샘플 그리드에서 한다(재합성 ΔE00 평균 0.30, p95 1.38이 바닥값). 반투명 가장자리는 알파 ≥128만 정답으로 센다.
- (e) 레이어 대응은 수동이며 `body`·`face_detail`·`clothes_hands`는 정답이 없다. 대응표가 바뀌면 버전을 올린다.
- (f) `noTruthVisualMetrics`는 "이 파츠 이하(flat2rig 순서) 파츠의 정답 합성"과 비교한다. flat2rig와 PSD의 상하 순서가 다른 곳(예: 눈썹 숨은 영역이 PSD에서는 앞머리 아래 이마)에서는 비교 대상이 실제 PSD 화면과 다를 수 있어 눈썹 값(0.41–0.98)은 해석하지 않는다.
- (g) 선화 판정은 파츠 가시 중앙값 대비 L* −25 고정 규칙이다(어두운 파츠의 선, 색선은 놓칠 수 있음). 팔레트 k-means는 결정적 초기화, k=8 고정이다.
- (h) 노출 맵 한계(T13b (e))를 그대로 물려받는다: 사용자 눈·입 모양 조정과 물리 흔들림을 포함하지 않는다. Windows 앱·Cubism 가져오기 검증 없음(`미검증`).
