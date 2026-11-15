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

