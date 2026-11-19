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

## T9 경계 후광 제거 · 영역 유형 분류 · `flat/cel` 팔레트 필드 — T14 기준선 대비 (2026-11-18)

- 실행: `cd src-tauri && cargo test` (lib 63/63 — 신규 `analysis::region_fill` 6: 셀 레벨셋 연장, 그라데이션 스냅 해제, 질감/선화 분류, 넓은 어두운 영역은 시드, 림 후광·상위 선화 제거와 자기 외곽선 유지·상위 레이어로 넘길 픽셀 목록과 그 계획 재현, 불투명 배경 감지; 신규 `rig::exposure::overlay_fill_over_a_lower_visible_seam_is_not_a_hidden_texel`; 갱신 1: `template_bank_preserves_rest_pose_visible_pixels_and_extra_hair_strands`는 가시 합성에서 "이음새 2px 림 밖 0 변화 + 변경 수 ≤ 림 보고 합계", 렌더러와 같은 유효 마스크 합성에서 "배경 후광·상위 레이어 자기 림 혼합 외 0 변화", 오버레이 텍셀 수 = 위로 넘긴 픽셀 수, 각 레이어 `fillMask`가 자기 가시 마스크와 겹치지 않음), `cargo clippy --all-targets -- -D warnings` (통과), `npm test` (45/45 — 신규 2: T9 이전 v3 프로젝트는 미분류로 읽힘·값 왕복·알 수 없는 값 거부, 브라우저 대체 경로는 `fillSynth: "diffusion"`·유형 없음), `npx tsc --noEmit` (통과). Windows 앱 모션·UI 표시는 확인하지 않았다(`미검증`).
- 전/후 비교: 작업 전 HEAD(`6eb8e41`) 릴리스 CLI(`artifacts/t9/bin-before/`; T14 커밋 산출물과 원화 3종 65/52/53 파일 동일 확인)와 T9 릴리스 CLI를 같은 입력·같은 플래그로 실행했다. 아래 수치는 **리뷰 수정(이음새 오버레이) 반영본**(`artifacts/t9/after/`)이며, 수정 전 첫 T9 결과는 `artifacts/t9/after_v1/`에 남겼다. T9 CLI는 정지 포즈 가시 소유자 맵 `owner.png`(레이어 번호 + 1, 16-bit)를 추가로 쓴다(분할 비교용).

```sh
flat2rig-cli --eval-holdout --eval-truth artifacts/t14/himikan-truth artifacts/design-review/input.png artifacts/t9/after/himikan
flat2rig-cli --eval-holdout ref/ritteang/Rritteang.png artifacts/t9/after/ritteang
flat2rig-cli --eval-holdout ref/sutteang/2d_.png       artifacts/t9/after/sutteang
tools/evaluation/.venv/Scripts/python.exe -I tools/evaluation/t9_report.py INPUT.png artifacts/t9/before/<원화> artifacts/t9/after/<원화> --auto-crops artifacts/t9/crops_auto
```

전체 표는 `artifacts/t9/report_{himikan,ritteang,sutteang}.md`(Git 제외). 실행 시간(평가 포함) Himikan 11.7초(전 11초), Sutteang 22.1초(전 21초).

### 불변 확인 (숨은 영역을 줄이지 않음)

- **가시 `mask` 분할**: 세 원화 모두 `foreground.png` 알파와 최상위 소유자 맵이 전/후 동일(`visible partition identical: True`), 파츠별 가시 픽셀 동일. 림 디컨태미네이션은 텍스처 RGB만 바꾼다.
- **숨은 영역(`fillMask`)**: 전 T9 텍셀은 모두 그대로이고(`parts/<id>.png` 유효 알파에서 잃은 텍셀 0, 숨은 영역 생성 로직 불변), **늘어난 것은 상위 레이어의 이음새 오버레이 텍셀뿐**이다(아래 "이음새 오버레이"). 파츠별 `fillPixels` 전 → 후(괄호 = 오버레이): Himikan body 166,265 → 166,293 (28) · clothes 36,414 → 36,454 (40) · clothes_inner 0 → 844 (844) · face 26,608 → 26,632 (24) · clothes_hands 0 → 23 (23) · eyeR_iris 1,718 → 1,724 (6) · eyeR_lower 0 → 1 (1) · eyeR_upper 556 → 564 (8) · eyeL_white 3,219 → 3,220 (1) · eyeL_iris 1,747 → 1,750 (3) · eyeL_lower 0 → 5 (5) · eyeL_upper 551 → 560 (9) · hair_side_R 1,205 → 1,293 (88) · hair_side_L 1,551 → 1,647 (96) · hair_front 0 → 30 (30); 그대로 hair_back 157,285 · neck 8,679 · eyeR_white 3,890 · browR/L 746/907 · mouth_inner 62. Rritteang body 264,712 → 264,730 (18) · neck 47,172 → 47,189 (17) · clothes 72,984 → 73,369 (385) · face 259,056 → 259,179 (123) · eyeR_iris 8,780 → 8,795 (15) · eyeR_lower 0 → 54 (54) · eyeR_upper 4,212 → 4,374 (162) · eyeL_white 84,033 → 84,193 (160) · eyeL_iris 10,367 → 10,388 (21) · eyeL_lower 268 → 277 (9) · eyeL_upper 3,465 → 3,823 (358) · browR/L 5,702 → 5,704 (2) / 5,491 → 5,497 (6) · mouth_inner 8 → 16 (8) · acc_glasses 0 → 61 (61) · hair_side_R 2,212 → 3,076 (864) · hair_side_L 3,029 → 4,296 (1,267) · hair_front 64,786 → 64,889 (103) · hat 0 → 141 (141) · acc_hairpin 0 → 111 (111); 그대로 hair_back 696,875 · eyeR_white 79,608. Sutteang body 308,614 → 308,663 (49) · neck 45,630 → 45,645 (15) · clothes 24,433 → 25,131 (698) · face 52,930 → 53,072 (142) · face_detail 0 → 1 (1) · clothes_hands 0 → 188 (188) · eyeR_white 8,629 → 8,665 (36) · eyeR_iris 3,419 → 3,437 (18) · eyeR_lower 65 → 68 (3) · eyeR_upper 894 → 999 (105) · eyeL_white 8,099 → 8,117 (18) · eyeL_lower 113 → 129 (16) · eyeL_upper 888 → 1,017 (129) · browL 1,179 → 1,204 (25) · hair_side_R 9,110 → 9,489 (379) · hair_side_L 4,661 → 4,914 (253) · hair_front 385 → 637 (252) · acc_hairpin 0 → 10 (10); 그대로 hair_back 663,257 · eyeL_iris 3,294 · browR 1,044 · mouth_inner 86. 줄어든 파츠 없음.
- **노출 맵·이음새**: 오버레이 텍셀은 정지 포즈에서 하위 레이어 가시 픽셀 위에 그려지는 것이므로 숨은 텍셀이 아니다 — `rig/exposure.rs`가 "더 아래 레이어가 가시인 픽셀 위의 `fillMask`"를 `overlayPx`로 따로 세고 숨은/노출 집계에서 뺀다(템플릿 숨은 영역은 위생 단계에서 더 위 가시 레이어 아래에만 있으므로 해당 없음). 숨은 합계 Himikan 408,423, Rritteang 1,591,612, Sutteang 1,128,338(전 = 후). 노출 합계는 83,266 → 82,395, 412,271 → 410,226, 335,176 → 334,038로 0.3–1.0% **줄었다** — 숨은 영역을 줄인 것이 아니라, 상위 레이어 메쉬가 이음새 오버레이만큼 넓어져 포즈에서 그 1–2px 띠를 더 덮기 때문이다. 같은 이유로 이음새 `holes` 52 → 52, 11,907 → 11,871, 1,974 → 1,973. 오버레이 합계 1,206 / 3,885 / 2,337px. 홀드아웃 실행 `masksIdentical`·`fillMasksIdentical` 세 원화 모두 `true`(오버레이 계획은 제품 분할·원본 픽셀만으로 정하므로 홀드아웃이 바꾸지 못한다). `deform`·`PARAM_DEFS`·이음새 요구량·숨은 영역 생성 로직은 바꾸지 않았다.
- **이음새 오버레이(리뷰 수정)**: 하위 레이어 림에서 지우는 상위 외곽선 절반·상위 쪽 안티앨리어스 혼합·얼굴의 눈/눈썹/입 잔여 선화는 하위 텍스처에서는 깨끗한 색으로 바꾸고, **원래 색을 상위 레이어 텍스처에, 그 픽셀을 상위 레이어 `fillMask`에** 넣는다(가시 `mask` 분할 불변, 메쉬 `effective_part_mask`가 덮음). 계획은 텍스처 단계 전에 제품 분할·원본으로 한 번 정하고, 텍스처 단계는 그 계획을 따른다(`decontaminate_rim(…, Some(plan))`). 정지 포즈에서는 상위가 하위 위에 그려져 원래 선 두께가 되고, 모션에서는 선이 상위 파츠와 함께 움직인다. 오버레이 합계가 위로 넘긴 수(상위 선화 + 상위 쪽 혼합 + 잔여 선화)보다 조금 큰 것(Himikan 1,206 vs 1,203, Rritteang 3,885 vs 3,848)은 하위 텍스처가 이미 분기(얼굴 반점 정리 등)에서 바뀌어 텍스처 단계 집계에서 빠진 계획 픽셀이다(오버레이는 원래 색으로 그대로 그려진다).

### 표 1 — 기본 포즈 vs 원화, ΔE00

전경 = 원화 알파 ≥128 ∩ 파츠 전경. 림 = 다른 소유자(또는 배경)로부터 4-인접 2px 이내 전경 픽셀. **이음새 외곽선** = 림 중 다른 *파츠*(배경 아님)에 붙어 있고 원화 L*이 소유 파츠 가시 중앙값보다 25 이상 낮은 픽셀. **정지 렌더** = `effective_foreground.png`(각 파츠 `mask ∪ fillMask`를 아래에서 위로 그린 것 — 앱이 정지 포즈에서 그리는 것과 같은 구성). 가시만 합성(`foreground.png`)은 `fillMask`를 그리지 않으므로 오버레이로 옮긴 선이 빠져 보인다(참고용).

| 원화 | 이미지 | 영역 | px | 평균 전 → 후 | p95 전 → 후 | >5 비율 전 → 후 | >10 비율 전 → 후 |
| --- | --- | --- | ---: | --- | --- | --- | --- |
| Himikan | **정지 렌더** | 전경 | 301,259 | 0.058 → 0.067 | 0.00 → 0.00 | 0.45% → 0.52% | 0.22% → 0.28% |
| Himikan | **정지 렌더** | 림 ≤2px | 23,021 | 0.736 → 0.861 | 6.16 → 7.61 | 5.71% → 6.66% | 2.85% → 3.54% |
| Himikan | **정지 렌더** | **이음새 외곽선** | 2,357 | **2.520 → 2.556** | 12.50 → 12.63 | 21.09% → 21.38% | 9.63% → 9.80% |
| Himikan | **정지 렌더** | 내부 | 278,238 | 0.002 → 0.002 | 0.00 → 0.00 | 0.01% → 0.01% | 0.01% → 0.01% |
| Himikan | 가시만 | 이음새 외곽선 | 2,357 | 0.038 → 4.575 | 0.00 → 29.06 | 0.17% → 17.69% | 0.17% → 16.80% |
| Rritteang | **정지 렌더** | 전경 | 1,225,446 | 0.194 → 0.188 | 0.71 → 0.71 | 0.24% → 0.31% | 0.18% → 0.19% |
| Rritteang | **정지 렌더** | 림 ≤2px | 78,515 | 0.840 → 0.830 | 1.05 → 2.69 | 2.92% → 4.14% | 2.07% → 2.53% |
| Rritteang | **정지 렌더** | **이음새 외곽선** | 12,880 | **2.556 → 1.004** | 19.82 → 0.76 | 5.35% → 2.10% | 5.34% → 1.91% |
| Rritteang | **정지 렌더** | 내부 | 1,146,931 | 0.150 → 0.144 | 0.70 → 0.71 | 0.06% → 0.05% | 0.05% → 0.03% |
| Rritteang | 가시만 | 이음새 외곽선 | 12,880 | 2.279 → 4.970 | 19.84 → 41.38 | 5.26% → 14.81% | 5.26% → 13.73% |
| Sutteang | **정지 렌더** | 전경 | 861,869 | 0.042 → 0.044 | 0.00 → 0.00 | 0.28% → 0.39% | 0.09% → 0.09% |
| Sutteang | **정지 렌더** | 림 ≤2px | 74,781 | 0.212 → 0.310 | 0.00 → 0.00 | 1.33% → 2.93% | 0.42% → 0.72% |
| Sutteang | **정지 렌더** | **이음새 외곽선** | 2,682 | **2.617 → 0.456** | 24.33 → 0.00 | 8.02% → 2.72% | 7.79% → 2.39% |
| Sutteang | **정지 렌더** | 내부 | 787,088 | 0.025 → 0.019 | 0.00 → 0.00 | 0.18% → 0.15% | 0.06% → 0.03% |
| Sutteang | 가시만 | 이음새 외곽선 | 2,682 | 2.599 → 4.572 | 24.33 → 32.93 | 7.79% → 15.55% | 7.79% → 14.84% |

- **이음새 외곽선은 정지 렌더에서 원래 수준이거나 더 가깝다**: Himikan 2.52 → 2.56(같은 수준), Rritteang 2.56 → 1.00, Sutteang 2.62 → 0.46. 전 T9 정지 렌더의 외곽선 오차(Rritteang·Sutteang p95 20–24)는 상위 레이어의 부분 알파 가장자리 밑으로 하위 레이어의 숨은 합성 색이 비쳐 보이던 것이고, 지금은 그 자리에 원래 선 색 오버레이가 있어 줄었다. 확대 비교: `artifacts/t9/crops/rest_{chin_neck,strand_edge,collar_line}_src_before_t9a_t9b.png`(원화 · 전 · 수정 전 T9 · 수정 후 T9, 정지 렌더) — 수정 전 T9에서 얇아졌던 턱선·머리카락 외곽선이 원화 두께로 돌아왔다.
- 정지 렌더의 남은 차이는 림의 **배경 후광 제거**(Himikan 1, Rritteang 3,753, Sutteang 295px)와 **상위 레이어 자기 림의 하위 쪽 혼합**(밝아지는 안티앨리어스, 223 / 1,848 / 2,015px; 어두운 외곽선 픽셀은 건드리지 않음)이다. 림 전체 평균은 Himikan +0.13, Rritteang −0.01, Sutteang +0.10. 내부는 변화 없음(±0.01).
- 림 처리 집계(Himikan / Rritteang / Sutteang): 위로 넘긴 상위 선화 515 / 1,590 / 329, 상위 쪽 혼합 688 / 1,741 / 1,670, 얼굴 잔여 선화 0 / 517 / 269, 그대로 둔 것: 상위가 갖지 않는 선화 136 / 25 / 283, 4px 안에 깨끗한 시드가 없는 림 1,856 / 10,326 / 9,476.
- **모션에서 이중 선화가 다시 생기지 않음**: `artifacts/t9/crops_auto/<원화>_{chin,neck_top,collar}_motion_turn.png`(원화 정지 · 전 · 후) — 턱선은 얼굴과 함께 움직이고 목 쪽에 두 번째 선이 남지 않는다. Sutteang 목 상단의 밝은 얼룩·흰 줄은 수정 후에도 사라진 상태다. 머리카락 아래 의상(`collar`)도 의상 쪽에 머리카락 외곽선 잔상이 없다.
- 구현 중 육안으로 발견해 고친 회귀(최종 수치에는 없음): 공간상 가장 가까운 시드가 다른 셀 영역(남색 칼라)이라 흰 셔츠 림에 어두운 점이 찍힘 → 혼합은 색이 가장 가까운 4px 안 시드로; 하위 쪽 혼합 판정이 남색 위 머리카락 외곽선을 지움 → 어두워지는 혼합은 유지; 얼굴 잔여 선화 판정이 홍조 해칭을 통째로 흰 피부로 덮음 → L* 25 이상 어둡고 피부/홍조색이 아닐 때만; 특징 가드 안의 얼굴 림이 먼 흰 피부 색으로 바뀜 → 4px 안 시드 없으면 유지; 질감(안경테) 림 손상 → `flat_cel`만 림 처리.

### 표 2 — T14 지표 전 → 후 (값이 바뀐 파츠, 노출 밴드)

유형 = `fillRegion`(셀 = 스냅 판정), 합성 = `fillSynth`. σL* = 국소 L* 표준편차 합성/정답. 정답 비교의 표본(노출 텍셀)은 이음새 오버레이로 상위 메쉬가 넓어져 일부 줄었다(표본 열 전 → 후) — Himikan `clothes`·`neck`의 개선 폭 일부는 표본 변화 몫이고, `hair_side_R`의 악화(정답 σL* 6.69 → 7.38)도 표본 집합이 바뀐 결과다. 홀드아웃은 표본이 같고 노출 빈도 가중값만 조금 바뀐다.

| 세트 | 파츠 | 유형 / 합성 | 표본(노출 px) | ΔE00 평균 / p95 | 가중 평균 | 팔레트 일치 | 그래디언트 불연속 | σL* 합성/정답 | 선화 끊김 |
| --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| Himikan 정답 | neck | flat_cel(셀) / palette_levelset | 126 → 110 | 6.40 / 13.88 → **2.82 / 12.45** | 7.79 → 5.38 | 0.52 → 0.67 | 2.24 → 1.19 | 1.58/1.61 → 1.04/1.84 | 0/0 → 0/0 |
| Himikan 정답 | clothes | flat_cel / local_field | 5,708 → 5,179 | 2.98 / 14.38 → **2.52 / 13.10** | 3.30 → 2.91 | 0.78 → 0.82 | 3.35 → 3.14 | 2.93/3.43 → 2.37/2.90 | 4/4 → 3/3 |
| Himikan 정답 | hair_side_R | flat_cel / class_field | 155 → 154 | 5.21 / 32.05 → 5.74 / 32.05 | 5.95 → 6.90 | 0.70 → 0.68 | 1.13 → 1.19 | 1.99/6.69 → 2.14/7.38 | 0/0 → 0/0 |
| Himikan 홀드아웃 | neck | flat_cel(셀) / palette_levelset | 88 | 3.25 / 10.23 → **1.43 / 5.69** | 3.25 → 1.43 | 0.69 → 0.88 | 2.58 → 1.23 | 2.05/1.30 → 1.29/1.30 | 0/0 → 0/0 |
| Himikan 홀드아웃 | clothes_inner | flat_cel / local_field | 1,171 | 36.87 / 41.41 → **2.13 / 7.75** | – | 0.02 → 0.50 | 36.21 → 0.56 | 4.26/1.46 → 0.25/1.46 | 0/0 → 0/0 |
| Himikan 홀드아웃 | clothes | flat_cel / local_field | 55,977 | 11.29 / 23.31 → 11.26 / 23.53 | 14.92 → 14.92 | 0.59 → 0.59 | 1.22 → 1.26 | 0.09/4.24 → 0.09/4.24 | 2/2 → 2/2 |
| Sutteang 홀드아웃 | clothes | flat_cel / local_field | 200,615 | 7.66 / 20.35 → 7.72 / 20.67 | 10.59 → 10.36 | 0.36 → 0.37 | 2.48 → 2.63 | 0.15/3.34 → 0.13/3.34 | 2/2 → 2/2 |
| Rritteang 홀드아웃 | acc_glasses | texture / (평가 전용) | 6,545 | 23.02 / 63.07 → 23.87 / 54.93 | – | 0.27 → 0.15 | 18.95 → 9.28 | 6.70/13.71 → 0.49/13.71 | 4/8 → 8/8 |

- **바뀌지 않은 값**: Himikan 정답 `face`·`hair_back`·`hair_side_R/L`·눈·입, 홀드아웃 `face`·`hair_side_R/L`·홍채·`mouth_inner`, Rritteang·Sutteang의 나머지 홀드아웃 파츠는 전/후 같은 값이다(얼굴·머리카락은 스냅되지 않으면 기존 공용 필드를 유지하고, 눈·입·눈썹 합성은 T10 범위로 그대로 두었다). 전 층 홀드아웃 파츠(Rritteang `clothes`·`face`·`neck`·`hair_back`, Sutteang `neck`·`body`·`hair_back` 등)는 홀드아웃 시 자기 시드가 없어 공용 필드로 돌아가므로 값이 같고, **제품 경로 변경(Sutteang 목·몸통의 자기 그림자 연장 등)은 수치로 측정되지 않는다**(육안 비교 이미지만).
- **나빠진 항목**: (1) Sutteang 홀드아웃 `clothes` ΔE 평균 +0.06, p95 +0.32, 그래디언트 불연속 +0.15(가중 평균은 −0.23, 팔레트 +0.01). 자기 시드 국소 필드가 기존 의상 필드(머리카락 증거·착용물·절대 기준 선화 제외)와 다른 시드 집합(상대 기준 가는 선화 제외)을 쓰기 때문이며, 이 파츠는 그라데이션 판정(셀 중심 비율 0.08)이라 스냅이 없다. (2) Himikan 홀드아웃 `clothes` p95 +0.22·불연속 +0.04(같은 이유; 칼라는 아래 판정). (3) Rritteang 홀드아웃 `acc_glasses`(평가 전용 — 제품에서는 숨은 영역 0px): 이전에는 착용물 분기가 홀드아웃 픽셀 자리의 치환 원본(덮는 레이어 색)을 그대로 복사해 테 모양이 우연히 남았고, 지금은 자기 국소 필드라 테 획이 끊긴다(끊김 4/8 → 8/8, 팔레트 0.27 → 0.15; p95·불연속은 개선). 덮는 레이어 색 복사는 결함 1과 같은 버그 유형이라 되돌리지 않았고, 안경테 같은 `texture`/`line` 착용물의 획 연장은 T15 범위다.
- **`flat/cel` 개선 판정 — 부분 충족**: 스냅(팔레트 레벨셋)이 실제로 쓰인 `flat/cel` 밴드는 Himikan `neck` 하나이며 정답·홀드아웃 모두 ΔE·팔레트 일치·불연속이 개선됐다. 자기 시드 국소 필드를 쓴 `flat/cel` 밴드는 Himikan `clothes_inner`(대폭 개선)·`clothes` 정답(개선)·홀드아웃(거의 동일, p95 소폭 악화)·Sutteang `clothes` 홀드아웃(소폭 악화)이다. 원화 3종 대부분이 그라데이션 채색 판정(셀 중심 비율 0.08–0.71 < 0.75)이라 스냅이 꺼진다 — 기준대로의 동작이다.

### 표 3 — 영역 유형 분류 (원화 3종, `stats.json` `partCoverage[].fillRegion`)

기준(시드 = 가시 알파 ≥128, 가시 경계 2px 침식, 선화·다른 부위 증거·홀드아웃 제외; 밴드 = 숨은 영역에서 `clamp(0.06·hh, 12, 48)`px 이내 시드, 64개 미만이면 전체): 역할 `semantic`(홍채·`mouth_inner`)·`line`(눈썹·눈꺼풀) → 선화 비율 ≥ 0.30 또는 2px 침식 후 남는 비율 < 0.15이면 `line` → 5×5 국소 σL* 중앙값 ≥ 3.0이면 `texture` → 그 외 `flat_cel`. 셀(스냅) = 군집 ≥ 2이고 ρ < 0.15 시드 비율 ≥ 0.75. 자기 시드 < 32개 또는 < 숨은 텍셀 × 0.10이면 자기 필드를 쓰지 않는다.

| 원화 | `flat_cel` (셀 = 스냅 판정) | `line` | `texture` | `semantic` |
| --- | --- | --- | --- | --- |
| Himikan | hair_back, body(셀), neck(셀), clothes, clothes_inner, face, face_detail(셀), clothes_hands, eyeR/L_white, hair_side_R/L, hair_front | eyeR/L_upper, eyeR/L_lower, browR/L, mouth_upper/lower | – | eyeR/L_iris, mouth_inner |
| Rritteang | hair_back, neck, clothes, face(셀), eyeR/L_white, hair_side_R/L, hair_front, hat | body(가시 169px, 침식 후 0.11), eyeR/L_upper/lower, browR/L, mouth_lower | acc_glasses(σ 3.84), acc_hairpin(σ 3.13) | eyeR/L_iris, mouth_inner |
| Sutteang | hair_back, body, neck, clothes, face, face_detail, clothes_hands, eyeR_white, hair_side_R/L, hair_front | eyeR/L_upper/lower, browR/L, mouth_upper/lower | eyeL_white(σ 3.86), acc_hairpin(σ 4.65) | eyeR/L_iris, mouth_inner |

- 숨은 텍셀 합성(`fillSynth`): Himikan `neck` = `palette_levelset`, `clothes` = `local_field`, `body`(셀 판정이지만 시드 1,983 < 숨은 166,265 × 0.10)·`face`·머리카락 = `class_field`; Rritteang `clothes` = `local_field`, `neck`(시드 1,548 < 47,172 × 0.10)·`face`(셀 판정이지만 시드 비율 미달)·머리카락 = `class_field`; Sutteang `body`·`neck`·`clothes` = `local_field`, `face`·머리카락 = `class_field`; 눈·입·눈썹 = `procedural`.
- Rritteang 목은 처음 구현(비율 규칙 없음)에서 가시 2,455px의 팔레트(어두운 보라 군집 포함)를 숨은 47,172px로 레벨셋 연장해 세로 줄무늬가 생겼다(`artifacts/t9/crops/ritteang_neck_tex_before_after.png`, 당시 결과). 비율 규칙과 목·몸통 시드의 피부 휘도 55% 미만 제외를 추가해 공용 필드로 되돌렸다.

### 비교 이미지 (왼쪽 원화 · 가운데 전 · 오른쪽 후, `artifacts/t9/`, Git 제외)

- 자동 크롭 `crops_auto/<원화>_{chin,neck_top,collar,eyeR_lids,eyeL_lids,mouth,hands}_{foreground,motion_turn,motion_blink}.png`(랜드마크 기준; `hands`는 Himikan·Sutteang `clothes_hands`). 육안 확인: Sutteang `neck_top_motion_turn` — 전에는 턱 밑 목 숨은 영역에 얼굴 피부색 밝은 얼룩과 흰 세로 줄이 보였고 후에는 목 그림자 톤으로 이어진다(**목 상단 밝은 얼룩 제거**). Himikan `crops/himikan_neck_turn_before_after.png`·`crops/himikan_collar_turn_before_after.png`(모션 전/후) — 목 아래 의상 숨은 영역의 흰 줄·밝은 띠가 사라지거나 약해졌다. 숨은 텍스처 자체: `crops/sutteang_{neck,body}_tex_before_after.png`.
- 정지 포즈 확대: `crops/rest_{chin_neck,strand_edge,collar_line}_src_before_t9a_t9b.png`(Himikan 정지 렌더: 원화 · 전 · 수정 전 T9 · 수정 후 T9 — 외곽선 두께 원화와 같음). `crops/zoom_{collar_line,strand_edge,chin}.png`·`crops/zoom_ritteang_{eyeL_region,hat_top,hat_halo}.png`는 수정 전 T9의 가시 합성(외곽선 얇아짐 확인 당시 자료).
- 눈꺼풀(`*_lids_motion_blink`): 감은 눈의 회색 반점(빈 픽셀)은 전/후 같다 — 클리핑·눈꺼풀 곡선은 T10 범위.

### T14 기준선에서 드러난 결함 판정

1. **Himikan 홀드아웃 `clothes_inner`가 머리카락 색을 복사(ΔE 36.87)** → **T9 범위, 수정함**. 원인: 착용물 분기(`hat` 외 `clothes_*`·`acc_*`)가 숨은 텍셀에 그 위치 원본 픽셀을 복사했다(홀드아웃에서는 치환된 덮는 레이어 색, 제품에서는 덮는 레이어의 실제 픽셀 — 같은 버그). 이제 자기 시드 팔레트/국소 필드(시드 부족 시 가시 평균색)를 쓴다: 2.13 / 7.75, 팔레트 0.50, 불연속 0.56. 제품에서는 현재 이 파츠의 숨은 영역이 0px이라, 같은 분기를 타는 다른 착용물의 숨은 영역에 해당한다.
2. **Himikan 홀드아웃 `clothes` 세일러 칼라 회색 평탄화(σL* 0.09 vs 4.24)** → **T9 범위 아님, 미수정**. 원인: 홀드아웃 밴드(옆머리 아래 이음새 최대 노출 깊이 ≈85px)가 칼라 전체(남색 바탕·흰 줄·엠블럼·흰 앞판)를 덮어 남은 가시 맥락이 회색 재킷 몸통뿐이다(`artifacts/t9/after/himikan/fidelity/holdout_clothes.png`). 시드에 칼라 색이 없으므로 시드 통계 기반 필드(평균·국소·팔레트 어느 것이든)는 회색을 낸다 — 시드 오염·평균색 연장이 아니라 **구조·무늬 복원** 문제다. 해당: T15(선화·구조 연장), T16(조건부 예시 합성), T11(수동 보정). 수치 11.29 → 11.26(동일 수준). 제품 경로의 같은 부위(정답 비교, 옆머리 아래 의상 노출 텍셀)는 2.98 → 2.80으로 개선됐다.

### T16 착수 조건 (갱신)

- 영역 유형이 생겨 밴드를 나눌 수 있게 됐다. 원화 3종에서 `texture`로 분류된 파츠는 착용물(안경·핀)과 Sutteang `eyeL_white`뿐이며, 제품 숨은 영역이 있는 `texture` 파츠는 Sutteang `eyeL_white`(눈 = T10)뿐이다. 무늬 복원 한계가 드러난 Himikan 칼라는 파츠 단위로 `flat_cel`(그라데이션) 판정이다 — 파츠 단위 분류로는 칼라 같은 부분 무늬를 `semantic`으로 나누지 못한다. T15가 아직 없으므로 조건 판단은 **보류**(T15 후 다시 판단).

### 남은 제한

- (a) 정지 포즈: 이음새 외곽선은 오버레이로 원래 두께를 유지한다(표 1, 분할 변경 없음). 남은 차이는 배경 후광 제거와 상위 레이어 자기 림의 밝아지는 혼합 교정이다. 오버레이 텍셀은 상위 메쉬를 1–2px 넓혀 노출 합계를 0.3–1.0% 줄인다(숨은 텍셀 수는 그대로). Cubism PSD 인계에서는 상위 레이어에 이 선 텍셀이 포함되고 하위 레이어 림은 깨끗한 색이 된다(Cubism 가져오기 미검증).
- (b) 분류의 "노출 밴드"는 합성 단계에 노출 맵이 없어 숨은 영역 인접 시드(반경 `clamp(0.06·hh, 12, 48)`)로 대신한다. 파츠 단위 분류라 한 파츠 안의 부분 무늬(칼라)는 구분하지 못한다.
- (c) 셀 스냅은 원화 3종 중 Himikan 목에서만 숨은 텍셀에 쓰였다(그라데이션 판정·시드 비율 규칙). 레벨셋은 조화 보간한 군집 지시 함수의 argmax라 경계가 이음새에서 대체로 직선으로 이어지며 곡선 경계의 굽음은 따라가지 않는다.
- (d) 전 층 홀드아웃 파츠의 제품 변경은 수치화되지 않는다(T14 제한 (a) 그대로). 수동 보완 `fill_occlusion`의 새 색 합성은 단위 테스트로만 확인했다(원화 실행 없음).
- (e) 브라우저 대체 경로(`engine.ts` `expandFillSeparated`)는 고급 합성을 하지 않고 `diffusion`으로 표시만 한다(과제 지시대로 Rust 전용).
- (f) 임계값(선화 L* −25·열림 2px, 질감 σ 3.0, 셀 ρ 0.15/0.75, 시드 비율 0.10, 혼합선 t ≥ 0.15·수직 잔차 12 + 0.2·|d|·ΔE ≥ 4, 상위 선화 L* −20·소유 탐색 3px, 하위 어두워짐 L* 3, F 창 4px)은 원화 3종 + Himikan 정답과 육안으로만 정했다.
- (g) Windows 앱에서 모션·`유형` 칩·보완 배너 표시를 확인하지 않았다(`미검증`). PSD 인계는 이미지 레이어 텍스처 림 색만 바뀌며 Cubism 가져오기는 검증하지 않았다.

## T10 눈 실루엣 밀폐 · 단일 눈 변형장 · 클리핑 단일 규칙 (2026-11-18, 리뷰 수정 반영)

- 실행: `cd src-tauri && cargo test` (69/69 — T10 신규: `rig::eye_seal` 4(개구부 선이 모든 가시 흰자/홍채를 감쌈, 휴지 포즈에서 모든 눈 레이어 정확 항등, 분석 눈 fixture 누출·틈 ≥ 99% 감소·행별 상한·시선 극값 홍채 이동 동일·감김 하강 비감소, Rust/TS 눈 동등성 fixture), `templates::t10_eye_seal_on_1024_artwork`, `templates::t10_rest_composite_matches_baseline_synthesis_exactly`; 갱신: 포락선 145포즈, 클리핑 규칙, 감긴 눈꺼풀·기울어진 머리 곡률·메쉬 뒤집힘 판정, T13b 요구량 채택 규칙), `cargo clippy --all-targets -- -D warnings` (통과), `npm test` (50/50 — 신규 5: 눈 동등성 fixture(프레임 1e-9, `deform` 1e-6), 휴지 정확 항등, 개구부 0 높이·반감김 접합·시선 이동량, `eyeSeal` 포즈 = `evaluateRigFrame` 표정(1e-12), 클리핑 단일 규칙), `npx tsc --noEmit` (통과). Windows 앱(WebView2) 표시는 확인하지 않았다(`미검증`).
- 비교 기준: **HEAD**(`7434d0d`) 릴리스 CLI(`artifacts/t10/bin-before/`, 94포즈) — `fillMask`·정지 합성·T13·T14 "전". **legacy**: T10 CLI `--legacy-eye`(기존 눈 식·클리핑 규칙, 같은 145포즈) — 밀폐 검사기는 HEAD에 없으므로 누출/틈/끊김 "전"은 이것이다. **T10**: `artifacts/t10/bin-after/`. 같은 입력·플래그:

```sh
flat2rig-cli [--legacy-eye] --eval-holdout --eval-truth artifacts/t14/himikan-truth artifacts/design-review/input.png artifacts/t10/<mode>/himikan
flat2rig-cli [--legacy-eye] --eval-holdout ref/ritteang/Rritteang.png artifacts/t10/<mode>/ritteang
flat2rig-cli [--legacy-eye] --eval-holdout ref/sutteang/2d_.png       artifacts/t10/<mode>/sutteang
tools/evaluation/.venv/Scripts/python.exe -I tools/evaluation/t10_report.py <원화> artifacts/t10/before/<원화> artifacts/t10/legacy/<원화> artifacts/t10/after/<원화>
```

전체 표(포즈 묶음별·이음새별)는 `artifacts/t10/report.md`(Git 제외). 실행 시간(평가 포함) Himikan 18.4초, Rritteang 47.2초, Sutteang 33.2초 — 템플릿 합성을 기준 합성과 한 번 더 돌리는 정지 불변 병합 때문에 늘었다(이전 T10 11.7 / 32.3 / 22.7초).

### 리뷰 수정 (회귀 2건 + 측정 1건)

1. **Rritteang 윗눈꺼풀 선 끊김 18 → 61(직전 T10) → 0.** 검사기 오탐이 아니었다. 원인: L 눈 u≈251–267(머리 좌표, 눈 중심 159.7) 구간에서 속눈썹 띠가 휴지 포즈에 앞머리(`hair_front`)에 가려 있는데, 그 아래 윗눈꺼풀 텍셀은 3–6px의 얇은 숨은 획(T13b 5b 블록, 윗선 획을 따라감)뿐이었고 ArtMesh 윤곽이 이 얇은 획을 덮지 않아(`rig.json` 삼각형 포함 검사로 확인) 눈꺼풀이 앞머리 아래로 내려오면 그 열에 눈꺼풀이 없었다(얼굴 피부가 보임). legacy도 같은 구멍이 있었고(감김 묶음 9 + 슬픔 9), 직전 T10은 같은 구멍이 더 많은 포즈에서 드러났다. 진단 자료: `artifacts/t10/break_{before,after}_closed_L.png`(포즈 래스터 소유자 색 지도: 빨강 = 윗눈꺼풀, 파랑 = 아랫눈꺼풀, 회색 = 피부, 초록 = 가리는 레이어), 새 CLI 산출물 `eye_seal/<pose>_owner.png`(16-bit 포즈 소유자). 수정:
   - **속눈썹 선 연장**(`templates.rs` 5c): 앞머리·눈썹·안경 등 상위 레이어가 휴지에 가린 곳에서 윗눈꺼풀 숨은 영역을 개구부 속눈썹 선(`EyeFrames`)을 따라 가시 속눈썹 띠 두께의 중앙값(최소 `EYE_LASH_KEEP_ED`·eyeDist, 메쉬가 덮는 두께)만큼 잇는다.
   - **윗/아랫눈꺼풀 텍셀은 개구부 보간으로 접히지 않는다**: 속눈썹 선 아래 윗눈꺼풀 텍셀(눈 끝 날개)은 속눈썹과 함께 평행이동하고, 아랫선 위 아랫눈꺼풀 텍셀도 마찬가지다(접혀 사라지면 눈 끝에서 선이 끊김).
   - **감김선이 개구부 모양을 따른다**: 눈 중심 열에서 기존 감김 이음새(`eye_v + 0.022·eyeDist`)가 개구부 높이에서 차지하는 비율 λ를 모든 열에 써서, 개구부가 얇은 눈 끝은 거의 움직이지 않는다(중심 하강량 동일).
   - **개구부 선은 가시 흰자∪홍채의 가장 큰 8-연결 성분에서만 만든다**(속눈썹 날개 안 흰 점이 끝 구간을 키우지 않도록).
2. **정지 기본 포즈 합성 변화 0px(원화 3종).** 원인: 직전 T10의 정지 유효 합성 변화(Rritteang 175px)는 `deform`이 아니라(정지 경로는 변형을 쓰지 않음) **커진 숨은 영역**이 반투명 `hair_side_L` 가장자리 아래로 비친 것이었다(145포즈 포락선만으로도 HEAD 대비 8,986px가 ≤ 3레벨 달라짐). 수정:
   - **휴지 포즈에서 `deform_eye` 정확 항등**: 흰자/홍채의 개구부 밖 접힘은 눈이 움직일 때만(감김·눈웃음·표정 형상, 홍채는 시선 포함) `EYE_FOLD_ONSET_CLOSE`(0.05)까지 선형으로 켜진다. Rust/TS 테스트가 휴지 포즈에서 모든 눈 레이어(개구부 밖 포함) 점이 비트 단위로 같음을 확인한다.
   - **템플릿 합성 정지 불변 병합**: 현재 합성 결과를 기준 합성(기존 눈 식 + 기존 94포즈 = HEAD 동작)과 병합한다. 기준의 숨은 텍셀·텍셀 색은 그대로 두고(숨은 영역 감소 없음), 기준에 없는 숨은 텍셀은 그 픽셀에서 전체 레이어 정지 합성(텍스처 RGB, 알파 = mask ∪ fillMask, CLI와 같은 `core::blend_straight_pixel`)이 8-bit로 같을 때만 추가한다(Rritteang 원화 알파 최대가 253이라 "위 레이어가 255"만으로는 판정할 수 없어 정확 비교로 바꿨다). 1024 fixture: 추가 숨은 텍셀 17,894, 합성 변화 0(테스트).
   - 측정: 세 원화 모두 `foreground.png`·`effective_foreground.png`가 HEAD와 **0px** 다르다(ΔE 0, 알파 차 0).
3. **검사기 누출 판정 교체(`t10-v3`)**: 눈을 뜬 행은 "같은 열에서 눈꺼풀 픽셀보다 위"가 아니라 **변형된 개구부 선**(휴지 속눈썹/아랫선 표본을 실제 윗/아랫눈꺼풀 `deform`으로 옮긴 선, 1px 허용) 밖의 흰자/홍채를 센다. 개구부 범위 밖은 숨은 텍셀이 보일 때만 센다. 열 판정은 눈 끝에서 눈꺼풀 날개가 흰자 아래로 감기는 곳과, 시선이 흰자 아래 숨은 영역을 정상적으로 드러내는 곳을 누출로 잘못 셌다. 감은 눈(`eye=0`)은 여전히 흰자/홍채 표본 전부를 센다. legacy도 같은 판정으로 다시 쟀다.
4. **Rritteang 홀드아웃 흰자 18.75 → 21.59(직전) → 18.75**: 측정 변화였다. 이 파츠는 가시 흰자 전체가 홀드아웃되는 경우(`wholeLayerHeldOut`, 가시 맥락 0)라 값이 숨은 영역 집합에 따라 다시 합성된 색만 반영했고, 직전 T10에서는 커진 흰자 숨은 영역이 시드 없는 필드 결과를 바꿨다. 정지 불변 병합 후 기존 숨은 텍셀과 색이 HEAD와 같아 값도 같다. 제품 가시 텍스처는 처음부터 바뀌지 않았다.

### 검사기 정의 (`src-tauri/src/rig/eye_seal.rs`, `t10-v3`, CLI `stats.json` `eyeSeal`)

노출 맵과 같은 ArtMesh 래스터·클리핑 규칙(앱의 흰자/홍채 알파 페이드 없음 — 기하만으로 판정), 눈마다 머리 좌표 1px 격자. 포즈: 포락선 `eyeSeal` 51 + `eye` 5 = 56포즈 × 눈 2 = 112행. (a) 누출: 위 3항. (b) 틈 = 윗눈꺼풀 아랫변 바로 아래에서 흰자/아랫눈꺼풀까지 이어지는 피부·배경 표본 + 흰자 아래에서 아랫눈꺼풀까지의 피부·배경 표본. (c) 끊김 = 윗눈꺼풀도 가리는 상위 레이어도 없는 열 묶음 수(처음과 마지막 윗눈꺼풀 열 사이). 휴지 포즈 값을 빼서 휴지 초과분을 결함으로 본다(`eye=0` 누출은 원시 값).

### 표 1 — 표정 × 시선 극값 누출 / 틈 / 끊김 (휴지 초과 px, legacy → T10, 112행 합)

| 원화 | 누출 | 틈 | 윗눈꺼풀 선 끊김 | 결함 행 |
| --- | --- | --- | --- | --- |
| 1024px 합성 fixture(`t13_fixture_1024`, 전 파이프라인) | 29,456 → 395 | 23,811 → 180 | 0 → 0 | 107 → 17 |
| Himikan | 25,144 → 557 | 8,370 → 180 | 0 → 0 | 108 → 82 |
| Rritteang | 344,081 → 1,700 | 87,301 → 0 | **18 → 0** | 104 → 74 |
| Sutteang | 48,873 → 198 | 36,245 → 144 | 0 → 0 | 91 → 49 |

포즈 묶음별 누출(legacy → T10): Himikan closed 4,617 → 18 · smile 5,661 → 27 · sad 5,076 → 234 · 윙크 3,424/2,579 → 45/139 · 깜빡임 0.15/0.5 474/2,699 → 0/5 · 시선만 극값 82 → 82(같음). Rritteang closed 49,779 → 324 · smile 86,004 → 297 · sad 70,767 → 720 · 윙크 43,722/43,515 → 126/198 · 깜빡임 4,002/37,371 → 0/0 · 시선 0 → 0. Sutteang closed 8,766 → 27 · smile 12,528 → 18 · sad 10,089 → 117 · 윙크 6,255/6,300 → 0/18 · 깜빡임 3,666 → 0 · 시선 14 → 14(같음).

**0건이 되지 않은 항목(완료 조건 미충족)과 원인**:
- 감김·미소·슬픔·윙크 누출: 감긴 눈 끝 열에서 흰자 숨은 영역이 눈꺼풀/안경테 사이로 남는 조각과 감김선 위 메쉬 현 조각. Rritteang R `closed` (401, 863)·L (724, 795) — 행당 17–48px, 바깥 눈 끝 안경테와 감긴 선 사이 흰자 숨은 텍셀. Himikan L `sad` (370, 394) 18px, R `sad` (300, 413) 8px, Sutteang R `sad` (508, 373) 13px(현 조각). 앱은 `eye ≤ 0.03`에서 흰자 알파, `≤ 0.06`에서 홍채 알파가 0이라 완전히 감긴 표정에서는 보이지 않지만 기하 검사는 결함으로 남긴다.
- 윙크 누출(눈 뜬 쪽, 눈웃음 0.15 + 머리 기울기 ±4°): Rritteang R `winkR` (398, 883) 22px, Himikan `winkR` 139px — 눈 끝 너머 머리카락 아래의 흰자 숨은 영역(T13 요구량으로 눈 끝 바깥까지 확장됨)이 머리 기울기에서 드러난다(숨은 영역 모양 문제, T11/T13 후속).
- 시선만 극값(눈 뜸) Himikan 82px·Sutteang 14px: legacy와 **같은 값** — 눈 끝에서 휴지 속눈썹 선 위로 홍채/흰자 숨은 텍셀이 보이는 원화 형상(T10 이전부터).
- 틈: 윙크에서 뜬 눈의 바깥 눈 끝 열(Himikan `winkL` R (251, 391) 14px·`winkR`, Sutteang `winkL` R (395, 385) 16px) — 눈웃음으로 눈꺼풀이 들릴 때 눈 끝 피부 열.
- 1024 fixture 잔여(누출 395, 틈 180): 시선 극값의 R 바깥 눈 끝 홍채(376, 342)와 왼윙크 R 눈 끝(391, 333).

### 표 2 — 움직임 범위가 줄지 않음 (최대값, legacy → T10, px)

| 원화 | 시선 극값 홍채 중심 이동 | 감김(closed) 윗눈꺼풀 하강 | smile | sad | 윙크 L/R | 깜빡임 0.5 |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 fixture | 27.30 → 54.70 (전체 최대; 시선 묶음 동일) | 최대 68.91 → 85.65 | | | | |
| Himikan | 6.9 → 6.9 | 26.6 → 31.9 | 7.7 → 16.7 | 46.9 → 56.5 | 11.9/12.0 → 20.8/21.0 | 9.3 → 12.7 |
| Rritteang | 16.4 → 16.4 | 92.1 → 112.0 | 38.5 → 74.4 | 142.0 → 173.5 | 49.9/46.5 → 86.0/82.3 | 36.0 → 47.9 |
| Sutteang | 10.9 → 10.9 | 41.7 → 50.0 | 11.9 → 26.2 | 73.2 → 88.0 | 22.4/15.0 → 36.7/29.2 | 14.6 → 19.8 |

시선 계수(0.045/0.025·eyeDist)·`PARAM_DEFS`·표정 곡률 계수는 그대로이고 시선 극값 홍채 이동은 전/후 같다(테스트로 고정). 부분 깜빡임+시선 행의 홍채 중심 이동은 Rritteang 20.3 → 18.7, Himikan 7.9 → 7.3, Sutteang 12.6 → 11.6으로 작다: 시선 이동 뒤 홍채가 줄어드는 개구부를 따라 압축되는 중심이 달라서이며 시선 범위 축소가 아니다. 감김 하강량은 모든 표정 묶음에서 같거나 늘었다.

### 표 3 — 숨은 영역(`fillMask`)·정지 합성 (HEAD → T10)

- 가시 `mask` 분할·정지 합성: 세 원화 모두 `foreground.png`·`effective_foreground.png` HEAD와 **0px** 차이(ΔE 0, 알파 차 0). 1024 fixture도 테스트로 0.
- `fillMask` 감소 파츠 **없음**. 변경: Himikan face 26,632 → 26,703 · eyeR/L_white 3,890/3,220 → 4,520/7,972 · eyeR/L_iris 1,724/1,750 → 5,848/6,336 · eyeR/L_upper 564/560 → 858/833. Rritteang face 259,179 → 260,502 · eyeR/L_white 79,608/84,193 → 83,324/89,283 · eyeR/L_upper 4,374/3,823 → 5,488/4,485. Sutteang face 53,072 → 53,108 · eyeR/L_white 8,665/8,117 → 10,264/10,152 · eyeR/L_iris 3,437/3,294 → 4,842/4,474 · eyeR/L_upper 999/1,017 → 1,161/1,172. 나머지 동일. 추가분은 모두 정지 합성을 바꾸지 않는 위치에만 들어갔다(2항).

### 표 4 — T13 이음새 커버리지 · 노출 (HEAD 94포즈 → T10 145포즈; 괄호 = legacy 145포즈)

| 원화 | 커버리지 `holes` | `lowerMisses` | 노출 래스터 구멍 px 합 |
| --- | --- | --- | --- |
| 1024 fixture | 0 → 0 | 3,734(94포즈) → 6,610 | 1,888(결합 전/후 같음) |
| Himikan | 52 → 103 (189) | 7,177 → 14,628 (15,040) | 17,558 → 20,854 (22,261) |
| Rritteang | 11,871 → 11,970 (11,970) | 78,905 → 138,532 (137,575) | 848,407 → 1,168,653 (1,168,405) |
| Sutteang | 1,973 → 1,973 (1,973) | 46,261 → 69,082 (69,920) | 7,114 → 17,577 (20,577) |

증가분은 포락선이 94 → 145포즈로 커진 것이다(같은 145포즈 legacy보다 Himikan·Sutteang은 적고 Rritteang은 같다). 눈 관련 이음새의 커버리지 구멍은 Himikan·Sutteang 0, Rritteang `eyeR/L_lower/hair_back` 756/342(T13b부터 있던 구멍, 전후 같음). 이음새 요구량은 `deform`에서 자동 재계산되고 눈 이음새는 `legacyEye` 비교로 줄지 않는다.

### 표 5 — T14 눈 파츠 (HEAD → T10, ΔE00 평균 / p95)

- Himikan 정답 비교: eyeR_white 1.06/4.82 → 0.87/4.15(팔레트 0.91 → 0.95), eyeL_white 1.07/4.78 → 0.97/4.64, eyeR_iris 13.44 → 13.36, eyeL_iris 10.66 → 10.76, eyeR_upper 38.51 → 38.53, eyeL_upper 33.82 → 33.82.
- 홀드아웃: Himikan 흰자 10.33 → 10.33 / 7.85 → 7.85, 홍채 30.39 → 29.74 / 35.02 → 34.35. **Rritteang 흰자 18.75 → 18.75 / 22.94 → 22.94**(4항), 윗눈꺼풀 19.24 → 19.15 / 21.31 → 21.35. Sutteang 흰자 25.47 → 25.75 / 30.64 → 30.99, 홍채 24.29 → 25.54 / 29.21 → 29.41(끊김 2/5 → 1/2, 2/2 → 1/1), 윗눈꺼풀 37.14 → 37.29 / 34.22 → 34.16. Sutteang 흰자·홍채의 소폭 변화는 가시 맥락이 남는 밴드 홀드아웃에서 추가된 숨은 텍셀(정지 불변 위치)이 시드 필드를 바꾼 것이다.

### 비교 이미지 (`artifacts/t10/`, Git 제외; 왼쪽 legacy · 오른쪽 T10, 하드 래스터)

- `compare_<원화>_a.png`: 휴지 · 감김(중앙/시선 −−) · 깜빡임 0.5. `compare_<원화>_b.png`: 미소 · 슬픔(중앙/시선 ++) · 왼윙크 · 오른윙크. 원본 렌더 `artifacts/t10/{legacy,after}/<원화>/eye_seal/*.png`(+ `_owner.png`), 도구 `artifacts/t10/{crop_eyes,zoom,owner_classes}.py`.
- 육안: legacy는 감긴 눈 아래로 흰자 띠·두 줄 눈꺼풀, 미소/슬픔의 흰자 초승달, 압축된 속눈썹의 점선이 보인다. T10은 감긴 눈이 하나의 이어진 속눈썹 선(미소 ∩, 슬픔 ∪)이고 흰자/홍채가 보이지 않는다. **안경 교차부(Rritteang)**: 테 안쪽 흰자 띠와 점선형 속눈썹 흔적이 없고, 앞머리 아래 구간도 속눈썹 선이 끊기지 않으며 안경테는 감긴 선 위에 그려진다.

### 남은 제한

- (a) 완료 조건(누출·틈 0건) 미충족: 표 1 잔여. 검사기는 앱 알파 페이드 없이 기하로 센다.
- (b) 개구부 선은 가시 흰자/홍채의 가장 큰 연결 성분에서 16개 열 구간으로 만든다. 흰자가 없는 눈(감긴 원화)은 기본 프레임을 쓴다.
- (c) 표정 형상은 LD 기본 프리셋 값이다(SD는 `roundness` 기준값만 다름). 입 모양은 앱이 대체 레이어로 그리므로 포락선에 싣지 않았다.
- (d) 정지 불변 병합은 템플릿 합성을 두 번 돌린다(CLI 실행 시간 약 1.4–1.6배). 불변은 CLI 유효 합성의 8-bit 혼합 기준이며 앱(Canvas/WebGL) 합성의 반올림까지 같다고 확인하지는 않았다.
- (e) 커버리지 검사는 개구부 밖 흰자/홍채 표본(접힘)을 건너뛴다. 배경 구멍은 노출 래스터가 계속 센다.
- (f) Windows 앱(WebView2)에서 정지 표정/시선 미리보기·재생·GL 경로의 클리핑과 감긴 눈 표시를 확인하지 않았다(`미검증`). PSD 인계는 이미지 레이어만 바뀌며(숨은 영역 증가) Cubism 가져오기는 검증하지 않았다.

## T15 선화·구조 연장 (숨은 윤곽선 포함) — 선화/채색 분리 · 획 연장 · 숨은 윤곽선 (2026-11-19)

- 실행: `cd src-tauri && cargo test` (75/75 — 신규 6: `analysis::lineart` 5(획이 노출 깊이까지 굵기·색 유지 연장·깊이 밖 미생성·가시 텍셀 불변, 반투명 상위 아래 정지 합성 비트 동일, 노출 0·눈썹 이음새 미생성, 실루엣 윤곽선의 숨은 윤곽선 연속(접합부 굵기·색), EDT 전수 대조), `analysis::templates::tests::t15_lineart_keeps_partition_fill_and_rest_composite`(T15 켬/끔 분할·`fillMask`·정지 합성 동일, 가시 텍셀 불변)), `cargo clippy --all-targets -- -D warnings`(통과), `npm test`(51/51 — 신규 1: `fillLines` 선택 필드 왕복·알 수 없는 값 거부·브라우저 대체 경로 `none`·Rust 보고 해석), `npx tsc --noEmit`(통과). Windows 앱(WebView2) 표시는 확인하지 않았다(`미검증`).
- 비교 기준: **HEAD**(`7df73a8`) 릴리스 CLI(`artifacts/t15/bin-before/`, T10 `bin-after`와 바이트 동일) → `artifacts/t15/before2/`. **T15** → `artifacts/t15/after/`. 같은 T15 바이너리의 `--no-lineart`(T15 끔) → `artifacts/t15/nolineart/`: HEAD 출력과 **모든 파일 동일**(Himikan 86 / Rritteang 73 / Sutteang 74 파일; `stats.json`은 `elapsedMs`·`lineart` 키, 새 `lineart/` 폴더 제외; `artifacts/t15/headcmp.py`). 측정 전용 `--lineart-no-rest-gate`(정지 합성 게이트 끔) → `artifacts/t15/nogate/`.

```sh
flat2rig-cli [--no-lineart|--lineart-no-rest-gate] --eval-holdout --eval-truth artifacts/t14/himikan-truth artifacts/design-review/input.png OUT/himikan
flat2rig-cli [...] --eval-holdout ref/ritteang/Rritteang.png OUT/ritteang
flat2rig-cli [...] --eval-holdout ref/sutteang/2d_.png       OUT/sutteang
tools/evaluation/.venv/Scripts/python.exe -I tools/evaluation/t15_report.py <원화> artifacts/t15/before2/<원화> artifacts/t15/after/<원화>
tools/evaluation/.venv/Scripts/python.exe -I tools/evaluation/t15_compare.py artifacts/t15/nolineart/<원화> artifacts/t15/after/<원화> artifacts/t15/compare/<원화>
```

전체 표는 `artifacts/t15/report/<원화>.md`(Git 제외). 새 CLI 출력: `stats.json` `lineart`(파츠별 보고·이중 선화 검사·렌더 목록), `partCoverage[].fillRegion.lineart`, `lineart/line_layer.png`(흐린 원화 위 가시 선화 파랑 · T15 획 빨강 · 숨은 윤곽선 초록 · 게이트 거부 노랑), `lineart/pose_*.png`(극값 포즈 하드 렌더 10장), `lineart/skip_*.png`(건너뛴 획 대표 사례: 원화 · 텍스처, ×4, 접점 빨강).

### 구현 정의 (`src-tauri/src/analysis/lineart.rs`, `t15-v1`)

- **선화/채색 분리**: 채색 레이어 = T9 텍스처(가시 = 원화(T9 림 정리 반영), 숨은 = T9 필드). 선화 레이어 = 가시 증거(가시 알파 ≥ 128, 홀드아웃 제외, T9 이음새 오버레이 텍셀 포함)에서 **강 선화** L* ≤ 파츠 가시 중앙값 − 25(T9 시드·T14 지표와 같은 값), **약 선화** L* ≤ 중앙값 − 15가 강 선화에 8-연결로 이어지면 포함(히스테리시스: 안티앨리어싱된 1px 획이 조각나지 않게), 그 합집합 중 5×5 열림(반경 2)에 사라지는 가는 획. 숨은 텍셀에 선 레이어를 합성해 `tex = α·line + (1 − α)·colour`로 합친다. 가시 텍셀·`mask`·`fillMask`는 바꾸지 않는다.
- **생성 범위**: 숨은 텍셀(오버레이 텍셀 제외) 또는 평가 홀드아웃 픽셀 중, 휴지에 그 텍셀을 덮는 상위 레이어와의 T13 이음새 `maxExposureDepthPx`(+2px 래스터 여유) 이내(파츠 가시 마스크로부터 유클리드 거리). 상위가 눈·눈썹·입이면 생성하지 않는다(T10 `deform_eye`·5b·5c 영역). **선 생성 범위 제한일 뿐 `fillMask`·숨은 채색은 그대로다.**
- **획 연장**(작업 2·4): 숨은 텍셀에 4-인접한 선화 성분이 후보. 후보가 아닌 분류(확신 판정이 아니므로 따로 집계): `seamOutline` = 성분 픽셀의 50% 이상이 상위 레이어 가시 픽셀로부터 2.5px 이내(상위 외곽선·이음새 잔여 — 연장하면 그 외곽선을 복제), `parallel` = 축이 숨은 쪽 방향과 75° 넘게 벌어짐(경계를 따라 달림), `contour` = 실루엣 밀착(실루엣 선은 숨은 윤곽선으로만 잇고, 파츠가 윤곽선 판정이 아니면 잇지 않음), `featureSeam`, `unexposed`, `noRoom`(첫 걸음부터 허용 텍셀 밖), `seamParallel`(연장이 곧바로 이음새·상위 외곽선과 평행해 멈추거나 가드가 텍셀을 모두 지움). 피팅: 창 `clamp(6·폭, 8, 24)`px, 주축·1px 슬랩 중심·가중 이차 중심선. **확신 판정(건너뜀 → `semantic` 플래그)**: 가시 길이 < `max(4px, 1.5·폭)`(`short`; 머리카락은 2σ 안 피팅 획 ≥ 2·결과 길이 ≥ 0.8이면 방향장으로 연장), 슬랩 중심 RMS 잔차 > `max(0.75px, 0.35·폭)`(`fitResidual`), 다른 획을 1.5px 안에서 만남(`crossing`). 진행: 곡률은 연장 중 45° 회전 이내로 제한, 머리카락은 같은 레이어 획 방향장(가우스 σ = `clamp(0.02·hh, 6, 24)`px) 혼합, 허용 텍셀을 벗어나거나 노출 깊이 도달(파츠 원화 획 끝 가늘어짐 길이로 테이퍼), 가시 선화 재연결, 다른 획 접합(교차하지 않음)에서 멈춤. 굵기 = 접점 슬랩 폭을 파츠 중앙 폭의 `[0.5, 1.25]×`로 제한, 색 = 창 안 어두운 절반의 중앙값.
- **숨은 윤곽선**(작업 3): 유효 마스크(`mask ∪ fillMask`, 메쉬와 같은 템플릿 경계) 바깥까지의 정확한 유클리드 거리(Felzenszwalb EDT)로 고리(≤1.5px)를 정의. 파츠의 가시 실루엣 고리 픽셀 중 1.5px 안에 선화가 있는 비율(`outlineCoverage`)이 0.5 이상(고리 ≥ 16px)일 때만 윤곽선 파츠. 숨은 고리를 **자기 레이어의 윤곽선 가시 고리 픽셀에서만** BFS로 전파(접합부 연속), 운반 접선에서 35° 넘게 꺾이면 정지(언더랩 가장자리 모서리는 실루엣이 아님), 굵기·색 = 접합부 국소(반경 4px) 값을 파츠 중앙값으로 `4×폭`에 걸쳐 혼합, 알파 = `clamp(폭 − 거리 + 1, 0, 1)`. 상위 레이어 외곽선이 같은 실루엣을 지나는 곳(공유 실루엣)에서는 그 외곽선을 아래 레이어로 **복사하지 않는다**(T9가 제거한 이음새 잔여와 같고, 상위가 조금만 움직여도 두 번째 외곽선으로 보인다 — 작업 중 Himikan `hair_back`에서 시도했을 때 교차 레이어 이중 선화 1,442px가 생겨 제거).
- **이중 선화 가드**: 진행 중 자기 가시 경계로부터 `2·폭+2`px 안에서 이음새 법선과 60° 넘게 벌어지면 정지, 상위 레이어 가시 외곽선이 `2·폭+3`px 안에서 60° 이내로 평행하면 정지. 그린 뒤 텍셀 단위로 다시 검사(자기 경계 `2·폭+2`px·30°, 상위 외곽선 — 획 `2·폭+3`px·60° / 윤곽선 `2·폭+2`px·30°).
- **정지 합성 게이트**: 바뀌는 숨은 텍셀마다 전체 레이어 정지 합성(`mask ∪ fillMask` 알파, `core::blend_straight_pixel`, CLI 유효 합성·`merge_rest_invariant`와 같은 함수)이 8-bit로 같을 때만 채택, 아니면 색 직선(원래 색 → 선 색) 위에서 합성이 같은 가장 먼 지점(1/64 단계, 색상 유지 — 처음 구현의 채널별 클램프는 Rritteang에서 하늘·분홍 얼룩을 만들어 교체)으로 `clamped`, 불가능하면 `rejected`.

### `merge_rest_invariant`와의 관계 (작업 지시의 확인 항목)

- T15는 `synthesize_template_bank_for_stage`에서 **병합 뒤** 실행한다(`apply_lineart_pass`). 원시 합성 안에서 돌리면 병합이 기준 합성(기존 94포즈·기존 눈 식)에 이미 있던 숨은 텍셀의 색을 기준 실행 색으로 되돌리므로, 그 텍셀의 선은 기준 실행이 그린 선(94포즈 깊이)으로 제한되고 현재 실행이 더 깊게 그린 부분은 버려진다. 병합 뒤에 실행하므로 **`merge_rest_invariant`가 버린 T15 선 텍셀은 0**이다. 남은 T15 텍셀 위치(보고 `texelsInBaseHidden` / `texelsInCurrentOnlyHidden`)는 원화 3종 모두 기준 숨은 텍셀 안이다(현재 실행에만 있는 숨은 텍셀 0).
- 대신 T15 자체 정지 합성 게이트가 반투명 상위 아래에서 선을 흐리게 하거나 버린다: Himikan 채택 33 · 클램프 0 · 거부 0, **Rritteang 채택 128 · 클램프 261 · 거부 2**(원화 알파 최대 253이라 모든 숨은 텍셀이 정지 합성에 2/255 비친다 → 선 대부분이 흐려짐), Sutteang 7 / 0 / 0. 원인 = 비트 단위 정지 불변(HEAD와 0px)과 반투명 상위 레이어. 대안 측정: `--lineart-no-rest-gate`로 게이트를 끄면 Rritteang 유효 정지 합성이 263px 바뀌고(가시 합성 0px) T14 선화 끊김은 세 원화 모두 게이트 켠 결과와 같다 — 현재 남은 끊김은 게이트 때문이 아니다. 허용 오차(예: 정지 합성 1레벨) 도입은 정지 불변 정의를 바꾸는 것이라 하지 않았다.

### 표 1 — 불변 확인 (HEAD → T15)

| 원화 | `fillMask` 파츠별 | 가시 분할(`owner.png`)·파츠별 가시 px | 정지 합성 변화 px (가시 `foreground.png` / 유효 `effective_foreground.png`) | 숨은 텍셀 색 변화 px (`fill_only.png`) | 노출·래스터 구멍·이음새 `holes` |
| --- | --- | --- | --- | --- | --- |
| Himikan | 전 파츠 동일(감소 0) | 0px · 동일 | 0 / 0 | 33 | 동일(숨은 423,153 · 노출 83,027 · `holes` 103) |
| Rritteang | 전 파츠 동일(감소 0) | 0px · 동일 | 0 / 0 | 380 | 동일(1,603,517 · 426,800 · 11,970) |
| Sutteang | 전 파츠 동일(감소 0) | 0px · 동일 | 0 / 0 | 5 | 동일(1,134,910 · 335,923 · 1,973) |

`deform`·`PARAM_DEFS`·이음새 요구량·결합 변형·숨은 영역 생성은 바꾸지 않았다(파츠별 `fillMask` 값은 `artifacts/t15/report/<원화>.md` "Invariants").

### 표 2 — T10 눈 검사기 (휴지 초과 px, HEAD → T15)

| 원화 | 누출 | 틈 | 윗눈꺼풀 끊김 | 결함 행 |
| --- | --- | --- | --- | --- |
| Himikan | 557 → 557 | 180 → 180 | 0 → 0 | 82 → 82 |
| Rritteang | 1,700 → 1,700 | 0 → 0 | 0 → 0 | 74 → 74 |
| Sutteang | 198 → 198 | 144 → 144 | 0 → 0 | 49 → 49 |

눈·눈썹·입 레이어와 그 아래 얼굴 숨은 텍셀에는 선을 그리지 않는다(`featureSeam`).

### 표 3 — T14 지표 전 → 후 (노출 밴드, 값이 바뀐 행; 끊김 = 끊긴 획 / 가로지르는 획)

| 세트 | 파츠 (이음새) | 노출 px (표본) | 선화 끊김 | 그래디언트 불연속 ΔE00 | ΔE00 평균 / p95 |
| --- | --- | ---: | --- | --- | --- |
| Himikan 정답 | 합계 | – | 129/131 → 129/131 | – | – |
| Himikan 정답 | clothes | 5,198 (5,179) | 3/3 → 3/3 | 3.14 → 3.16 | 2.52 / 13.10 → 2.72 / 13.66 |
| Himikan 정답 | clothes ← hair_side_L | 2,056 | 0/0 → 0/0 | 2.89 → 2.93 | 2.02 / 13.23 → 2.52 / 15.09 |
| Himikan 홀드아웃 | 합계 | – | **9/9 → 7/9** | – | – |
| Himikan 홀드아웃 | clothes (← hair_side_R / ← hair_side_L) | 55,977 | 2/2 → 0/2 (1/1 → 0/1, 1/1 → 0/1) | 1.26 → 1.17 | 11.26 / 23.53 → 11.20 / 23.44 |
| Himikan 홀드아웃 | hair_side_R ← hair_front | 385 | 4/4 → 4/4 | 3.11 → 3.04 | 4.90 / 30.34 → 4.90 / 30.34 |
| Himikan 홀드아웃 | hair_side_L ← hair_front | 667 | 3/3 → 3/3 | 3.13 → 2.98 | 4.75 / 27.38 → 4.90 / 27.38 |
| Rritteang 홀드아웃 | 합계 | – | 11/11 → 11/11 | – | – |
| Rritteang 홀드아웃 | hair_side_L (← hair_front 850px) | 1,081 | 0/0 → 0/0 | 2.74 → 2.40 (2.29 → 1.87) | 5.82 / 31.11 → 5.51 / 28.98 (4.15 / 21.76 → 3.75 / 18.92) |
| Rritteang 홀드아웃 | hair_side_R ← hair_front | 881 | 1/1 → 1/1 | 2.87 → 2.86 | 4.20 / 17.25 → 4.16 / 17.25 |
| Rritteang 홀드아웃 | acc_glasses (← hair_front 3,786px) | 6,545 | 8/8 → 8/8 | 9.28 → 9.19 | 23.87 / 54.93 → 23.63 / 54.78 |
| Rritteang 홀드아웃 | face (← eyeR_lower / ← eyeL_lower) | 45,443 | 0/0 → 0/0 | 0.48 → 0.48 | 5.49 / 14.32 → 5.53 / 14.46 (3.34 → 3.46, 5.90 → 5.94) |
| Sutteang 홀드아웃 | 합계 | – | **10/12 → 7/12** | – | – |
| Sutteang 홀드아웃 | clothes ← hair_side_R | 128,559 | 2/2 → 1/2 | 2.15 → 2.02 | 6.73 / 16.75 → 6.78 / 16.94 |
| Sutteang 홀드아웃 | hair_side_R ← hair_front | 5,341 | 3/3 → 2/3 | 2.23 → 2.18 | 5.59 / 15.83 → 5.63 / 16.34 |
| Sutteang 홀드아웃 | hair_side_L ← hair_front | 2,816 | 1/1 → 0/1 | 2.14 → 2.13 | 4.34 / 12.64 → 4.58 / 13.90 |
| Sutteang 홀드아웃 | hair_back ← hair_front (전 층) | 9,295 | 0/0 → 0/0 | 4.56 → 4.47 | 10.20 / 21.93 → 10.21 / 21.97 |

- **나빠진 항목**: Himikan 정답 `clothes` ΔE00 2.52 → 2.72(p95 13.10 → 13.66), 옆머리 L 아래 2.02 → 2.52 — 숨은 윤곽선 37px가 PSD 의상 정답과 정확히 같은 위치가 아니어서(어긋난 어두운 선 픽셀의 ΔE가 큼) 평균이 오른다. 홀드아웃에서 끊김을 줄인 Sutteang `hair_side_L`·`clothes`는 선 위치가 조금 어긋나 ΔE p95가 오른다(12.64 → 13.90, 16.75 → 16.94). Rritteang 홀드아웃 `face`(아랫눈꺼풀 아래 밴드) 3.34 → 3.46: 홀드아웃 픽셀의 상위 판정(가장 가까운 상위 레이어 BFS)이 눈 레이어가 아닌 곳에서 선이 그려진 것으로 보이며(평가 전용 경로), 제품 경로의 눈 아래 얼굴 숨은 텍셀은 바뀌지 않는다(표 2 불변, `featureSeam`).
- **Himikan 정답 `hair_back` 125/125 끊김은 그대로다.** 원인(작업 중 확인, `artifacts/t15/wip/hb_cmp.png`): 정답의 끊김은 (a) 머리 실루엣을 따라 앞머리·옆머리 반투명 가장자리 아래에 있는 뒷머리 외곽선(공유 실루엣 — 상위 외곽선 복사 금지 규칙과 정지 게이트에 걸림)과 (b) 옆머리 아래로 이어지는 뒷머리 가닥 가장자리 선(가시 쪽 가닥 선이 옆머리 가장자리에 붙어 있어 `seamOutline`)이다. 공유 실루엣 윤곽선을 허용하고 게이트를 끈 시험에서는 125 → 83으로 줄었지만, 같은 실행에서 교차 레이어 이중 선화가 1,442px 생겨 채택하지 않았다.

### 표 4 — 연장 통계 (T15, 원화별 합계; 파츠별은 `artifacts/t15/report/<원화>.md`)

| 원화 | 숨은 영역에 닿는 선화 성분 | 연장(+방향장 / 접합) | 건너뜀(`semantic`): short / fitResidual / crossing | 후보 아님: seamOutline / parallel / seamParallel / contour / featureSeam / unexposed / noRoom | 획 텍셀 | 숨은 윤곽선 길이 px (텍셀) | 게이트 채택 / 클램프 / 거부 |
| --- | ---: | --- | --- | --- | ---: | --- | --- |
| Himikan | 93 | 0 (+0 / 0) | 0 / 1 / 0 | 87 / 0 / 2 / 1 / 0 / 0 / 2 | 0 | 37 (33) | 33 / 0 / 0 |
| Rritteang | 180 | 8 (+0 / 0) | 0 / 3 / 0 | 138 / 3 / 6 / 9 / 8 / 0 / 5 | 86 | 296 (305) | 128 / 261 / 2 |
| Sutteang | 119 | 3 (+0 / 0) | 0 / 2 / 0 | 83 / 17 / 6 / 2 / 0 / 0 / 6 | 7 | 0 (0) | 7 / 0 / 0 |

- **건너뛴 획은 확신 판정 대상(연장 + 건너뜀) 중 Himikan 1/1, Rritteang 3/11, Sutteang 2/5**이며 모두 `fitResidual`이다(Himikan `clothes` (352,732) 잔차 1.19 > 0.75; Rritteang `hair_back` (876,958) 1.34 > 1.05, `hair_front` (966,534) 1.76 > 0.75 · (930,572) 0.88 > 0.75; Sutteang `clothes` (972,702) 1.17 > 0.75, `hair_side_R` (332,414) 1.29 > 1.05). 대표 이미지 `artifacts/t15/after/<원화>/lineart/skip_*.png` — Rritteang `hair_front`는 핀 끝을 감싸는 U자 외곽선이라 이차 중심선으로 설명되지 않는다(건너뜀이 맞음). Himikan은 확신 판정 대상이 1개뿐이고 그것이 건너뛰어져 연장 0이다.
- **연장하지 않은 대부분은 `seamOutline`이다**(87/93, 138/180, 83/119). 숨은 텍셀은 정의상 상위 레이어 아래에 있으므로, 숨은 영역에 닿는 선화 성분 대부분은 상위 레이어 가장자리를 따라 붙은 짧은 성분(상위 외곽선의 분할 잔여, 또는 상위와 맞닿은 하위 가장자리선)이다. 처음 구현은 이 규칙 없이 연장했고(Himikan `body`: 머리카락 끝 외곽선 잔여를 목 아래 몸통 숨은 영역으로 대각선 연장) 고개 회전 포즈에서 교차 레이어 이중 선화 160px가 생겼다(`artifacts/t15/wip/himikan_eval7`, `artifacts/t15/wip/dbg2.png` — 빨강 = 당시 연장된 잔여 선). 규칙을 끄면 같은 성분 대부분은 `short`(55건) 건너뜀이 된다 — 연장할 근거가 있는 획이 아니다.
- 접합부 연속(윤곽선, 접합부 4px 창): Himikan `clothes` 굵기 차 평균 0.00px · ΔE00 평균 5.73(합성 쪽은 4×폭에 걸쳐 파츠 중앙색으로 혼합되어 차이가 남), Rritteang `body`·`neck`은 `report/ritteang.md`.

### 표 5 — 숨은 윤곽선 대상 구간 (`fillMask`가 실루엣을 넓힌 고리 px, T15)

| 원화 | 파츠 | 윤곽선 판정 (가시 실루엣 선화 비율) | 숨은 고리 px | 눈·눈썹·입 아래 | 노출 깊이 밖 | 노출 밴드 안 | 그중 공유 실루엣(상위 외곽선 위치) | 윤곽선 도달 | 상위 외곽선 가드로 생략 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Himikan | hair_back | 아니오 (0.57, 가시 고리 < 16px) | 5,976 | 1 | 2,706 | 3,269 | 1,998 | 0 | 0 |
| Himikan | body / neck / face | 아니오 (0.24 / 0.02 / 0.03) | 1,746 / 477 / 917 | 0 / 0 / 34 | 1,576 / 471 / 684 | 170 / 6 / 199 | 53 / 4 / 26 | 0 | 0 |
| Himikan | clothes | 예 (0.65) | 279 | 0 | 66 | 213 | 20 | 37 | 6 |
| Himikan | hair_side_R / L | 아니오 (0.47 / 0.42) | 170 / 206 | 0 | 135 / 151 | 35 / 55 | 31 / 51 | 0 | 0 |
| Rritteang | hair_back | 아니오 (0.23) | 24,201 | 1,949 | 11,046 | 11,206 | 4,837 | 0 | 0 |
| Rritteang | body | 예 (0.62) | 6,563 | 0 | 6,367 | 196 | 44 | 21 | 2 |
| Rritteang | neck | 예 (0.64) | 1,340 | 0 | 892 | 448 | 88 | 275 | 135 |
| Rritteang | clothes | 아니오 (0.48) | 2,023 | 0 | 0 | 2,023 | 849 | 0 | 0 |
| Rritteang | face | 아니오 (0.34) | 5,314 | 560 | 3,585 | 1,169 | 272 | 0 | 0 |
| Rritteang | hair_side_R / L / hair_front | 아니오 (0.13 / 0.03 / 0.07) | 312 / 372 / 1,555 | 0 | 286 / 302 / 1,422 | 26 / 70 / 133 | 18 / 50 / 59 | 0 | 0 |
| Sutteang | 7 파츠 합계 | 모두 아니오 (hair_back 0.00, body 0.05, neck 0.01, clothes 0.17, face 0.10, hair_side_R/L 0.05 / 0.16) | 14,119 | 86 | 8,503 | 5,530 | 2,428 | 0 | 0 |

- **빠진 이유**: (1) 대부분의 파츠는 가시 실루엣에 T14 기준(중앙값 − 25 L*) 선화가 없다(`outlineCoverage` < 0.5). Himikan은 윤곽선이 옅은 채색(얼굴 0.03), Rritteang 머리카락은 어두운 채색이라 외곽선이 중앙값보다 25 L* 이상 어둡지 않다(hair_front 0.07). Sutteang 턱선은 분할에서 **`clothes` 가시 영역에 배정**되어 있어(`artifacts/t15/dbg_face.py` — 턱선 픽셀의 정지 소유자가 `clothes`, 얼굴 오버레이로 넘어온 것은 11px) 얼굴 실루엣 선화 비율이 0.10이다(분할 문제, T11 수동 보정 범위). 원화 규칙상 윤곽선이 없는 실루엣에 새 윤곽선을 그리지 않는다. (2) 노출 깊이 밖 고리는 포즈 포락선에서 드러나지 않는다(새 획 생성 금지 범위). (3) 공유 실루엣은 상위 외곽선 복사 금지(이중 선화). (4) Rritteang `neck` 135px는 옷깃 외곽선 바로 아래·옆으로 평행해 가드로 생략.
- Rritteang `clothes`(0.48)는 기준 0.5 바로 아래다. 기준을 낮추는 것은 판정 기준을 바꾸는 것이라 하지 않았다.

### 표 6 — 이중 선화 (신규 0 확인)

- **방법**: (a) 휴지 기하 — 남은 T15 선 텍셀 중 자기 가시 경계(휴지 이음새)로부터 `2·폭+2`px 안에서 그 경계와 30° 이내로 평행한 텍셀 수(`seamParallelTexels`). (b) 모션 — 포락선 20포즈(옆머리·고개 극값, 머리/몸 반대 6, 머리카락 부호 조합 8)를 노출 맵과 같은 ArtMesh 래스터로 그려, 노출된 T15 선 픽셀(커버리지 ≥ 0.5)에서 획 법선 양쪽으로 `2·폭+3`px를 훑어 같은 레이어 채색을 사이에 두고 **다른 레이어의 선화가 30° 이내로 평행**하게 나오면 교차 레이어 이중 선화로 센다(`crossLayerDoublePx`). 같은 규칙을 원화 가시 선화에 적용한 값(`baselineCrossLayerDoublePx`, HEAD에도 있는 패턴)을 함께 낸다.
- **결과(T15)**: `crossLayerDoublePx` **Himikan 0 · Rritteang 0 · Sutteang 0**, `seamParallelTexels` 0 · 0 · 0. 기준선(원화 선화의 같은 패턴) 159 · 3,909 · 1,145px. 같은 레이어 평행(머리카락 결 등, 참고) 0 · 2 · 0px.
- 작업 중 경과(코디네이터 지적 포함): Himikan `hair_side_R` 16px(`hairSide=1`, 앞머리 가닥 외곽선 3px 옆으로 옆머리 가장자리선이 연장됨) → 상위 외곽선 평행 가드로 0. Rritteang 21 → 11 → 2 → 0: 뒷머리 실루엣 선이 자유 획으로 옷 아래까지 연장되던 것(실루엣 밀착 성분은 항상 윤곽선 분류), 옷 주름선 연장이 옆머리 외곽선과 얕은 각도로 만나던 것(획 진행·텍셀 가드 60°, 굵기 상한 1.25×). 정지 포즈 이중 선화는 정지 합성 0px 변화로 생기지 않는다. T9가 제거한 얼굴 잔여 선화·경계 후광은 되살리지 않는다(가시 텍셀 불변, 눈·입 아래 얼굴 숨은 텍셀 미변경, 이음새 오버레이 텍셀 미변경).

### 비교 이미지 (`artifacts/t15/compare/<원화>/pose_*.png`, Git 제외; 왼쪽 `--no-lineart`(= HEAD) · 오른쪽 T15, ×3, 하드 래스터)

- 포즈: `hairSide=±1`, `angleX=±30`, `angleZ=±30`, `opp:angleX±/bodyX∓`, `hairSign:+++/---`. 창: 얼굴 R/L 가장자리(눈 바깥 0.75·eyeDist, 눈–턱 중간 높이), 목(턱 아래 0.45·eyeDist).
- **육안**: 세 원화 모두 얼굴·목 가장자리에서 전/후 차이가 거의 보이지 않는다. 얼굴 가장자리는 표 5대로 윤곽선 파츠가 아니어서 숨은 윤곽선이 없고, Rritteang 목 옆선(275px)·옷 주름 연장(86 텍셀)은 정지 게이트로 대부분 흐려져(클램프 261) 하드 렌더에서 옅다. **완료 조건 "옆머리 흔들림·고개 회전 포즈에서 얼굴·목 가장자리 윤곽선이 연속"은 충족했다고 보고하지 않는다.**

### 실행 시간 (릴리스 CLI, `--eval-holdout`(+Himikan `--eval-truth`) 포함, 순차 실행)

| 원화 | HEAD | T15 | T15 `--no-lineart` |
| --- | ---: | ---: | ---: |
| Himikan | 22.6 s | 27.6 s | 22.1 s |
| Rritteang | 62.5 s | 70.4 s | 67.6 s |
| Sutteang | 43.5 s | 49.9 s | 47.1 s |

T15 패스(합성 1회당; 평가 실행은 홀드아웃 합성 포함 2회)와 CLI 진단 출력(20포즈 이중 선화 검사·10포즈 렌더)이 늘어난 몫이다.

### 남은 제한

- (a) **제품 효과가 작다.** 원화 3종에서 연장된 획 0 / 8 / 3개, 숨은 윤곽선 37 / 296 / 0px. 확신 판정이 원인이 아니라(건너뜀 1 / 3 / 2), 숨은 영역에 닿는 선화 대부분이 상위 가장자리 잔여(`seamOutline`)이고 실루엣 대부분에 T14 기준 선화가 없는(옅은·어두운 채색) 원화 특성, 공유 실루엣 복사 금지, 정지 게이트 때문이다.
- (b) T14 완료 조건: 홀드아웃 선화 끊김은 Himikan 9 → 7, Sutteang 10 → 7로 줄었고 Rritteang 11 → 11, Himikan 정답 129 → 129는 그대로다 → **부분 충족**.
- (c) 정지 불변(8-bit)을 지키기 위해 Rritteang(알파 ≤ 253)의 선은 대부분 흐려진다(클램프 261, 거부 2). 허용 오차를 도입하면 정지 합성이 HEAD와 달라진다(게이트 끔 시 유효 합성 263px).
- (d) 이중 선화 검사는 포락선 20포즈 하위 집합과 하드 래스터 기준이며 앱 알파·물리 흔들림은 포함하지 않는다. 기준선 패턴(원화 선화 159–3,909px)이 많아 검사 정의 자체가 넓다(서로 다른 물체의 두 가장자리도 센다).
- (e) 윤곽선·선화 판정은 L* 중앙값 − 25 고정 규칙(T14와 같음)이라 색선·어두운 파츠의 외곽선을 놓친다. 분할이 외곽선을 다른 파츠에 준 경우(Sutteang 턱선)는 고치지 않는다(T11).
- (f) 브라우저 대체 경로(`engine.ts`)는 선 합성을 하지 않는다. 결과 구분용 선택 필드 `Part.fillLines`(`extended` | `none`, 없으면 T15 이전 = 미상)를 추가했고 대체 경로·수동 보완은 `none`이다(schema v3 유지, 구 프로젝트는 필드 없음으로 읽힘). 숨은 텍셀 단위 `semantic` 표시는 `RegionReport.lineart.semantic`(획 위치·사유·값·한계)에만 있고 프로젝트 마스크로 저장하지 않는다(T11 입력).
- (g) Windows 앱(WebView2) 모션 표시, PSD 인계 후 Cubism 가져오기 검증 없음(`미검증`).

---

## B2 이음새 결합 변형 (`hair_body`) 검증 (2026-11-19)

- **대상 커밋**: B2 체크포인트 1(`7f99341` — `derive_torso_frame` + 야코비안 안전 `hair_body` 이음새 결합 + Rust/TS 일치).
- **실행 검증**:
  - `cargo test --manifest-path src-tauri/Cargo.toml --lib b2_hair_body_parity_fixture_matches_rust_deform`: **1 passed, 0 failed** (공유 fixture `tests/fixtures/hair-body-parity.json`, 기울어진 머리 `-15°` · `clothes`+`hands`+저알파 노이즈 · 8극단 포즈).
  - `npx vitest run tests/pipeline.test.ts -t "B2 hair_body"`: **1 passed** (역할 기반 커스텀 ID 포함 Rust↔TS 일치).
- **4개 데이터셋 Before → After 수치 비교 (`artifacts/b2/before/` → `artifacts/b2/after/`)**:

| 데이터셋 | `seamCoverage.holes` | `lowerMisses` | `exposure.holePxTot` | `exposure.holePxMax` | `maxDepthPx` | `flipped/inv` |
| --- | --- | --- | --- | --- | --- | --- |
| `himikan-truth` | 1,453 → **797** (-45.1%) | 9,192 → **7,269** | 61,588 → **25,100** (-59.2%) | 2,668 → **1,173** (-56.0%) | 43.2 → 43.2 | 0 → **0** |
| `st-himikan` | 1,409 → **474** (-66.4%) | 7,895 → **6,474** | 63,515 → **23,907** (-62.4%) | 2,692 → **1,182** (-56.1%) | 23.2 → 23.2 | 0 → **0** |
| `st-sutteang` | 11,225 → **1,969** (-82.5%) | 35,662 → **13,615** | 1,250,984 → **1,043,197** (-16.6%) | 31,120 → **22,091** (-29.0%) | 198.4 → **61.6** | 0 → **0** |
| `default` | 103 → **83** (-19.4%) | 14,628 → **8,369** | 19,478 → **10,176** (-47.8%) | 748 → **341** (-54.4%) | 204.2 → **182.2** | 0 → **0** |

- **주요 이음새별 구멍 감소 및 칼라 보호**:
  - `himikan-truth`: `hair_side_R > clothes` `380 → 42`, `hair_side_L > clothes` `359 → 39`, `clothes > hair_back` `462 → 320`, `face > hair_back` `0 → 0`.
  - `st-himikan`: `hair_front > clothes` `739 → 77`, `clothes > hair_back` `630 → 378`, `clothes > neck` `4 → 0` (목 뒤 칼라 열 보호), `neck > hair_back` `36 → 10`.
  - `st-sutteang`: `clothes > hair_back` `7,423 → 1,371`, `hair_front > clothes` `2,902 → 124`, `hands > hair_back` `427 → 33`, `hair_front > hands` `268 → 176`.
  - `default`: `hair_side_R > clothes` `37 → 29`, `hair_side_L > clothes` `30 → 17`, `face > hair_back` `0 → 0`, `hair_side_L > face` `0 → 0`.
  - **잔여 한계(솔직 기록)**: `himikan-truth`에서는 `clothes > hair_back`(`|dx|/ed = 0.32..0.55`)과 `clothes > neck`(`|dx|/ed = 0.00..0.51`)이 동일 칼라 띠에서 겹쳐 `clothes > neck`이 `3 → 29`(rel `26.6 → 26.6` 불변)로 소폭 남았다. 다만 실제 렌더 노출 구멍 픽셀 총합은 `61,588 → 25,100`(-59.2%)으로 감소했다.

- **금지 불변량 증명 표**:

| 불변 항목 | 검증 근거 | 결과 |
| --- | --- | --- |
| 얼굴(`face`)·눈(`eyeL_*`, `eyeR_*`)·입(`mouth_*`)·입 위 앞머리(bangs, `y ≤ mouth.y` & `|pu| ≤ 0.70·ed`) | `b2_hair_body_parity_fixture_matches_rust_deform` (`tests/fixtures/hair-body-parity.json`) | 최대 변위 차이 `0.000 px` (`< 1e-6`) |
| 머리카락 흔들림 진폭(`hair_front`/`hair_side`/`hair_back`의 `ParamHairFront = ±1`에서 `raw - head` 차이) | `b2_hair_body_parity_fixture_matches_rust_deform` | **불변 (유지됨)** |
| 모션 최대 각도(`ParamAngleX/Y/Z = ±30`, `ParamBodyAngleX/Z = ±10`, `PARAM_DEFS`) | `src/types.ts` · `src-tauri/src/core.rs` · `b2_hair_body_parity_fixture_matches_rust_deform` | **불변 (축소 없음)** |

- **비교 이미지 (`artifacts/b2/compare/`)**:
  - `artifacts/b2/compare/himikan_truth_compare.png` 및 `artifacts/b2/compare/himikan_truth_zoom3x.png`
  - `artifacts/b2/compare/st_himikan_compare.png`
  - `artifacts/b2/compare/st_sutteang_compare.png`
  - `artifacts/b2/compare/default_compare.png`
  - 육안 점검: `angleZ_30`, `angleZ_m30`, `same3_headpmm_breath1`에서 옆머리·어깨 머리카락이 꺾임/찢어짐 없이 매끄러운 S-곡선으로 어깨·칼라를 따라가며, `st-sutteang`의 어깨 지그재그 찢어짐과 배경 노출이 크게 줄었다.
- **미검증 항목**: Tauri 데스크톱 앱(WebView2) 실화면 육안 확인, Cubism Editor 가져오기.
