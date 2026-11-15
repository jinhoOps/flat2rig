# flat2rig

일러스트 한 장을 **앱 안에서 표정 선택·움직임 조정·재생·저장**까지 완결하는 로컬 데스크톱 Live2D/2D 리깅 스튜디오(Tauri v2 + Rust + React/TypeScript)입니다. Live2D Cubism Editor 후반 작업용 PSD·리깅 초안·파라미터 CSV 내보내기도 선택적인 고급 출력으로 지원합니다.

Rust 백엔드는 `ort` CPU 런타임으로 ONNX 얼굴·랜드마크·배경·파츠 추론(YOLOv3, HRNetV2, U2NetP, MobileSAM)을 실행하고, 4단계 하이브리드 실루엣 합성(`analysis/templates.rs`), `delaunator` 기반 윤곽·역할별 ArtMesh 생성, `glam` 변형 샘플 및 극단 포즈 품질 검증을 수행합니다. React·TypeScript는 4단계 워크플로우(`1. 준비 → 2. 표정 → 3. 조정 → 4. 재생`) UI와 실시간 WebGL/Canvas 2D 재생을 제공합니다.

## 실행

Windows의 Rust MSVC 도구 모음, Visual Studio C++ Build Tools, WebView2 Runtime, Node.js 20 이상이 필요합니다.

```powershell
npm ci --legacy-peer-deps
npm run desktop
```

`ort`의 ONNX Runtime 라이브러리는 최초 Cargo 빌드 시 내려받습니다. 모델 5종(`models/manifest.json`)은 별도 준비합니다:

```powershell
# 1) U2NetP 배경 모델 다운로드 (앱 내 'ONNX 모델 관리'에서도 SHA-256 검증 자동 다운로드 지원)
powershell -ExecutionPolicy Bypass -File scripts/download-model.ps1

# 2) 얼굴 검출(YOLOv3), 28점 랜드마크(HRNetV2), MobileSAM 인코더/디코더 내보내기
python tools/model-export/export_models.py
```

앱 상단 헤더의 `모델` 또는 작업 설정의 `ONNX 모델 관리` 모달에서 5개 역할별 SHA-256 검증 상태를 확인하고 `.onnx` 파일을 연결하거나 `U2NetP`를 내려받을 수 있습니다. 모델 없이도 테두리 색상 제거와 기하 템플릿 분할·수동 보정이 가능합니다.

```powershell
npm run desktop:build
# 빠른 로컬 검증용 디버그 빌드
npm run desktop:build -- --debug
```

결과: `src-tauri/target/release/flat2rig.exe` 또는 `src-tauri/target/debug/flat2rig.exe`. 이 명령은 실행 파일을 생성하며 설치 프로그램은 만들지 않습니다. 실행 파일을 다른 폴더로 옮길 때는 같은 폴더에 생성된 DLL도 포함해야 합니다.

## 작업 흐름 (4단계 하이브리드 실루엣 적층)

1. **이미지 열기 (`1. 준비`)**: 정면 흉상 PNG/JPEG를 엽니다(드래그 앤 드롭 시 활성 작업이 있으면 저장/버림/취소 확인). 긴 변 최대 2,048px로 작업하며 원본도 `.flat2rig`(Schema v2)에 보관합니다. `YOLOv3 + HRNetV2`가 얼굴 후보와 28점·리깅 랜드마크를 추출하며(`crown`은 기하 추정 표시), 다중 얼굴 전환과 `F`/`L` 수동 수정 보존을 지원합니다.
2. **체형 선택 및 4단계 하이브리드 적층 (`1. 준비`)**: `체형 실루엣 (자동 판별 LD/SD · 일반 등신 LD · SD·치비)` 설정 후 순차 적층 또는 `전체 한 번에 생성 (1→4단계)`를 실행합니다.
   - **1단계 · 실루엣 소체 (`base_silhouette`)**: 앞머리 잔선·레이스 구멍·안경다리 간섭 없는 순수 캐노니컬 SD/LD 소체 실루엣(`body`, `neck`, `face`, `eyeR_white`, `eyeL_white`, `mouth_inner`)을 생성합니다.
   - **2단계 · 이목구비 적층 (`features_overlay`)**: 원화의 볼터치·코(`face_detail` → `face` 클리핑), 홍채(`eye*_iris` → `eye*_white` 클리핑), 상·하 눈꺼풀, 눈썹, 입술을 소체 위에 적층합니다.
   - **3단계 · 의상·소품 적층 (`clothes_overlay`)**: 의상(`clothes*`, `clothes_hands`)과 안경(`acc_glasses`)을 적층합니다.
   - **4단계 · 머리카락 적층 (`hair_overlay`)**: 앞·옆·뒷머리·모자·머리장식(`hair_front*`, `hair_side*`, `hair_back*`, `hat`, `acc_*`)을 최종 Z-순서로 적층합니다. 2~4단계를 바로 실행하면 누락된 선행 단계를 자동으로 먼저 수행해 병합하며, 사용자가 `확인함` 또는 수동 편집한 파츠는 보존됩니다.
3. **세부 보정 및 클리핑 (`1. 준비`)**: 레이어 그룹/개별 눈 토글, `Space+드래그` 캔버스 패닝, `P`(클릭 `+`/`-` 프롬프트 보정 — 후보 #2 자동 우선 선택), `B`/`E` 브러시, `+ 앞머리/모자/의상/안경` 가변 레이어 추가를 지원합니다. 가시 마스크(`mask`)와 가림 보완(`fillMask`)을 엄격히 분리하며, `face_detail → face` 및 `eye*_iris → eye*_white` 클리핑이 정지·모션 상태 모두에 적용됩니다(솔로 미리보기 제외).
4. **표정·조정·재생 (`2. 표정`, `3. 조정`, `4. 재생`)**: 8종 기본 표정 프리셋과 사용자 커스텀 표정 저장·불러오기, 눈·입 대체 모양 미세 조정, 실시간 눈 깜빡임·호흡·유휴 고개/시선 재생 및 **투명 배경 PNG 저장**을 지원합니다.
5. **프로젝트 저장 및 선택적 고급 내보내기 (`고급`)**: `.flat2rig` ZIP(커스텀 표정·보완 마스크·원본 포함) 저장·복원과 함께, Cubism Editor 후반 작업용 레이어드 PSD, `masks/*.visible.png` / `*.fill.png`, 자체 `rig-draft.json`, 파라미터 CSV, `validation.json`을 내보냅니다.

**전달 경계:** PSD(`src/io.ts`)는 이미지 레이어·오프셋·`normal` 블렌드 모드만 전달합니다. `multiply`/`additive` 합성 모드는 실제 Cubism Editor 가져오기 검증 전까지 지원 표시나 조용한 치환을 하지 않습니다. 자체 메쉬·디포머·표정 프리셋은 Cubism에 자동 적용되지 않으며, CSV 샘플을 연결하지 않으면 참고용 목록만 출력합니다. Live2D Cubism Editor가 설치되어 있지 않아 실제 Editor 가져오기(`cubismEditorImport`)는 `"not-tested"`로 기록됩니다.

**자동화 한계:** ONNX 모델과 MobileSAM 클릭 보정은 초안·보조 수단이며, 경계 색 확장은 생성형 인페인팅이 아닌 비생성형 침식·샘플링 보완으로 가려진 눈·입·머리카락 내부 디테일을 새로 그려 넣지 않습니다. 정교한 가림 복원은 외부 PNG 교체가 필요합니다.

## 검증

```powershell
npm test
npx tsc --noEmit
npm run build
cargo test --manifest-path src-tauri/Cargo.toml --lib
cargo clippy --manifest-path src-tauri/Cargo.toml --all-targets -- -D warnings
# Rust vs Python ONNX 정합성 및 CPU 지연 측정 (tools/model-export/.venv 필요)
tools/model-export/.venv/Scripts/python.exe tools/evaluation/evaluate.py
# 데스크톱 빌드 후 실제 WebView/IPC/드래그 드롭 확인
npm run test:native
```

Rust CLI로 4단계별 파츠 분리 및 체형 실루엣을 직접 확인하려면:

```powershell
cargo run --manifest-path src-tauri/Cargo.toml --bin flat2rig-cli -- --stage base_silhouette --proportion auto INPUT.png artifacts/stage1
cargo run --manifest-path src-tauri/Cargo.toml --bin flat2rig-cli -- --stage full INPUT.png artifacts/native-smoke
cargo run --manifest-path src-tauri/Cargo.toml --bin flat2rig-cli -- --analyze --full INPUT.png artifacts/analyze.json
```

CLI는 `--stage`(`full`, `base_silhouette`, `features_overlay`, `clothes_overlay`, `hair_overlay`, `base_body`)와 `--proportion`(`auto`, `ld`, `sd`)를 지원하며, 등록된 ONNX 모델이 있으면 자동 사용하고 없으면 기하 폴백으로 파츠 PNG·합성 PNG·윤곽 메쉬·품질 경고를 포함한 `stats.json`을 생성합니다.

`npm run dev`는 UI 검증용 브라우저 모드입니다. 이 모드에는 TypeScript 연산 대체 경로가 있으며 ONNX 모델 연결과 네이티브 저장 대화상자는 데스크톱 앱에서만 사용합니다.

문서 및 사양: [제품 계획](docs/product-plan.md) · [개발 태스크](docs/development-tasks.md) · [MVP 사양](docs/mvp-spec.md) · [검증 기록](docs/validation.md) · [모델 안내](models/README.md) · [구현 계획](docs/implementation-plan.md).

화면 테마는 [DESIGN.md](DESIGN.md)를 기준으로 [src/design.css](src/design.css)에 구현했습니다. 불투명 캔버스와 도킹 레이아웃을 유지하고, 패널·HUD·모달에 글래스 토큰을 적용합니다. 투명 효과/모션 축소 설정과 미지원 엔진 폴백을 포함합니다.
