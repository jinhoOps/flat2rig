# flat2rig

일러스트 한 장을 **Live2D Cubism Editor 후반 작업용 파츠와 기초 리깅 자료**로 준비하는 로컬 데스크톱 앱입니다.

Rust + Tauri v2를 기반으로 `ort`로 ONNX 얼굴·랜드마크·배경·파츠 추론(YOLOv3, HRNetV2, U2NetP, MobileSAM), `delaunator`로 윤곽·랜드마크 기반 메쉬 생성, `glam`으로 변형 샘플과 극단 포즈 품질 검증을 계산합니다. React·TypeScript는 보정 UI와 Canvas 2D 미리보기를 제공합니다.

## 실행

Windows의 Rust MSVC 도구 모음, Visual Studio C++ Build Tools, WebView2 Runtime, Node.js 20 이상이 필요합니다. 검증 환경은 Node 20.16 / Rust 1.99입니다.

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
# 빠른 로컬 검증용 빌드
npm run desktop:build -- --debug
```

결과: `src-tauri/target/release/flat2rig.exe` 또는 `src-tauri/target/debug/flat2rig.exe`. 이 명령은 실행 파일을 생성하며 설치 프로그램은 만들지 않습니다. 실행 파일을 다른 폴더로 옮길 때는 같은 폴더에 생성된 DLL도 포함해야 합니다.

## 작업 흐름

1. 정면 흉상 PNG/JPEG를 엽니다. 긴 변 최대 2,048px로 작업하며 원본도 프로젝트(Schema v2)에 보관합니다.
2. `YOLOv3 + HRNetV2` ONNX가 얼굴 후보와 28점·리깅 랜드마크를 자동 추출합니다(`crown`은 기하 추정 표시). 다중 얼굴 후보를 전환하거나 `F`/`L`로 수동 수정할 수 있으며, 재분석 시 수동 수정 보존 여부를 확인합니다.
3. 세부 18파츠 또는 눈·입 통합 모드로 MobileSAM + 랜드마크 분할 초안을 생성합니다. `P`(클릭 `+`/`-` 프롬프트 보정 및 후보 #1~#4 선택)와 `B`/`E` 브러시로 다듬고, 생성 출처(`모델`/`프롬프트`/`기하`/`수동`)와 검토 상태(`초안`/`확인됨`/`미해당`)를 분리 관리합니다.
4. 가림 보완은 가시 마스크(`mask`)와 분리된 `fillMask` 레이어로 경계 12px 인접색 외삽(`neighbor`)만 수행합니다. 생성형 인페인팅이 아니므로 가려진 눈·입·머리카락 내부 디테일은 복원되지 않으며, 보완 오버레이·원본 대비 합성 차이 보기 또는 동일 크기 PNG 교체로 검증합니다.
5. 리깅 미리보기에서 내보내기와 동일한 윤곽·랜드마크 기반 Delaunay 메쉬로 변형과 극단 포즈(±30°)·구멍 횡단 품질 경고를 확인합니다.
6. 프로젝트를 `.flat2rig`로 저장하거나 Cubism 인계 ZIP/PSD를 내보냅니다.

내보내는 ZIP에는 `compositeOrder`와 보완 레이어까지 반영한 레이어드 PSD, 파츠·분리 마스크 PNG(`masks/*.visible.png`, `masks/*.fill.png`), 다시 열 수 있는 Schema v2 프로젝트, 자체 리깅 JSON, 랜드마크 도해, 파라미터 CSV, 검증 요약(`validation.json`)과 작업 가이드가 포함됩니다.

**전달 경계:** PSD는 이미지 레이어를 전달합니다. 자체 메쉬·디포머·변형 키는 Cubism에 자동 적용되지 않습니다. CSV 샘플을 연결하지 않으면 참고용 목록을 출력하며, 샘플을 연결한 CSV도 대상 Cubism 버전에서 검증해야 합니다. Live2D Cubism Editor가 설치되어 있지 않아 실제 Editor 가져오기(`cubismEditorImport`)는 `"not-tested"`로 기록됩니다.

**자동화 한계:** ONNX 모델과 MobileSAM 클릭 보정은 초안·보조 수단이며, 가림 보완은 인접 픽셀 외삽(neighbor)으로 가려진 눈·입·머리카락 디테일을 새로 그려 넣지 않습니다. 정교한 가림 복원은 외부 PNG 교체가 필요합니다.

## 검증

```powershell
npm test
npx tsc --noEmit
npm run build
cargo test --manifest-path src-tauri/Cargo.toml
cargo clippy --manifest-path src-tauri/Cargo.toml --all-targets -- -D warnings
# Rust vs Python ONNX 정합성 및 CPU 지연 측정 (tools/model-export/.venv 필요)
tools/model-export/.venv/Scripts/python.exe tools/evaluation/evaluate.py
# 데스크톱 빌드 후 실제 WebView/IPC/드래그 드롭 확인
npm run test:native
```

Rust CLI로 이미지 분석 및 파츠 분리를 확인하려면:

```powershell
cargo run --manifest-path src-tauri/Cargo.toml --bin flat2rig-cli -- INPUT.png artifacts/native-smoke
cargo run --manifest-path src-tauri/Cargo.toml --bin flat2rig-cli -- --analyze --full INPUT.png artifacts/analyze.json
```

CLI는 등록된 ONNX 모델(얼굴·랜드마크·MobileSAM·U2NetP)이 있으면 자동 사용하고, 없으면 기본 비율/기하 폴백으로 파츠 PNG, 합성 PNG, 윤곽 메쉬 JSON, 품질 경고를 포함한 `stats.json`을 생성합니다.

`npm run dev`는 UI 검증용 브라우저 모드입니다. 이 모드에는 TypeScript 연산 대체 경로가 있으며 ONNX 모델 연결과 네이티브 저장 대화상자는 데스크톱 앱에서만 사용합니다.

구현 범위와 데이터 계약: [MVP 사양](docs/mvp-spec.md). 모델 출처와 설정: [모델 안내](models/README.md).

자동 얼굴 검출·랜드마크·파츠 분리의 다음 구현 순서와 완료 조건: [구현 계획](docs/implementation-plan.md).

화면 테마는 [DESIGN.md](DESIGN.md)를 기준으로 [src/design.css](src/design.css)에 구현했습니다. 불투명 캔버스와 도킹 레이아웃을 유지하고, 패널·HUD·모달에 글래스 토큰을 적용합니다. 투명 효과/모션 축소 설정과 미지원 엔진 폴백을 포함합니다.
