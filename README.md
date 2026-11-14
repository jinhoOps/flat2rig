# flat2rig

일러스트 한 장을 **Live2D Cubism Editor 후반 작업용 파츠와 기초 리깅 자료**로 준비하는 로컬 데스크톱 앱입니다.

Rust + Tauri v2를 기반으로 `ort`로 ONNX 배경 제거, `delaunator`로 메쉬 생성, `glam`으로 변형 샘플을 계산합니다. React·TypeScript는 보정 UI와 Canvas 2D 미리보기를 제공합니다.

## 실행

Windows의 Rust MSVC 도구 모음, Visual Studio C++ Build Tools, WebView2 Runtime, Node.js 20 이상이 필요합니다. 검증 환경은 Node 20.16 / Rust 1.99입니다.

```powershell
npm ci --legacy-peer-deps
npm run desktop
```

`ort`의 ONNX Runtime 라이브러리는 최초 Cargo 빌드 시 내려받습니다. 모델은 별도입니다.

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download-model.ps1
```

앱 오른쪽에서 `models/u2netp.onnx`를 연결합니다. 모델 없이도 단순 배경의 테두리 색상 제거와 마스크 보정이 가능합니다. 앱 실행 중 이미지 업로드나 모델 자동 다운로드는 하지 않습니다.

```powershell
npm run desktop:build
# 빠른 로컬 검증용 빌드
npm run desktop:build -- --debug
```

결과: `src-tauri/target/release/flat2rig.exe` 또는 `src-tauri/target/debug/flat2rig.exe`. 이 명령은 실행 파일을 생성하며 설치 프로그램은 만들지 않습니다. 실행 파일을 다른 폴더로 옮길 때는 같은 폴더에 생성된 DLL도 포함해야 합니다.

## 작업 흐름

1. 정면 흉상 PNG/JPEG를 엽니다. 긴 변 최대 2,048px로 작업하며 원본도 프로젝트에 보관합니다.
2. `F`로 얼굴 피부 영역을 지정하고 `L`로 양쪽 눈·입·턱·얼굴 상단 위치를 맞춥니다.
3. 18개 파츠 초안을 생성합니다. `B`/`E`로 마스크를 보정하고 레이어별로 확인합니다.
4. 가림 보완 또는 동일 캔버스 크기의 PNG 교체로 빈 영역을 정리합니다.
5. 리깅 미리보기에서 눈·입·작은 고개 회전·머리카락 흔들림을 확인합니다.
6. 프로젝트를 `.flat2rig`로 저장하거나 Cubism 인계 ZIP/PSD를 내보냅니다.

내보내는 ZIP에는 레이어드 PSD, 파츠·마스크 PNG, 다시 열 수 있는 프로젝트, 자체 리깅 JSON, 랜드마크 도해, 파라미터 CSV와 작업 가이드가 포함됩니다.

**전달 경계:** PSD는 이미지 레이어를 전달합니다. 자체 메쉬·디포머·변형 키는 Cubism에 자동 적용되지 않습니다. CSV 샘플을 연결하지 않으면 참고용 목록을 출력하며, 샘플을 연결한 CSV도 대상 Cubism 버전에서 검증해야 합니다.

**자동화 수준:** ONNX 모델은 배경 제거용입니다. 눈·입·머리 등 세부 파츠는 랜드마크 기반 기하 초안이며 의미 분리 AI가 아닙니다. 가림 보완은 인접색 확장으로, 생성형 인페인팅이 아닙니다.

## 검증

```powershell
npm test
npm run build
cargo test --manifest-path src-tauri/Cargo.toml --lib
cargo clippy --manifest-path src-tauri/Cargo.toml --all-targets -- -D warnings
# 데스크톱 빌드 후 실제 WebView/IPC/드래그 드롭 확인
npm run test:native
```

Rust 경로만 이미지로 확인하려면:

```powershell
cargo run --manifest-path src-tauri/Cargo.toml --bin flat2rig-cli -- INPUT.png artifacts/native-smoke models/u2netp.onnx
```

마지막 인자로 랜드마크 JSON을 추가할 수 있습니다. CLI는 파츠 PNG, 합성 PNG, 메쉬 JSON과 처리 통계를 저장합니다. PSD 생성은 앱의 내보내기 경로를 사용합니다.

`npm run dev`는 UI 검증용 브라우저 모드입니다. 이 모드에는 TypeScript 연산 대체 경로가 있으며 ONNX 모델 연결과 네이티브 저장 대화상자는 데스크톱 앱에서만 사용합니다.

구현 범위와 데이터 계약: [MVP 사양](docs/mvp-spec.md). 모델 출처와 설정: [모델 안내](models/README.md).

화면 테마는 [DESIGN.md](DESIGN.md)를 기준으로 [src/design.css](src/design.css)에 구현했습니다. 불투명 캔버스와 도킹 레이아웃을 유지하고, 패널·HUD·모달에 글래스 토큰을 적용합니다. 투명 효과/모션 축소 설정과 미지원 엔진 폴백을 포함합니다.
