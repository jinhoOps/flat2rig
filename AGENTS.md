# flat2rig Project Rules & Invariants

flat2rig는 일러스트 1장을 Live2D Cubism Editor 후반 작업용 파츠 PSD와 리깅 초안으로 준비하는 로컬 데스크톱 앱(Tauri v2 + Rust + React/TypeScript)이다. 아래 불변 규칙은 코드·계획·내보내기 수정 시 항상 지킨다.

## 1. 좌표계와 좌우(L/R) 기준
- 작업 캔버스 좌표는 **좌상단 `(0, 0)`**, X 오른쪽 증가, Y 아래쪽 증가, 픽셀 단위다.
- 모든 `L`/`R` 파츠와 랜드마크(`eyeL`, `eyeR`, `browL`, `browR` 등)는 **캐릭터 기준**이다.
  - 정면 이미지에서 캐릭터 왼쪽(`L`)은 화면 오른쪽(`x`가 더 큰 쪽), 캐릭터 오른쪽(`R`)은 화면 왼쪽(`x`가 더 작은 쪽)이다.
- 검출·랜드마크·크롭 좌표 변환은 임의 비율 추정이 아니라 `src-tauri/src/analysis/transforms.rs`의 `glam` 역행렬 왕복을 따른다.

## 2. 파츠 마스크와 가림 보완 분리
- **보이는 영역(`mask`)과 가림 보완 영역(`fillMask`)을 섞지 않는다.**
  - `mask`의 파츠별 합은 원본 전경 알파와 일치하는 비겹침 분할을 유지한다.
  - 가림 보완(`fillMask` / 확장 텍스처)은 상위 파츠 아래 겹침을 허용하며 별도 필드(`masks/<id>.fill.bin`)로 보존한다.
- **수동 수정 보존**: 사용자가 수정·승인(`approved`)한 랜드마크나 파츠 마스크는 자동 재분석 시 동의 없이 덮어쓰지 않는다.
- **가변 다중 머리카락 지원**: 18개 기본 ID(`PART_DEFS` / `IDS`) 외에 추가되는 머리카락 레이어(`hair_front_*`, `hair_side_*`, `hair_back_*` 등)는 고정 18개 배열에 묶어 누락시키지 않고 합성 순서·변형·PSD 내보내기에 포함한다.
- 비생성형 경계 확장을 구현·수정할 때는 마스크 최외곽 경계(머리색·선화 혼입 구간)를 침식한 뒤 내부 색을 샘플링해 확장한다.

## 3. Rust ↔ TypeScript 리깅/메쉬 일치
- `src-tauri/src/core.rs`의 `deform` / 파라미터 범위와 브라우저 대체 경로 `src/engine.ts`는 동일한 수식·축 부호·피벗 규칙을 유지한다. 한쪽을 수정하면 반대쪽과 테스트(`tests/pipeline.test.ts`, `cargo test`)를 함께 갱신한다.
- Live2D 파라미터 기본 범위(`PARAM_DEFS` in `src/types.ts`):
  - `ParamAngleX` / `ParamAngleY` / `ParamAngleZ`: `-30 .. 30`
  - `ParamEyeLOpen` / `ParamEyeROpen`: `0 .. 1` (기본 `1`)
  - `ParamMouthOpenY`: `0 .. 1` (기본 `0`)
  - `ParamHairFront`: `-1 .. 1` (기본 `0`)
- 머리카락 변형은 ID 문자열 접두사(`hair`)에만 의존하지 말고 파츠 역할/속성 기반으로 동작하도록 확장한다.

## 4. ONNX 모델 단일 진실 원천
- 모델 파일명, 입력/출력 shape, RGB/BGR 순서, 정규화(`mean`/`std`/`scale`), 후처리 버전, SHA-256, 라이선스는 `models/manifest.json`을 단일 기준으로 삼는다.
- 모델 관련 변경은 `onnx-model-pipeline` 스킬 절차에 따라 Python 원본 대비 Rust `ort` 오차를 검증한 뒤 앱 상태를 노출한다.
- 앱 실행 경로는 Rust + `ort` CPU를 유지하며, Python(`tools/`)은 export·검증·평가에만 사용한다.

## 5. Cubism 인계 경계 정직성
- PSD(`src/io.ts`)는 이미지 레이어·오프셋·블렌드 모드(`normal`)만 전달한다. Cubism이 특정 영문 레이어 이름(`PartFace` 등)을 마법처럼 자동 리깅 규격으로 인식한다고 가정하지 않는다.
- 자체 `rig-draft.json`이나 참고용 `parameters-reference.csv`를 Cubism 공식 직접 가져오기 포맷으로 표기하지 않는다.
- 검증하지 않은 항목(예: 실제 Cubism Editor 가져오기)은 `validation.json`이나 문서에서 통과한 것처럼 표시하지 않는다.
