# flat2rig Project Rules & Invariants

flat2rig는 일러스트 1장에서 눈·입 모양과 표정을 선택하고 기본 움직임을 세부 조정하여 앱 안에서 저장·재생하는 로컬 데스크톱 앱(Tauri v2 + Rust + React/TypeScript)이다. Cubism Editor 후반 작업용 PSD와 리깅 초안은 선택적인 고급 출력으로 유지한다. 핵심 목표는 **단순 정지 원화 보존이 아니라, 원본 일러스트의 시각적 정체성을 유지하면서 보이지 않는 숨은 영역까지 확장·완성해 자연스럽게 움직이는 캐릭터 파츠를 만드는 것**이다. 제품 목표는 `docs/product-plan.md`, 구현 지시와 우선순위(`소체 피팅 → 숨은 영역 확장 → 자연스러운 합성 → 모션 시 겹침 검증`)는 `docs/development-tasks.md`를 따른다. 아래 불변 규칙은 코드·계획·내보내기 수정 시 항상 지킨다.

## 1. 좌표계와 좌우(L/R) 기준
- 작업 캔버스 좌표는 **좌상단 `(0, 0)`**, X 오른쪽 증가, Y 아래쪽 증가, 픽셀 단위다.
- 모든 `L`/`R` 파츠와 랜드마크(`eyeL`, `eyeR`, `browL`, `browR` 등)는 **캐릭터 기준**이다.
  - 정면 이미지에서 캐릭터 왼쪽(`L`)은 화면 오른쪽(`x`가 더 큰 쪽), 캐릭터 오른쪽(`R`)은 화면 왼쪽(`x`가 더 작은 쪽)이다.
- 검출·랜드마크·크롭 좌표 변환은 임의 비율 추정이 아니라 `src-tauri/src/analysis/transforms.rs`의 `glam` 역행렬 왕복을 따른다.

## 2. 파츠 마스크 · 숨은 영역 확장 · 겹침 분리
- **보이는 영역(`mask`)과 가림 보완·숨은 영역(`fillMask`)을 섞지 않는다.**
  - `mask`의 파츠별 합은 원본 전경 알파와 일치하며(파츠 수만큼의 반올림 오차 이내), 정지 원화의 시각적 정체성을 보존한다. 분할 방식은 입력에 따라 다르다.
    - **레이어 입력(PSD·레이어 디렉터리, 기본 경로)**: 가시 `mask` = 위 파츠에 덮이지 않은 알파 `a_p · Π_{q 위}(1 − a_q)`인 **알파 가중 분할**이다. 안티에일리어싱 내부 경계와 반투명 레이어 픽셀은 여러 파츠가 나눠 가진다. `fillMask` = 덮인 곳의 레이어 알파이므로 `effective_part_mask` = 레이어 알파다. 정의는 `src-tauri/src/analysis/layer_input.rs`와 `src/layerParts.ts`가 같게 유지하고 `tests/fixtures/layer-split-parity.json`으로 검사한다. 픽셀 소유자가 필요한 계산(`visible_owner`, `rest_owner`)은 `mask > 0`인 가장 위 파츠를 쓴다.
    - **단일 이미지 자동 분리(레거시)**: 한 픽셀에 한 파츠인 비겹침 분할을 유지한다(`mobilesam_parts_preserve_partition`).
  - 숨은 영역(`fillMask` / 확장 텍스처)은 상위 파츠 아래 겹침을 허용하며 별도 필드(`masks/<id>.fill.bin`)로 보존한다. 메쉬(`effective_part_mask`)와 최종 텍스처(`texture`)는 `mask`와 `fillMask`를 모두 덮는 완전한 형태를 기준으로 생성한다.
- **부위별 모션 비례 겹침 여유분**: 2~4px 고정 패딩으로 끝내지 않고 원화 해상도와 실제 모션 변형량(`core.rs`/`engine.ts`)을 반영해 부위별 여유분(얼굴 이마·두피, 턱 아래 목 연결, 눈 흰자·홍채 가림 확장, 머리카락 뿌리, 옷깃 뒤 목/몸통)을 확보한다.
  - 여유분은 `deform` 계수를 다른 함수에 손으로 옮겨 적어 계산하지 않는다. 이음새 계약(`SeamGraph`)별로 `deform`을 포즈 포락선에서 수치 샘플링해 산출한다(T13). 기존 `part_overlap_margins`는 폴백이다.
  - 이음새 양쪽 변형장을 결합해 **이음새의 상대 변위**를 줄이는 것은 허용되며 권장된다. 이것은 최대 각도·파라미터 범위 축소(금지)와 다르다.
- **내측 샘플링 · 국소 색상 필드 · 추정 영역 관리**:
  - 비생성형 경계 확장을 구현·수정할 때는 외곽 경계(배경 흰 테두리·선화·이웃 파츠 혼입 구간)를 침식한 뒤 내측 깨끗한 색상을 시드로 삼는 국소 색상 필드(명암 변화 반영)로 연장하며, 경계 후광을 `pad_texture_rgb` 등으로 바깥에 번지게 두지 않는다.
  - 샘플링만으로 복원할 수 없는 복잡 무늬·선화는 추정 영역(`fillMask` / `fillKind`)으로 구분해 관리하고 사용자가 수동 보정할 수 있게 유지한다.
  - 숨은 영역 품질은 허용 모션에서 실제로 드러나는 **노출 영역(`ExposureMap`)**을 기준으로 측정·개선한다. 영역 유형(`flat/cel`·`line`·`texture`·`semantic`)에 따라 합성기를 달리하며, 단일 조화 보간 필드로 그림자 경계·선화를 뭉개 놓고 완료 처리하지 않는다.
  - 생성형 인페인팅은 선택 연구 항목이다. 도입 시 4절 절차를 따르고 결과 출처를 `generated`로 구분하며, 원화 `mask` 영역에는 적용하지 않는다.
- **수동 수정 보존**: 사용자가 수정·승인(`approved`)한 랜드마크나 파츠 마스크(`mask`/`fillMask`)는 자동 재분석 시 동의 없이 덮어쓰지 않는다.
- **가변 다중 머리카락 지원**: 18개 기본 ID(`PART_DEFS` / `IDS`) 외에 추가되는 머리카락 레이어(`hair_front_*`, `hair_side_*`, `hair_back_*` 등)는 고정 18개 배열에 묶어 누락시키지 않고 합성 순서·변형·PSD 내보내기에 포함한다.
- **모션·이음새 우선 검증**: 모션 최대 각도를 줄여 결함을 숨기지 않고, ① 기본 포즈 색상·실루엣 자연스러움 ② 얼굴·목 반대 이동 시 틈 ③ 눈 감김 시 홍채·흰자 누출 ④ 머리카락 이동 시 빈 배경 노출 ⑤ 겹침 경계 이중 선화·흰 테두리를 움직임 상태에서 검증하고, ⑥ 노출 영역 원본감(정답 PSD 비교·자기 지도 홀드아웃 지표)을 수치로 기록한다. 결함 0만으로 "원본처럼 보인다"고 보고하지 않는다.

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

## 6. 파츠 분리의 원칙
- **한 레이어당 한 파츠**: 독립적으로 움직이거나 변형해야 하는 요소는 각각 별도 이미지 레이어로 분리한다. 폴더로 묶더라도 여러 파츠를 하나의 이미지 레이어로 합치지 않는다.
- **움직이는 모습을 기준으로 분리**: 정지 화면의 영역 구분만으로 끝내지 않고 눈 깜빡임·시선 이동·입 개폐·고개 회전·머리카락 흔들림에서 필요한 분리 단위와 겹침 순서를 고려한다. 예를 들어 흰자와 홍채, 위·아래 눈꺼풀은 독립적인 움직임이 필요하면 별도 파츠로 둔다. 움직일 때 드러나는 가림 영역은 기존 `fillMask` 규칙에 따라 별도로 보완한다.
- **레이어 합성 모드는 일반(`normal`)·곱하기(`multiply`)·가산(`additive`) 세 가지로 제한**한다. 현재 PSD 내보내기는 5절에 따라 `normal`만 지원한다. `multiply`·`additive`는 해당 모드의 저장과 실제 Cubism Editor 가져오기를 검증한 뒤 지원하며, 그 전에는 지원한다고 표시하거나 원본 합성 모드를 조용히 `normal`로 바꾸지 않는다.

## 7. 빌드 캐시(`src-tauri/target`) 관리
- `src-tauri/target/`(`debug/build`, `release/build`, `debug/incremental`, `deps` 등)는 **Cargo 빌드 캐시 및 중간 산출물**이다. 프로파일(`dev`/`test`/`release`), 피처, `build.rs` 입력이 바뀔 때마다 새 `<crate>-<hash>` 폴더가 생성되며 이전 캐시가 자동 삭제되지 않아 수십 GB까지 누적될 수 있다.
- **확인 기준**: 디스크 여유 공간 점검이나 대규모 빌드·의존성 변경 후 `src-tauri/target/` 용량이 과도하게 커진 경우(예: 15GB 이상) `debug/incremental`, `deps`, `build` 비율을 확인한다.
- **정리 원칙**:
  - 실행 중인 `tauri dev`나 Cargo 빌드 프로세스가 없을 때만 정리한다.
  - 선택 정리: `src-tauri/`에서 `cargo clean --release` 또는 `cargo clean --profile dev` (전체 정리는 `cargo clean`). `cargo-sweep`이 설치된 경우 `cargo sweep -t 14` 등으로 오래된 해시 캐시만 제거할 수 있다.
  - `target/`과 달리 `models/`(`manifest.json`, `.onnx`) 등 실제 런타임 에셋은 빌드 캐시가 아니므로 절대 삭제하지 않는다.
