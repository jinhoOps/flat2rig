# flat2rig MVP 사양

> 이전 Cubism 인계 중심 MVP의 기록이다. 2026-11-15 제품 전환 이후 목표와 신규 범위는 [제품 전환 계획](product-plan.md), 구현 지시는 [작업 지시서](development-tasks.md)를 따른다. 아래 구현 설명은 작성 당시 상태이며 현재 기능 목록으로 간주하지 않는다.

## 목적

정면 캐릭터 흉상 일러스트를 Cubism Editor 후반 작업용 자료로 준비한다. 사용자의 마스크·랜드마크 수정과 외부 페인팅 결과 반입을 포함하는 반자동 편집기다.

## 구현 구성

| 영역     | 구현                                                                      |
| -------- | ------------------------------------------------------------------------- |
| 데스크톱 | Tauri v2 / Rust, Windows 우선                                             |
| 추론     | `ort` 2 rc10 / ONNX Runtime 1.22 CPU, 명시적으로 선택한 로컬 U2Net/U2NetP |
| 메쉬     | Rust `delaunator`, 외곽/내부 샘플 및 삼각형 내부 마스크 검증              |
| 변형     | Rust `glam` DVec2/DMat2, 파라미터별 정점 위치 샘플                        |
| 보완     | Rust 인접색 확장, 사용자 확인 및 PNG 교체                                 |
| UI       | React / TypeScript / Vite, Canvas 2D 텍스처 변형                          |
| 내보내기 | `ag-psd` PSD writer + 재읽기, `fflate` ZIP                                |

네이티브 `generate_parts`, `build_rig`, `fill_part`는 `spawn_blocking`으로 실행한다. ONNX 추론이나 메쉬 연산 중 UI 스레드를 차단하지 않는다. 취소 시 결과를 적용하지 않는다. 실행 중인 네이티브 추론 자체를 강제 종료하지 않으므로 반환까지 기다릴 수 있다.

브라우저 개발 모드는 네이티브 기능 없이 화면을 확인하기 위한 대체 경로를 제공한다. 배경 모델 추론은 데스크톱에서만 실행된다.

## 입력과 보정

- PNG/JPEG, 한 명의 정면 흉상, 긴 변 작업 해상도 최대 2,048px.
- 원본/작업 이미지 별도 보관. 외부 파츠 PNG는 작업 캔버스와 같은 크기를 사용.
- 얼굴 영역 지정 및 눈 L/R·입·턱·얼굴 피부 상단의 5개 랜드마크 이동.
- 배경 ONNX 추론 또는 테두리 연결 유사색 제거. 복잡한 배경은 ONNX/수동 보정 사용.
- 파츠 마스크 추가/제거, 보완, PNG 교체, 이름·순서·가시성, 미해당 표시, 확인 상태.
- 실행 취소/다시 실행 최대 24단계, 프로젝트 저장·재열기.

기본 파츠는 뒤머리, 몸통, 목, 얼굴, 좌우 흰자/홍채/위·아래 눈꺼풀, 좌우 눈썹, 입 안/위·아래 입술, 앞머리의 18개다. 사용자 정의 파츠도 추가할 수 있다. 좌우는 캐릭터 기준이다.

가림 보완은 12px 인접색 확장이다. 보이지 않는 형태를 추론해 그리는 기능은 없으며 정교한 복원은 외부 도구에서 수정한 PNG를 반입한다. 이미지를 실제 의미에 따라 분리하는 AI 모델과 자동 랜드마크 모델은 포함하지 않는다.

## 미리보기와 리깅 초안

| ID                    | 정의 범위 | 미리보기 범위 / 기본값 |
| --------------------- | --------- | ---------------------- |
| ParamAngleX / Y       | -30~30    | -10~10 / 0             |
| ParamAngleZ           | -30~30    | -15~15 / 0             |
| ParamEyeLOpen / ROpen | 0~1       | 0~1 / 1                |
| ParamMouthOpenY       | 0~1       | 0~1 / 0                |
| ParamHairFront        | -1~1      | -1~1 / 0               |

메쉬 와이어프레임, 슬라이더, 자동 테스트 모션, 스프링·댐퍼 기반 머리카락 테스트를 제공한다. 단일 이미지의 실제 3D 회전을 재구성하지 않는다.

원화 알파 경계가 미리보기에서 잘리지 않도록 텍스처 전체를 덮는 격자로 렌더링한다. 와이어프레임과 JSON의 메쉬는 마스크 내부 검증을 통과한 삼각형을 사용한다. 동일 변형 함수를 적용하지만 삼각분할이 다르므로 미리보기와 내보낸 정점 보간은 픽셀 단위로 동일하다고 보장하지 않는다.

리깅 JSON에는 기본 정점, UV, 삼각형, 파라미터 샘플 정점 위치, 피벗용 랜드마크와 디포머 구성 제안이 들어간다. 피부/눈/입의 보완이 부족하면 모션에서 틈이 보이며 이를 점검하는 용도다.

## 저장·내보내기 계약

`.flat2rig`는 ZIP 파일이다. `project.json`에 버전·캔버스·원본/작업 이미지·랜드마크·파츠 메타데이터를 저장하고, 마스크는 파츠 ID별 바이트 배열로 저장한다. 승인 상태와 보완 텍스처도 유지한다. 압축 해제 용량/파츠 수/마스크 크기/랜드마크 좌표를 검증한다.

좌표는 작업 캔버스 좌상단 (0,0), 픽셀 단위, X 오른쪽, Y 아래다. ID는 이름 변경과 재정렬 후에도 유지된다. 재생성은 기본 파츠 ID로 돌아가며 기존 보완을 교체한다. 모델을 사용했다면 SHA-256을 프로젝트와 인계 자료에 기록한다.

인계 ZIP:

```text
character_import.psd
project.flat2rig
parts/<id>.png
masks/<id>.png
rig-draft.json
parameters-reference.csv   # CSV 샘플이 없을 때
parameters.csv             # 샘플이 연결되었을 때
guides/landmarks.png
handoff.md
validation.json
```

- PSD는 RGB 8bit Normal 레이어와 알파를 사용한다. 투명 가장자리를 잘라도 원래 위치 오프셋을 보존한다.
- 표시 숨김/솔로는 편집 보조다. 내보내기에는 미해당 상태를 제외한 모든 파츠를 표시한다. 원본 가이드 이미지는 PSD에 끼워 넣지 않는다.
- 파라미터 참고 CSV는 공식 가져오기 형식으로 표시하지 않는다.
- Editor에서 내보낸 CSV 샘플을 연결하면 열 이름/순서와 type 값을 따르고 새 GUID는 비운다. 영어 헤더를 우선 지원하며 CSV 샘플은 세션 동안만 연결된다.
- 저장 대화상자에서 선택한 경로에 임시 파일을 기록한 뒤 교체한다. 저장 취소는 변경을 수행하지 않는다.
- `validation.json`은 PSD 구조 재읽기와 Editor 실검증을 구분한다. 실제 Cubism 가져오기는 아직 검증하지 않았다.

## Cubism 후반 작업

PSD를 가져와 파츠를 배치하고, 메쉬/디포머를 정리한 뒤 눈·입·고개 등의 변형 키를 설정한다. 자체 JSON은 Cubism 직접 가져오기 포맷이 아니다. 외부 연동 API를 메쉬 작성 API로 가정하지 않으며 CMO3/MOC3를 직접 생성하지 않는다. 물리 설정 파일과 모델 템플릿 전이는 현재 구현 범위에서 제외한다.

확인 근거: [PSD 가져오기](https://docs.live2d.com/en/cubism-editor-manual/psd-import/), [PSD 주의사항](https://docs.live2d.com/en/cubism-editor-manual/precautions-for-psd-data/), [일러스트 가공](https://docs.live2d.com/en/cubism-editor-tutorials/psd/), [CSV 가져오기](https://docs.live2d.com/en/cubism-editor-manual/csv-export-import/), [외부 앱 연동](https://docs.live2d.com/en/cubism-editor-manual/external-application-integration/), [물리 설정](https://docs.live2d.com/en/cubism-editor-manual/physical-operation-setting/).

## 남은 품질 검증

실제 Cubism Editor에서 PSD/샘플 기반 CSV 가져오기와 저장·재열기, 다양한 원화의 보정 비용, 깨끗한 Windows PC 배포, 목표 PC에서의 FPS/메모리는 별도 검증이 필요하다. 빌드 성공과 단위/패키지 테스트 통과를 이러한 품질 보증으로 해석하지 않는다.
