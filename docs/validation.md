# 로컬 검증 기록

2026-11-15 · Windows · Node 20.16 · Rust 1.99 · WebView2

- `npm test`: 10개 통과. 마스크 분할·편집, 메쉬 구멍 보존, 프로젝트 왕복, 손상 파일 거절, 실제 PSD 바이트/알파/오프셋, ZIP 산출물, CSV 형식 검증.
- `cargo test --manifest-path src-tauri/Cargo.toml --lib`: 6개 통과.
- `cargo clippy --manifest-path src-tauri/Cargo.toml --all-targets -- -D warnings`: 통과.
- `npm run desktop:build -- --debug`: 실행 파일 생성.
- `npm run test:native`: 빌드된 실행 파일의 실제 WebView에서 이미지 입력, 드롭 영역 표시, PNG 드롭, 잘못된 파일 거절, Rust 파츠 생성, ONNX 배경 추론, 메쉬/변형 샘플, 가림 보완, 투명 효과 축소를 검증. JavaScript 예외 없음.
- `sutteang/2d_.png` 1,024 × 1,024px 이미지: Rust CLI + U2NetP 실제 추론으로 18개 파츠 PNG와 메쉬 JSON 생성 확인. 기하 템플릿의 분리 품질을 보증하는 평가가 아님.
- 모델 SHA-256: `309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8`.

임시 결과와 스크린샷은 Git에서 제외한 `artifacts/`에 저장됩니다. 네이티브 UI 검사는 해당 앱의 테스트용 WebView 디버깅 포트를 열고 검사 후 프로세스를 종료합니다.

Cubism Editor에서 PSD/CSV를 실제 가져오는 검증, 미설치 PC 배포, 다양한 원화의 자동 분리 품질과 보정 시간, 성능 기준 측정은 수행하지 않았습니다. 자체 메쉬/JSON을 Cubism에 직접 적용하는 기능은 없습니다.
