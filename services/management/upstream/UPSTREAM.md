# 기반 코드 출처 및 프로젝트 변경사항

이 저장소는 Chronos 공식 원본 저장소가 아니라, 공식 구현을 M5 매출 예측에 적용·확장한 프로젝트입니다.

- 기반 저장소: `https://github.com/amazon-science/chronos-forecasting`
- 기반 릴리스: `v1.5.3`
- 기반 커밋: `fcd09fe8b6bf57643bde7847bf42a6719b7ee205`
- 프로젝트 추가 사항: `src/chronos_sales`, M5 설정, 테스트, 실험 노트북, 한국어 가이드
- 공식 구현 수정 사항: `scripts/training/train.py`에 `data_freq`와 `resume_from_checkpoint`를 노출하여 중단된 Hugging Face Trainer 실행이 모델, 옵티마이저, 스케줄러, RNG 및 학습 단계를 복원할 수 있도록 했습니다.
