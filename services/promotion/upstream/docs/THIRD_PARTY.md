# 외부 구성요소

- 배경 제거: [rembg](https://github.com/danielgatis/rembg), MIT. 모델은 명시적으로 `u2net`을 지정합니다. 라이브러리 기본 모델을 사용하지 않습니다.
- U2Net: [공식 구현 및 라이선스](https://github.com/xuebinqin/U-2-Net), Apache-2.0. ONNX 가중치는 rembg의 공식 릴리스에서 처음 한 번 내려받으며 저장소에 포함하지 않습니다. 모델 재학습은 없습니다.

상품 사진의 배경 분리는 로컬에서 수행합니다. 상품 사진을 외부 배경 제거 서비스로 보내지 않습니다. 별도로 올린 스타일 참고 이미지는 CLOVA Studio의 HCX-005에 전달됩니다.

- 모든 동봉 글꼴은 SIL Open Font License 1.1로 배포되며 상업용 포스터에 사용할 수 있습니다. 폰트 파일과 이름은 수정하지 않았으며 저작권 문구와 라이선스를 동봉합니다. [SIL 공식 사용 안내](https://openfontlicense.org/how-to-use-ofl-fonts/).

| 글꼴 | 배포 원본 | 동봉 라이선스 |
|---|---|---|
| NAVER 나눔고딕 Regular/Bold · NHN 2010 | [Google Fonts](https://github.com/google/fonts/tree/main/ofl/nanumgothic) | [OFL](../promotion_assistant/fonts/nanumgothic/OFL.txt) |
| NAVER 나눔명조 Regular/ExtraBold · NHN 2010 | [Google Fonts](https://github.com/google/fonts/tree/main/ofl/nanummyeongjo) | [OFL](../promotion_assistant/fonts/OFL.txt) |
| 고운바탕 · Project Authors 2021 | [Google Fonts](https://github.com/google/fonts/tree/main/ofl/gowunbatang) | [OFL](../promotion_assistant/fonts/gowunbatang/OFL.txt) |
| 고운돋움 · Project Authors 2021 | [Google Fonts](https://github.com/google/fonts/tree/main/ofl/gowundodum) | [OFL](../promotion_assistant/fonts/gowundodum/OFL.txt) |
| 배민 주아 · Jua Project Authors 2018 | [Google Fonts](https://github.com/google/fonts/tree/main/ofl/jua) | [OFL](../promotion_assistant/fonts/jua/OFL.txt) |
| 배민 도현 · Do Hyeon Project Authors 2018 | [Google Fonts](https://github.com/google/fonts/tree/main/ofl/dohyeon) | [OFL](../promotion_assistant/fonts/dohyeon/OFL.txt) |
| 검은고딕 · Black Han Sans Project Authors 2015 | [Google Fonts](https://github.com/google/fonts/tree/main/ofl/blackhansans) | [OFL](../promotion_assistant/fonts/blackhansans/OFL.txt) |
| NAVER 나눔손글씨 펜 · NHN 2010 | [Google Fonts](https://github.com/google/fonts/tree/main/ofl/nanumpenscript) | [OFL](../promotion_assistant/fonts/nanumpenscript/OFL.txt) |

- `examples/product_fixtures.py`, `examples/render_gallery.py`의 상품은 배치 검증을 위한 자체 제작 도형 일러스트입니다. 실제 상품 사진이나 생성형 AI 사진이 아닙니다. 참고 광고의 타사 로고·사진·문구는 저장소에 포함하지 않습니다.
- 커피 예시 사진: Rachel Michetti, courtesy of Pikolo Espresso Bar, CC0. [scikit-image 공식 coffee 데이터 설명](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.coffee). `examples/assets/coffee.jpg`는 해당 이미지의 JPEG 사본입니다. 시안 비교 이미지에는 배경 제거·크기 조정·문구 합성을 적용했습니다. 매장명·가격은 가상 정보입니다.
