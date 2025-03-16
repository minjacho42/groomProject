# 악성 댓글 순화 프로젝트

## 프로젝트 개요

본 프로젝트는 사용자가 입력한 댓글에서 비윤리적인 표현을 자동으로 탐지하고, 가장 부정적인 단어를 제거(마스킹)한 후 GPT-2 모델을 통해 해당 부분을 자연스럽고 윤리적인 표현으로 재생성하는 자연어 처리 시스템입니다.

## 주요 프로세스

### 1. 비윤리적 표현 탐지

- pretrained된 KcELECTRA 모델을 사용하여 댓글의 비윤리적 표현을 탐지합니다.
- 댓글의 부정도를 평가하여 가장 부정적인 영향을 주는 핵심 단어를 탐색합니다.

### 2. 부정적 단어 마스킹

- 탐지된 단어 중 댓글의 부정도를 가장 효과적으로 낮출 수 있는 단어를 선택하여 마스킹합니다.
- 마스킹된 댓글은 GPT-2 모델을 위한 입력 데이터로 사용됩니다.

### 3. 마스킹된 표현 재생성 (GPT-2 기반)

- KoGPT2 모델을 이용하여 마스킹된 위치에 자연스러운 윤리적 표현을 생성하여 댓글을 순화합니다.
- 마스킹 처리 및 표현 생성 과정은 `generaterMethod.py`에 정의되어 있으며, GPT-2 모델의 로딩 및 관리는 `modelLoader.py`를 통해 이루어집니다.

## 기술 스택

- **개발 언어 및 프레임워크**: Python, PyTorch, Streamlit
- **모델**: pretrained KcELECTRA, KoGPT2
- **라이브러리**: Huggingface Transformers, Datasets

## 참고 논문 및 GitHub

**논문**:

- Li, Juncen, et al. "Delete, retrieve, generate: A simple approach to sentiment and style transfer."
- 배장성, et al. "마스크 언어 모델 기반 비병렬 한국어 텍스트 스타일 변환"
- Joosung Lee. "Stable Style Transformer: Delete and Generate Approach with Encoder-Decoder for Text Style Transfer"

**GitHub 레퍼런스**:

- [Stable-Style-Transformer](https://github.com/rungjoo/Stable-Style-Transformer)
- [Transformer-DRG-Style-Transfer](https://github.com/agaralabs/transformer-drg-style-transfer)
- [KcELECTRA](https://github.com/Beomi/KcELECTRA)
- [KoGPT2](https://github.com/SKT-AI/KoGPT2)

