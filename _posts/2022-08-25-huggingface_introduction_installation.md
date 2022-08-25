---
title: "Huggingface 🤗 Transformers 소개와 설치"
excerpt: "🤗 Transformers 소개와 설치"
use_math: true
toc: true
toc_sticky: true
categories:
    - nlp
tags:
    - huggingface
    - transformers
sidebar:
    nav: sidebarTotal
---

Hugging Face의 Transformers 라이브러리를 활용하여 SOTA 모델들을 학습해보고 자연어처리 Task를 수행하는 시간을 앞으로 가져볼 것입니다.

* [Transformers](https://huggingface.co/docs/transformers/index) 홈페이지에 있는 Tutorials와 How-To-Guides 등을 순차적으로 수행합니다.


이번 시간에는 Hugging Face의 Transformers 라이브러리를 소개하고 직접 설치해보는 것까지 수행합니다.

## 🤗 Transformers

Transformers 라이브러리는 이름처럼 Transformer 계열의 모델들을 쉽게 사용할 수 있도록 다양한 기능을 제공하는 라이브러리입니다.
[수록된 모델](https://huggingface.co/docs/transformers/index#supported-models)은 132여가지 정도 되며 관련된 코드를 제공해줄 뿐만아니라 pretrained된 모델을 쉽게 다운로드하여 fine-tuning을 통해 여러가지 task를 손쉽게 수행할 수 있도록 도와줍니다.

대표적으로 수행할 수 있는 [Task](https://huggingface.co/docs/transformers/index#transformers)는 다음과 같습니다.
* Text:  text classification, information extraction, question answering, summarization, translation, and text generation in over 100 languages.

Hugging Face는 Transformers 뿐만 아니라 Tokenizers, Datasets와 같은 라이브러리를 추가로 제공하여 task를 수행하기 위한 tokenizer, dataset을 손쉽게 다운로드 받아 사용할 수 있도록 하고 있습니다.

## 🤗 Transformers 설치하기

pip를 이용해 설치할 수 있습니다.
* [Installation Guide](https://huggingface.co/docs/transformers/installation)
* transformers를 설치하면 tokenizer도 같이 설치됩니다.
* datasets는 별도로 다운로드 받아야합니다.



```python
!pip install transformers
!pip install datasets
```
