---
title: "[Huggingface 🤗 Transformers Tutorial] 1. Pipelines for inference"
excerpt: "🤗 Transformers의 pipeline() 메소드를 이용하여 자연어처리 task를 간단하게 수행합니다."
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

※ 이 글의 원문은 [이 곳](https://huggingface.co/docs/transformers/pipeline_tutorial)에서 확인할 수 있습니다.
※ 모든 글의 내용을 포함하지 않으며 새롭게 구성한 내용도 포함되어 있습니다.

[pipeline()](https://huggingface.co/docs/transformers/v4.21.1/en/main_classes/pipelines#pipelines) 함수를 이용해 모델의 구조를 정확히 모르더라도 pretrained 모델을 이용해 자연어처리 task를 수행할 수 있습니다.
* [pipeline()을 이용해 수행할 수 있는 task](https://huggingface.co/docs/transformers/v4.21.1/en/main_classes/pipelines#transformers.pipeline.task)
    * text-classification
    * text-generation
    * token-classification
    * fill-mask

[Model Hub](https://huggingface.co/models)에서 pretrained model들을 확인해볼 수 있습니다. 각 모델별로 수행할 수 있는 task가 모두 다르므로 task에 적합한 모델을 찾아야합니다.

## pipeline()을 이용해 fill-mask task 수행하기

task에 적합한 model을 찾았다면 AutoModel, AutoTokenizer 클래스를 이용하여 model과 model에 사용되는 tokenizer를 간단하게 다운로드할 수 있습니다.
* AutoClass에 관해서는 다음 글에서 다룹니다.
* 이번에는 fill-mask를 수행하기 때문에 AutoModelForMaskedLM 클래스를 이용하여 모델을 불러옵니다. (AutoModel을 이용할 경우 에러가 발생합니다.)


```python
!pip install transformers
```

한국어 fill-mask task를 수행하기위해 BERT pretrained 모델 중에서 [bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased)를 불러옵니다.
* 다양한 언어를 다룰 수 있는 multilingual model입니다.

from_pretrained()에 model 이름을 넣으면 손쉽게 pretrained model, tokenizer를 불러올 수 있습니다.

* 일반적으로 model에 사용되는 configuration, tokenizer가 모두 다르기 때문에 사용하려는 model에 적합한 configuration, tokenizer를 불러와야합니다.


```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

MODEL_NAME = 'bert-base-multilingual-cased'
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
```

먼저 tokenizer가 정상적으로 동작하는지 확인합니다.
* 원문: 이순신은 조선 중기의 무신이다.
* mask: 이순신은 [MASK] 중기의 무신이다.

fill-mask task를 수행하려면 text내에 [MASK] special token이 포함되어 있어야합니다.


```python
text = "이순신은 [MASK] 중기의 무신이다."

tokenizer.tokenize(text)
```




    ['이', '##순', '##신', '##은', '[MASK]', '중', '##기의', '무', '##신', '##이다', '.']



BERT는 WordPiece 방식의 tokenization을 사용하기 때문에 ##이라는 특별한 prefix가 붙어있는 token들을 확인할 수 있습니다.
* ##은 해당 token이 원래는 앞 token과 붙어있다는 것을 의미합니다. e.g.) 이순신 → 이, ##순, ##신

pipeline()을 이용해 한국어 fill-mask task를 수행하기위한 함수를 만듭니다.


```python
from transformers import pipeline

kor_mask_fill = pipeline(task='fill-mask', model=model, tokenizer=tokenizer)
```

kor_mask_fill 함수를 이용하여 fill-mask task를 수행합니다.


```python
text = "이순신은 [MASK] 중기의 무신이다."

kor_mask_fill("이순신은 [MASK] 중기의 무신이다.")
```




    [{'score': 0.874712347984314,
      'token': 59906,
      'token_str': '조선',
      'sequence': '이순신은 조선 중기의 무신이다.'},
     {'score': 0.0643644854426384,
      'token': 9751,
      'token_str': '청',
      'sequence': '이순신은 청 중기의 무신이다.'},
     {'score': 0.010954903438687325,
      'token': 9665,
      'token_str': '전',
      'sequence': '이순신은 전 중기의 무신이다.'},
     {'score': 0.004647187888622284,
      'token': 22200,
      'token_str': '##종',
      'sequence': '이순신은종 중기의 무신이다.'},
     {'score': 0.0036106701008975506,
      'token': 12310,
      'token_str': '##기',
      'sequence': '이순신은기 중기의 무신이다.'}]



[MASK] 자리에 들어갈 token들을 리스트 형태로 반환합니다.
* score: 점수
* token: token id
* token_str: token text
