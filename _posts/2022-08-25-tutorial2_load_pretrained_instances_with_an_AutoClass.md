---
title: "[Huggingface 🤗 Transformers Tutorial] 2. Load pretrained instances with an AutoClass"
excerpt: "🤗 Transformers의 AutoClass의 종류와 활용방법에 대해 배워봅니다."
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

※ 이 글의 원문은 [이 곳](https://huggingface.co/docs/transformers/autoclass_tutorial)에서 확인할 수 있습니다.
※ 모든 글의 내용을 포함하지 않으며 새롭게 구성한 내용도 포함되어 있습니다.

🤗 Transformers는 [Model Hub](https://huggingface.co/models)에 등록된 수많은 pretrained model을 간편하게 사용할 수 있도록 [AutoClass](https://huggingface.co/docs/transformers/v4.21.2/en/model_doc/auto#auto-classes)를 제공합니다.
* from_pretrained 메소드를 이용해서 손쉽게 model과 tokenizer을 load할 수 있게 해줍니다.

이번 시간에는 AutoClass를 이용해서 tokenizer, model을 load하는 방법을 배워보겠습니다.


```python
!pip install transformers
```

## AutoTokenizer

AutoTokenizer를 이용해서 학습에 필요한 tokenizer를 손쉽게 load할 수 있습니다.

AutoTokenizer.from_pretrained()를 이용해서 [bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased)의 tokenizer를 불러옵니다.


```python
MODEL_NAME = 'bert-base-multilingual-cased'
```


```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
```

bert-base-multilingual-cased tokenizer를 사용하기위해 필요한 config, vocab 등을 다운로드 받습니다.


```python
tokenizer
```




    PreTrainedTokenizerFast(name_or_path='bert-base-multilingual-cased', vocab_size=119547, model_max_len=512, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'})



tokenizer.vocab을 이용하여 vocab을 확인할 수 있습니다.


```python
tokenizer.vocab

# 결과는 생략합니다..
```



tokenizer를 이용해서 한국어 문장을 tokenization을 해보겠습니다.

1. tokenizer를 call하는 방식
2. tokenize()
3. encode()


첫번째로, tokenizer를 call하는 방식을 이용해 tokenization을 진행해보겠습니다.

tokenizer를 call하는 방식을 이용하면 주어진 text를 tokenization한 뒤 model 입력에 필요한 모든 요소를 반환해줍니다.


```python
text = '이순신은 조선 중기의 무신이다.'
```


```python
# 1. tokenizer를 call하는 방식

tokenized_text = tokenizer(text)

print(tokenized_text)
```

    {'input_ids': [101, 9638, 119064, 25387, 10892, 59906, 9694, 46874, 9294, 25387, 11925, 119, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}


input_ids, token_type_ids, attention_mask가 반환된 것을 볼 수 있습니다.

다운로드 받은 tokenizer가 BERT 모델을 위한 tokenizer이기 때문에 🤗 Transformers에 정의된 BERT architecture가 요구하는 입력 형태인 input_ids, token_type_ids, attention_mask로 맞추어 반환한 것을 확인할 수 있습니다.

간단하게 각각의 역할에 대해 설명하면,
* input_ids: tokenized된 text에 앞 뒤로 special token을 추가한뒤 id값으로 변형한 값을 뜻합니다.
* token_type_ids: BERT의 segment embedding을 뜻합니다. 현재 문장이 1개가 들어왔기 때문에 token_type_ids가 모두 0인 것을 확인할 수 있습니다.
* attention_mask: pad token은 0으로 나머지 token들은 1을 가집니다. attention 계산을 위해 불필요한 값들을 masking 처리한다고 보면 됩니다.

반환된 값을 그대로 model의 입력에 넣어주면 됩니다.

두번째로, tokenize()를 이용하여 tokenization하는 방법입니다.

tokenize() 메소드는 주어진 text를 단순히 tokenization만 해줍니다.


```python
print(tokenizer.tokenize(text))
```

    ['이', '##순', '##신', '##은', '조선', '중', '##기의', '무', '##신', '##이다', '.']


이순신은 조선 중기의 무신이다. → ['이', '##순', '##신', '##은', '조선', '중', '##기의', '무', '##신', '##이다', '.']로 tokenization된 것을 확인할 수 있습니다.

세번째로, encode()를 이용하는 방법입니다.

encode()는 tokenized된 text에 앞 뒤로 special token을 추가한뒤 id값으로 바꿉니다.
input_ids와 동일합니다.


```python
print(tokenizer.encode(text))
```

    [101, 9638, 119064, 25387, 10892, 59906, 9694, 46874, 9294, 25387, 11925, 119, 102]


input_ids와, encode()의 반환값을 모두 decode()를 이용하여 원래 문장으로 다시 변환할 수 있습니다.

앞 뒤에 [CLS], [SEP] token이 붙은 것을 확인할 수 있습니다.


```python
tokenized_text1 = tokenizer(text)
print(tokenized_text1['input_ids'])
print(tokenizer.decode(tokenized_text1['input_ids']))

tokenizer_text2 = tokenizer.encode(text)
print(tokenizer_text2)
print(tokenizer.decode(tokenizer_text2))
```

    [101, 9638, 119064, 25387, 10892, 59906, 9694, 46874, 9294, 25387, 11925, 119, 102]
    [CLS] 이순신은 조선 중기의 무신이다. [SEP]
    [101, 9638, 119064, 25387, 10892, 59906, 9694, 46874, 9294, 25387, 11925, 119, 102]
    [CLS] 이순신은 조선 중기의 무신이다. [SEP]


## AutoModel

AutoModel을 이용하여 base model을 load할 수 있습니다.
* base model이라 함은 task를 위한 classifier등이 부착되지 않은 vanilla 형태를 가리킵니다.
* AutoModel을 이용해 task를 수행하기 위해서는 별도의 classifier등을 부착해야합니다.

🤗 Transformers에서는 다양한 task들에 적합한 model archtecture를 이미 AutoClass 형태로 제공하고 있습니다. 예를 들어 AutoModelForMaskedLM은 masked langunage modeling을 위한 AutoClass입니다.
* AutoClass 목록은 [여기](https://huggingface.co/docs/transformers/model_doc/auto#auto-classes)서 확인할 수 있습니다.
* AutoModelForQuestionAnswering
* AutoModelForSequenceClassification
* AutoModelForTokenClassification


'bert-base-multilingual-cased'의 base model과 Token Classification model을 load하여 형태가 어떻게 다른지 확인해보겠습니다.



```python
MODEL_NAME = 'bert-base-multilingual-cased'
```


```python
from transformers import AutoModel, AutoModelForTokenClassification

base_model = AutoModel.from_pretrained(MODEL_NAME)
token_class_model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
```


```python
base_model
```




    BertModel() # 내용 일부를 생략했습니다.




```python
token_class_model
```




    BertForTokenClassification(
      (bert): BertModel() # 내용 일부를 생략했습니다.
      (dropout): Dropout(p=0.1, inplace=False)
      (classifier): Linear(in_features=768, out_features=2, bias=True)
    ) 



Token Classification model에 classifier가 추가된 것을 확인할 수 있습니다.
