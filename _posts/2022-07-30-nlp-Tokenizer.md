---
title: "[NLP] NLTK, spaCy, torchtext를 이용하여 영어 토큰화(English Tokenization)작업 수행하기"
excerpt: "토큰화를 위해 사용되는 자연어처리 라이브러리인 NLTK, spaCy, torchtext에 대해 알아본다."
use_math: true
toc: true
toc_sticky: true
categories:
    - nlp
tags:
    - spaCy
    - torchtext
    - NLTK
sidebar:
    nav: sidebarTotal
---

NLP Task를 수행하기 위해 필요한 데이터 전처리 과정 중 Tokenization(토큰화)를 직접 실습해본다.

Python 기본 라이브러리를 이용해 토큰화를 진행할 수도 있지만 시간이 오래걸리고 고려해야할 점이 많다.
* 구두점, 특수문자를 단순하게 제외해서는 안되는 경우: Ph.D.(학위)
* 줄임말
* 한 단어인데 띄어쓰기가 안에 있는 경우: 상표 등
* 공백 단위의 토큰화를 적용할 수 없는 경우: 's (소유격), don't, doesn't (do + not 형태) 등


공개되어있는 자연어 처리 Library를 사용하여 빠르게 토큰화를 하는 방법을 알아본다.
(sub-word 단위의 토큰화는 여기서 수행하지 않는다.)

## 1. NLTK

NLTK는 Natural Language Toolkik의 약자로 교육용으로 개발된 자연어 처리 및 문서 분석용 Python Package이다.

주요 기능
* 토큰화(Tokenization)
* 말뭉치(Corpus)
* 형태소 분석, 품사 태깅(PoS) 등

NLTK는 pip를 이용해 설치할 수 있다.


```python
!pip install nltk
```

완료가 되면 다음과 같이 nltk가 설치되어 있는 것을 확인할 수 있다.


```python
!pip show nltk
```

NLTK의 Tokenizer(토크나이저)를 사용하기 위해서는 데이터(NLTK Data)를 설치해야한다.

nltk를 import하고 nltk.download()를 이용해서 토큰화에 필요한 데이터를 설치할 수 있다.
* nltk.download(): GUI로 이루어진 NLTK 다운로더가 나타난다. 필요한 데이터를 클릭해 설치할 수 있다.
* nltk.download('data_name'): download()의 인자로 필요한 데이터의 이름을 넘겨주면 해당 데이터만 다운로드 받을 수 있다.
    * nltk.download('popular')


```python
import nltk

nltk.download('popular')
```

### 1.1. word_tokenize()

word_tokenize()를 이용해 문장을 토큰화할 수 있다.


```python
from nltk.tokenize import word_tokenize
```


```python
text = 'David\'s book wasn\'t famous, but his family loved his book.'
```


```python
word_tokenize(text)
```




    ['David',
     "'s",
     'book',
     'was',
     "n't",
     'famous',
     ',',
     'but',
     'his',
     'family',
     'loved',
     'his',
     'book',
     '.']



### 1.2. WordPunctTokenizer

WordPunctTokenizer는 work_tokenize와는 달리 '(구두점)을 별도의 토큰으로 구분해서 토큰화를 진행한다.


```python
from nltk.tokenize import WordPunctTokenizer
```


```python
punct_tokenizer = WordPunctTokenizer()

punct_tokenizer.tokenize(text)
```




    ['David',
     "'",
     's',
     'book',
     'wasn',
     "'",
     't',
     'famous',
     ',',
     'but',
     'his',
     'family',
     'loved',
     'his',
     'book',
     '.']



wasn't이 wasn, ', t 로  분리된 것을 알 수 있다.

### 1.3. sent_tokenize()

여러 문장으로 이루어진 text를 1개의 문장씩 토큰화하는 함수이다.


```python
from nltk.tokenize import sent_tokenize
```


```python
text2 = 'David\'s book wasn\'t famous, but his family loved his book. Seventy years later, his book began to be known to the public.'
```


```python
sent_tokenize(text2)
```




    ["David's book wasn't famous, but his family loved his book.",
     'Seventy years later, his book began to be known to the public.']



### 1.4. 불용어(stopword) 처리하기

자주 등장하지만 실제 의미 분석을 하거나 작업을 수행하는데 크게 기여하지 않는 단어들을 불용어(stopword)라고 한다.

불용어는 문장의 길이를 늘리기 때문에 실제 학습할 때 불용어를 제거하는 작업을 수행할 수도 있다.


```python
from nltk.corpus import stopwords
```


```python
stop_word_list = stopwords.words('english')
```


```python
for word in stop_word_list[:10]:
    print(word)
```

    i
    me
    my
    myself
    we
    our
    ours
    ourselves
    you
    you're



```python
token_list = [ token for token in word_tokenize(text) if token not in stop_word_list]
print(token_list)
print(word_tokenize(text))
```

    ['David', "'s", 'book', "n't", 'famous', ',', 'family', 'loved', 'book', '.']
    ['David', "'s", 'book', 'was', "n't", 'famous', ',', 'but', 'his', 'family', 'loved', 'his', 'book', '.']


참고
* [딥 러닝을 이용한 자연어 처리 입문](https://wikidocs.net/book/2155)
* [NLTK 자연어 처리 패키지](https://datascienceschool.net/03%20machine%20learning/03.01.01%20NLTK%20%EC%9E%90%EC%97%B0%EC%96%B4%20%EC%B2%98%EB%A6%AC%20%ED%8C%A8%ED%82%A4%EC%A7%80.html)

## 2. spaCy 

spaCy는 Python과 Cython으로 작성된 고급 자연어 처리를 위한 Python Package이다.

주요 기능
* POS Tagging
* Morphology
* Lemmaztization
* Tokenization
* Named Entities 등

[spaCy - Tokenization](https://spacy.io/usage/linguistic-features#tokenization), [spaCy - Processing Pipelines](https://spacy.io/usage/processing-pipelines)에서 자세한 내용을 확인할 수 있다.


spaCy는 pip를 이용하여 설치할 수 있다.


```python
!pip install spacy
```

설치가 완료되면 다음과 같이 확인할 수 있다.


```python
!pip show spacy
```

spaCy 역시 토큰화에 필요한 데이터를 다운로드 해야한다.


```python
!python -m spacy download en_core_web_sm
```

### 2.1. spaCy를 이용해 토큰화 수행하기


```python
import spacy
```


```python
spacy_en = spacy.load('en_core_web_sm')
```


```python
text = 'David\'s book wasn\'t famous, but his family loved his book.'
```


```python
for token in spacy_en.tokenizer(text):
    print(token)
```

    David
    's
    book
    was
    n't
    famous
    ,
    but
    his
    family
    loved
    his
    book
    .


spaCy를 이용해 토큰화를 수행하면 기본적으로 토큰외에도 PoS(품사), lemma등의 정보를 알 수 있다.


```python
for token in spacy_en.tokenizer(text):
    print(f"token: {token.text}, PoS: {token.pos_}, lemman: {token.lemma_}")
```

    token: David, PoS: , lemman: 
    token: 's, PoS: , lemman: 
    token: book, PoS: , lemman: 
    token: was, PoS: , lemman: 
    token: n't, PoS: , lemman: 
    token: famous, PoS: , lemman: 
    token: ,, PoS: , lemman: 
    token: but, PoS: , lemman: 
    token: his, PoS: , lemman: 
    token: family, PoS: , lemman: 
    token: loved, PoS: , lemman: 
    token: his, PoS: , lemman: 
    token: book, PoS: , lemman: 
    token: ., PoS: , lemman: 


### 2.2. 불용어(stopword)


```python
stop_words = spacy.lang.en.stop_words.STOP_WORDS
```


```python
for i, stop_word in enumerate(stop_words):
    if i == 10: break
    print(stop_word)
```

    very
    become
    twelve
    hereupon
    into
    say
    ‘ll
    each
    throughout
    ’s


## 3. torchtext

자연어처리를 위해 만들어진 PyTorch 라이브러리이다.
* [Docs](https://pytorch.org/text/stable/index.html)

pip를 이용해 설치할 수 있다. PyTorch version 주의하여 적합한 version을 설치해야한다. 
* [Installation Guide](https://github.com/pytorch/text#installation)


```python
!pip install torchtext
```


```python
!pip show torchtext
```

### 3.1. get_tokenizer()를 이용해 토큰화 수행하기

[get_tokenizer()](https://pytorch.org/text/stable/data_utils.html?highlight=get_tok#torchtext.data.utils.get_tokenizer)를 이용하여 torchtext에서 사용되는 tokenizer를 불러올 수 있다.


```python
from torchtext.data import get_tokenizer

tokenizer = get_tokenizer('basic_english')
```


```python
text = 'David\'s book wasn\'t famous, but his family loved his book.'
```


```python
tokens = tokenizer(text)
```


```python
tokens
```




    ['david',
     "'",
     's',
     'book',
     'wasn',
     "'",
     't',
     'famous',
     ',',
     'but',
     'his',
     'family',
     'loved',
     'his',
     'book',
     '.']


