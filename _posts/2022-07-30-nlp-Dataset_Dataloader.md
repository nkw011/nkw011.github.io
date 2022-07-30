---
title: "[NLP] 자연어처리에 사용되는 Dataset(데이터셋), Dataloader 만들기"
excerpt: "spaCy의 Tokenizer와 torchtext을 활용하여 말뭉치를 단어 사전으로 바꾼다. 단어 사전을 활용해 데이터셋을 구성하고 학습을 위한 데이터로더를 구성하는 방법에 대해 배운다."
use_math: true
toc: true
toc_sticky: true
categories:
    - nlp
tags:
    - spaCy
    - torchtext
    - Vocab
    - Dataset
    - DataLoader
sidebar:
    nav: sidebarTotal
---

Word-Level 단위 Language Modeling을 하기위한 Dataset을 구성하는 방법과 Machine Translation을 위한 Dataset, Dataloader 구성 방법에 대해 알아본다.

Dataset
* Language Modeling: WikiText-2
* Mahcine Translation: Multi-30k

## 0. torchdata 설치하기

torchtext.datasets을 이용해 데이터셋을 불러오려면 torchdata를 설치해야한다.
자세한 내용은 이전 포스트를 확인하면 좋을 것 같다.


```python
!pip install folium==0.2.1
```


```python
!pip install torchdata==0.4.0
```

## 1. Language Modeling을 위한 데이터셋 구성하기

### 1.1. WikiText-2 불러오기

train 세트, val 세트, test 세트를 모두 불러와서 사용한다.


```python
from torchtext.datasets import WikiText2

wiki_train, wiki_val, wiki_test = WikiText2()
```


```python
# 데이터셋 체크

print("[Train]")
for i, text in enumerate(wiki_train):
    if i == 5: break
    print(text)

print("\n[Val]")
for i, text in enumerate(wiki_val):
    if i == 5: break
    print(text)

print("\n[Test]")
for i, text in enumerate(wiki_test):
    if i == 5: break
    print(text)
```

    [Train]
     
    
     = Valkyria Chronicles III = 
    
     
    
     Senjō no Valkyria 3 : <unk> Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit . Valkyria of the Battlefield 3 ) , commonly referred to as Valkyria Chronicles III outside Japan , is a tactical role @-@ playing video game developed by Sega and Media.Vision for the PlayStation Portable . Released in January 2011 in Japan , it is the third game in the Valkyria series . <unk> the same fusion of tactical and real @-@ time gameplay as its predecessors , the story runs parallel to the first game and follows the " Nameless " , a penal military unit serving the nation of Gallia during the Second Europan War who perform secret black operations and are pitted against the Imperial unit " <unk> Raven " . 
    
     The game began development in 2010 , carrying over a large portion of the work done on Valkyria Chronicles II . While it retained the standard features of the series , it also underwent multiple adjustments , such as making the game more <unk> for series newcomers . Character designer <unk> Honjou and composer Hitoshi Sakimoto both returned from previous entries , along with Valkyria Chronicles II director Takeshi Ozawa . A large team of writers handled the script . The game 's opening theme was sung by May 'n . 
    
    
    [Val]
     
    
     = Homarus gammarus = 
    
     
    
     Homarus gammarus , known as the European lobster or common lobster , is a species of <unk> lobster from the eastern Atlantic Ocean , Mediterranean Sea and parts of the Black Sea . It is closely related to the American lobster , H. americanus . It may grow to a length of 60 cm ( 24 in ) and a mass of 6 kilograms ( 13 lb ) , and bears a conspicuous pair of claws . In life , the lobsters are blue , only becoming " lobster red " on cooking . Mating occurs in the summer , producing eggs which are carried by the females for up to a year before hatching into <unk> larvae . Homarus gammarus is a highly esteemed food , and is widely caught using lobster pots , mostly around the British Isles . 
    
     
    
    
    [Test]
     
    
     = Robert <unk> = 
    
     
    
     Robert <unk> is an English film , television and theatre actor . He had a guest @-@ starring role on the television series The Bill in 2000 . This was followed by a starring role in the play Herons written by Simon Stephens , which was performed in 2001 at the Royal Court Theatre . He had a guest role in the television series Judge John <unk> in 2002 . In 2004 <unk> landed a role as " Craig " in the episode " Teddy 's Story " of the television series The Long Firm ; he starred alongside actors Mark Strong and Derek Jacobi . He was cast in the 2005 theatre productions of the Philip Ridley play Mercury Fur , which was performed at the Drum Theatre in Plymouth and the <unk> <unk> Factory in London . He was directed by John <unk> and starred alongside Ben <unk> , Shane <unk> , Harry Kent , Fraser <unk> , Sophie Stanton and Dominic Hall . 
    
     In 2006 , <unk> starred alongside <unk> in the play <unk> written by Mark <unk> . He appeared on a 2006 episode of the television series , Doctors , followed by a role in the 2007 theatre production of How to Curse directed by <unk> <unk> . How to Curse was performed at Bush Theatre in the London Borough of <unk> and Fulham . <unk> starred in two films in 2008 , <unk> <unk> by filmmaker Paris <unk> , and <unk> Punch directed by <unk> Blackburn . In May 2008 , <unk> made a guest appearance on a two @-@ part episode arc of the television series Waking the Dead , followed by an appearance on the television series <unk> in November 2008 . He had a recurring role in ten episodes of the television series <unk> in 2010 , as " <unk> Fletcher " . <unk> starred in the 2011 film <unk> directed by Paris <unk> . 
    



```python
print(f"train set size: {len([text for text in wiki_train])}")
print(f"val set size: {len([text for text in wiki_val])}")
print(f"test set size: {len([text for text in wiki_test])}")
```

    train set size: 36718
    val set size: 3760
    test set size: 4358


### 1.2. Tokenizer를 이용하여 Vocab 구성하기


torchtext의 get_tokenizer()를 이용하여 Tokenizer를 불러온다. 불러온 이후 build_vocab_from_iterator()를 이용하여 Vocab을 구성한다.

test 세트, val 세트는 학습용이 아니기 때문에 Vocab에 포함시키면 안된다. 따라서 train 세트만 사용해 Vocab을 구성하고 val세트, test 세트 내에 처음보는 단어들은 \<unk\> 토큰을 사용해 처리한다.



```python
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
```


```python
tokenizer = get_tokenizer('basic_english')

wiki_vocab = build_vocab_from_iterator(map(tokenizer, wiki_train), min_freq=3, specials=['<unk>'])
```


```python
wiki_id2token = wiki_vocab.get_itos()
wiki_token2id = wiki_vocab.get_stoi()
```


```python
wiki_id2token[0], wiki_token2id['<unk>']
```




    ('<unk>', 0)



### 1.3. Encoding하기

train 세트, val 세트, test 세트를 Vocab을 이용하여 id값들로 바꾸어준다.

이 때, 만약 Vocab에 없는 단어라면 \<unk\> 토큰을 사용한다.


```python
import torch
```


```python
def encode(tokenizer, token2id, data):
    encoded = [
              torch.tensor(list(map(lambda x: token2id.get(x, '<unk>'), tokens))).long()
              for tokens in map(tokenizer, data)
             ]
    return torch.cat(encoded)
```


```python
wiki_train_seq = encode(tokenizer, wiki_token2id, wiki_train)
wiki_val_seq = encode(tokenizer, wiki_token2id, wiki_val)
wiki_test_seq = encode(tokenizer, wiki_token2id, wiki_test)
```


```python
print(f"Train sequence size: {wiki_train_seq.size()}")
print(f"Val sequence size: {wiki_val_seq.size()}")
print(f"Test sequence size: {wiki_test_seq.size()}")
```

    Train sequence size: torch.Size([2049990])
    Val sequence size: torch.Size([214417])
    Test sequence size: torch.Size([241859])


### 1.4. batch 구성하기

현재 데이터는 id로 이루어진 sequence이다. 이 sequence가 모델에 입력으로 들어가게되면 id와 매핑이 되는 임베딩 값으로 바뀌게된다. 그렇게 되면 shape이 1차원에서 2차원으로 바뀌게 된다.
* total_len: 전체 길이, emb_dim: 임베딩 차원
* (total_len,) → (total_len, emb_dim)

하지만 sequence를 모델의 입력으로 그대로 사용하게되면 전체 길이를 학습하기 때문에 시간이 오래 걸린다. 따라서 보통 batch 단위로 잘라서 학습하게된다.
* batch_size: batch 크기
* seq_len: 학습의 크기로 정한 sub-sequence length

먼저 전체 sequence를 batch_size만큼 나눈다.
```
total_sequence: [a, b, c, d, e, f, g, h, i, j, k, l] / shape: (12,)
batch_size: 3

[[a,b,c,d],
 [e,f,g,h],
 [i,j,k,l]]
shape: (3,4)
```
학습의 크기로 정한 seq_len만큼 다시 나눈다.
```
batch_size: 3
seq_len:2

[[[a,b],[c,d]],
 [[e,f],[g,h]],
 [[i,j],[k,l]]]
shape: (3,2,2)
```
모델의 입력으로 batch_size(한 번에 학습가능한 데이터 갯수)만큼 sub-sequence가 들어가기 위해서 shape의 첫번째 차원과 두번째 차원을 transpose한다.
* view가 아닌 transpose를 사용하는 이유는 데이터의 순서를 유지하기 위해서이다.

```
batch_size: 3
seq_len:2

[[[a,b],[e,f],[i,j]],
 [[c,d],[g,h],[k,l]]]
shape: (2,3,2)
```
이렇게 바꾸게 되면 데이터가 다음과 같이 모델의 입력으로 들어가게된다.

```
batch_size: 3
seq_len:2

첫번째: [[a,b],[c,d],[e,f]] / shape: (batch_size, seq_len)

두번째: [[g,h],[i,j],[k,l]] / shape: (batch_size, seq_len)
```
임베딩 과정을 거치게 되면 shape이 다음과 같이 바뀌게된다.
* (batch_size, seq_len) → (batch_size, seq_len, emb_dim)

이렇게 batch_size가 shape의 처음에 오도록 데이터셋을 구성하는 방식을 batch_first 형태라고 부른다.


```python
def batchfy(data, batch_size, seq_len):
    num_sample = data.size()[0] // (batch_size * seq_len)
    data = data[:num_sample*batch_size*seq_len] # 남은 길이는 제외한다.
    data = data.view(batch_size,-1,seq_len).transpose(0,1)
    return data
```


```python
batch_size=128
seq_len=64

wiki_train_batch = batchfy(wiki_train_seq, batch_size, seq_len)
wiki_val_batch = batchfy(wiki_val_seq, batch_size, seq_len)
wiki_test_batch = batchfy(wiki_test_seq, batch_size, seq_len)
```


```python
print(f"Train batch size: {wiki_train_batch.size()}")
print(f"Val batch size: {wiki_val_batch.size()}")
print(f"Test batch size: {wiki_test_batch.size()}")
```

    Train batch size: torch.Size([250, 128, 64])
    Val batch size: torch.Size([26, 128, 64])
    Test batch size: torch.Size([29, 128, 64])



```python
wiki_train_batch[0].size()
```




    torch.Size([128, 64])



## 2. Machine Translation을 위한 데이터셋 구성하기

### 2.1. Multi30k 불러오기

Multi30k는 독일어-영어로 이루어진 데이터셋이다. 
* language_pair를 이용하여 ('de', 'en'), ('en', 'de') 순서를 정할 수 있다.
    * 영어를 독일어로 변형하기 위해 ('en', 'de') 방식으로 불러올 것이다.
* 나머지는 WikiText-2 파라미터와 동일하다.


```python
from torchtext.datasets import Multi30k

multi_train, multi_valid, multi_test = Multi30k(language_pair=('en','de'))
```


```python
for i, (eng, de) in enumerate(multi_train):
    if i == 5: break
    print(f"index:{i}, English: {eng}, das Deutsche: {de}")
```

    index:0, English: Two young, White males are outside near many bushes., das Deutsche: Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.
    index:1, English: Several men in hard hats are operating a giant pulley system., das Deutsche: Mehrere Männer mit Schutzhelmen bedienen ein Antriebsradsystem.
    index:2, English: A little girl climbing into a wooden playhouse., das Deutsche: Ein kleines Mädchen klettert in ein Spielhaus aus Holz.
    index:3, English: A man in a blue shirt is standing on a ladder cleaning a window., das Deutsche: Ein Mann in einem blauen Hemd steht auf einer Leiter und putzt ein Fenster.
    index:4, English: Two men are at the stove preparing food., das Deutsche: Zwei Männer stehen am Herd und bereiten Essen zu.



```python
len([text for text in multi_train]), len([text for text in multi_valid]), len([text for text in multi_test])
```

한 번에 (영어, 독일어) pair를 1개씩 불러올 수 있다.

### 2.2. Tokenizer를 이용하여 Vocab 구성하기

번역하려는 문장을 Source, 번역된 문장을 Target이라고 부른다.

Source, Target의 언어 도메인이 다르기 때문에 Vocab을 따로 구성해야한다.

'spacy'를 이용하여 영어와 독일어를 토큰화한 후 Vocab을 따로 구성해본다.
* get_tokenizer에 tokenizer에 'spacy'를 넘겨주면 spacy tokenizer를 사용할 수 있다.
    * tokenizer를 명시한 경우 language에 사용하려는 언어를 명시해주어야한다.


```python
# 독일어, 영어 토큰화를 위한 데이터 다운로드
!python -m spacy download en_core_web_sm
!python -m spacy download de_core_news_sm
```


```python
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
```


```python
en_tokenizer = get_tokenizer(tokenizer='spacy', language='en_core_web_sm')
de_tokenizer = get_tokenizer(tokenizer='spacy', language='de_core_news_sm')
```

마찬가지로 train 세트만 이용해서 Vocab을 구성하겠다.

special token도 같이 포함한다.
* \<unk\>: unknown token
* \<sos\>: start of sentence
* \<eos\>: end of sentence
* \<pad\>: padding



```python
from functools import partial
```


```python
en_vocab = build_vocab_from_iterator(map(en_tokenizer, [english for english, _ in multi_train]), min_freq=2, specials=["<unk>", "<sos>", "<eos>", "<pad>"])
de_vocab = build_vocab_from_iterator(map(de_tokenizer, [de for _ , de in multi_train]), min_freq=2, specials=["<unk>", "<sos>", "<eos>", "<pad>"])
```

    /usr/local/lib/python3.7/dist-packages/torch/utils/data/datapipes/iter/combining.py:249: UserWarning: Some child DataPipes are not exhausted when __iter__ is called. We are resetting the buffer and each child DataPipe will read from the start again.
      "the buffer and each child DataPipe will read from the start again.", UserWarning)



```python
en_token2id = en_vocab.get_stoi()
de_token2id = de_vocab.get_stoi()

en_id2token = en_vocab.get_itos()
de_id2token = de_vocab.get_itos()

print(len(en_token2id),len(de_token2id)) # vocab 크기
```

    6191 8014



```python
# 스페셜 토큰 체크

en_token2id['<unk>'], en_token2id['<sos>'], en_token2id['<eos>'], en_token2id['<pad>']
```




    (0, 1, 2, 3)




```python
# 스페셜 토큰 체크
de_token2id['<unk>'], de_token2id['<sos>'], de_token2id['<eos>'], de_token2id['<pad>']
```




    (0, 1, 2, 3)



### 2.3. Vocab을 이용하여 전처리 하기


Source는 문장을 바로 id값들로 바꾸면 되지만, Target은 문장의 앞 뒤로 \<sos\>, \<eos\>를 넣어야한다.


```python
class Language:
    unk_token_id = 0
    sos_token_id = 1
    eos_token_id = 2
    pad_token_id = 3

    def __init__(self, src_tokenizer, tgt_tokenizer, src_token2id, tgt_token2id, src_id2token, tgt_id2token):
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

        self.src_token2id = src_token2id
        self.tgt_token2id = tgt_token2id

        self.src_id2token = src_id2token
        self.tgt_id2token = tgt_id2token

    def src_encode(self, src_text):
        source_sentence = [ self.src_token2id.get(token, self.src_token2id['<unk>']) for token in self.src_tokenizer(src_text) ]
        return source_sentence
    
    def tgt_encode(self, tgt_text):
        target_sentence = [self.tgt_token2id['<sos>']] \
        + [ self.tgt_token2id.get(token, self.tgt_token2id['<unk>']) for token in self.tgt_tokenizer(tgt_text) ] \
        + [self.tgt_token2id['<eos>']]
        return target_sentence
    
    def src_decode(self, ids):
        sentence = list(map(lambda x: self.src_id2token[x], ids))
        return " ".join(sentence)

    def tgt_decode(self, ids):
        sentence = list(map(lambda x: self.tgt_id2token[x], ids))[1:-1]
        return " ".join(sentence)
```


```python
pre_process = Language(en_tokenizer, de_tokenizer, en_token2id, de_token2id, en_id2token, de_id2token)
```


```python
en_test, de_test = next(iter(multi_train))
```


```python
en_encoded = pre_process.src_encode(en_test)
de_encoded = pre_process.tgt_encode(de_test)
```


```python
print(f"source original: {en_test}")
print(f"target original: {de_test}")
```

    source original: Two young, White males are outside near many bushes.
    target original: Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.



```python
print(f"source decoded: {pre_process.src_decode(en_encoded)}")
print(f"source decoded: {pre_process.tgt_decode(de_encoded)}")
```

    source decoded: Two young , White males are outside near many bushes .
    source decoded: Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche .


### 2.4. Custom Dataset 구성하기

torch.utils.data.Dataset을 상속받아 train 데이터셋, valid 데이터셋, test 데이터셋을 구성한다.


```python
from torch.utils.data import Dataset
```


```python
class MultiDataset(Dataset):
    def __init__(self, data, language):
        self.data = data
        self.language = language
        self.sentences = self.preprocess()

    def preprocess(self):
        # dataset 안에 길이가 0인 문장이 존재한다. 
        sentences = [ (self.language.src_encode(eng), self.language.tgt_encode(de)) 
                      for eng, de in self.data if len(eng) > 0 and len(de) > 0]

        return sentences

    def __getitem__(self, idx):
        return self.sentences[idx]

    def __len__(self):
        return len(self.sentences)
```


```python
language = Language(en_tokenizer, de_tokenizer, en_token2id, de_token2id, en_id2token, de_id2token)
```


```python
multi_train_dataset = MultiDataset(multi_train, language)
multi_val_dataset = MultiDataset(multi_valid, language)
multi_test_dataset = MultiDataset(multi_test, language)
```


```python
multi_train_dataset[0]
```




    ([19, 25, 15, 1169, 808, 17, 57, 84, 336, 1339, 5],
     [1, 21, 85, 257, 31, 87, 22, 94, 7, 16, 112, 7910, 3209, 4, 2])




```python
multi_val_dataset[0]
```




    ([6, 39, 13, 36, 17, 1667, 2541, 342, 4, 282],
     [1, 14, 38, 24, 243, 2744, 0, 11, 20, 892, 2])




```python
multi_test_dataset[0]
```




    ([6, 12, 7, 28, 91, 68, 2670, 20, 122, 5],
     [1, 5, 12, 10, 6, 178, 108, 8, 16, 78, 0, 4, 2])




```python
len(multi_train_dataset), len(multi_val_dataset), len(multi_test_dataset)
```




    (29000, 1014, 1000)



### 2.5. Dataloader 구성하기




#### collate_fn

Batch 안에 존재하는 sequence의 길이가 모두 다르기 때문에 padding을 이용하여 크기를 맞춰줘야한다.

collate_fn은 Dataloader 내에서 원하는 형태의 Batch로 가공하기 위해 사용되는 함수이다. 

torch.nn.utils.rnn.pad_sequnce를 활용하여 sequence에 padding을 추가하고 collate_fn 함수를 활용하여 Batch내 sequence의 길이를 맞추는 작업을 한다.
* pad_sequence는 padding을 뒤부터 채워넣는다.


```python
import torch
from torch.nn.utils.rnn import pad_sequence
```


```python
def collate_fn(batch_samples):
    pad_token_id = Language.pad_token_id

    src_sentences = pad_sequence([torch.tensor(src) for src, _ in batch_samples], batch_first=True, padding_value=pad_token_id)
    tgt_sentences = pad_sequence([torch.tensor(tgt) for _, tgt in batch_samples], batch_first=True, padding_value=pad_token_id)

    return src_sentences, tgt_sentences
```

#### batch_sampler

padding을 채울 때 비슷한 길이의 sequence들이 하나의 Batch안에 있을수록 더 적은 padding을 사용할 수 있을 것이다.
* sequence bucketing이라고도 불린다.

batch_sampler는 Data를 sampling해 Batch를 구성하는 방법이다.
batch_sampler를 활용해서 비슷한 길이의 sequence가 하나의 Batch안에 존재할 수 있게 Batch를 구성한다.
* batch_sampler는 Batch를 구성하는 방법을 가리키고 collate_fn은 구성된 Batch를 가공하는 역할이라고 보면 된다.

기계 번역은 하나의 sample안에 Source와 Target이 존재하므로 2개의 문장을 모두 고려해야한다. 쉬운 방법으로 Source에 맞춰 sampling을 하는 것이다. Source와 Target의 길이가 대체로 비슷하기 때문에 Source나 Target을 기준으로 sampling을 구성하면된다.
* 여기서는 source에 맞춰 bucketing을 간단하게 구현하겠다.
* [Comprehensive Hands-on Guide to Sequence Model batching strategy: Bucketing technique](https://rashmi-margani.medium.com/how-to-speed-up-the-training-of-the-sequence-model-using-bucketing-techniques-9e302b0fd976)

마지막으로 모델이 길이에 편향되어 학습하지 않도록 shuffle해주는 것도 중요하다.


```python
import random
```


```python
def batch_sampling(sequence_lengths, batch_size):
    '''
    sequence_length: (source 길이, target 길이)가 담긴 리스트이다.
    batch_size: batch 크기
    '''

    seq_lens = [(i, seq_len, tgt_len) for i,(seq_len, tgt_len) in enumerate(sequence_lengths)]
    seq_lens = sorted(seq_lens, key=lambda x: x[1])
    seq_lens = [sample[0] for sample in seq_lens]
    sample_indices = [ seq_lens[i:i+batch_size] for i in range(0,len(seq_lens), batch_size)]

    random.shuffle(sample_indices) # 모델이 길이에 편향되지 않도록 섞는다.

    return sample_indices
```

#### DataLoader

Batch 크기를 5로 설정하고 collate_fn과 batch_sampler를 활용해 Dataloader를 구현하였다.


```python
from torch.utils.data import DataLoader
```


```python
batch_size=5

sequence_lengths = list(map(lambda x: (len(x[0]), len(x[1])), multi_train_dataset))

batch_sampler = batch_sampling(sequence_lengths, batch_size)

train_loader = DataLoader(multi_train_dataset, collate_fn=collate_fn, batch_sampler=batch_sampler)
```


```python
for src, tgt in train_loader:
    print(src)
    print(tgt)
    break
```

    tensor([[ 111,   14, 2250,   15,  150,   15,   68,   15,    7, 1071,  302,  139,
               74,   18,   34,   15,    7,   44,   13,    4,   90,    5],
            [   6,   25,   35,   14,    4,   31,   23,   15,   30,  177,   11, 1761,
              492,   18,  249,   82,   48,    8,    0,    0,  772,    5],
            [   6,   39,   13,   36,   17,   32,   11,   37,    4, 1097,   15,   88,
               13,  159,   17,  217,  987,   11,   88,   17,  121,    5],
            [  19,  117,   15,   54,  294,   11,   54,   26,   15,  127, 1994,   43,
               13,   43, 1701,    9,    8,  101,   84,    8,  860,    5],
            [1167,    7,   51,  394,   23,   11, 1527, 1371,   15,   46,    4,  220,
               14,  515, 2116,    7,   27,   30, 1013,  106, 2408,    5]])
    tensor([[   1,    5,   12,   10, 1347,    8,  142,    9,  108,    7,    0,    8,
               70,   28,    6,  137,   15,  212,   24,   25,   49,   79,   33,    4,
                2,    3],
            [   1,    5,   75,   35,   10,    6,   51,   41,    8,  226,  161,    9,
             1878,    8,  474,   23,   26,    0,    0, 1021,  119,   29,  167,    4,
                2,    3],
            [   1,   14,   38,   31,   32,    9,   30,    7,    6, 2865,    8,  344,
               24,  208,  646, 1286,    8,   37,  105,   25,  190,    4,    2,    3,
                3,    3],
            [   1,   21,  123,    8,   15,  754,    9,   15,  117,    8,   59, 2658,
               11,   26,  454,    7,   16,  112,   24,    6,  852,    4,    2,    3,
                3,    3],
            [   1,    5,   12,    7,    6,  657,  421,   41,    9,    6, 2158,    8,
               39,   15,  224,    9,   98,  151, 1140, 1150,    7,  200,   47,    0,
                4,    2]])

