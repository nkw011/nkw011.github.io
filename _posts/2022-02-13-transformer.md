---
title: "Transformer"
excerpt: "Transformer, Scaled Dot-Product Attention, Multi-head Attention"
use_math: true
toc: true
toc_sticky: true
categories:
    - deeplearning
tags:
    - python
    - pytorch
    - deeplearning
    - ai
sidebar:
    nav: sidebarTotal
---

> [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

encoder-decoder에서 RNN을 Attention으로 모두 대체한 최초의 모델인 Transformer에 대해 살펴본다.

## 1. RNN을 사용한 모델의 단점

RNN, LSTM과 같이 recurrent한 특성을 가진 모델은 들어오는 입력 순서에 따라 $h_t$(, $c_t$)를 계산하기 때문에 학습시 병렬 처리를 하지 못한다는 한계가 존재한다. 입력 데이터가 길수록 그 한계가 명확히 늘어난다. 병렬 처리를 하지 못하기 때문에 학습 시간이 오래걸린다.

Transformer는 기존 encoder-decoder 형태에서 recurrence한 부분을 모두 attention 매커니즘으로 대체한 최초의 모델이다.
RNN, LSTM을 모두 attention 매커니즘을 이용해 대체함으로써 input과 output사이의 global dependency를 한꺼번에 계산할 수 있게 되었다.

Transformer는 발표 당시 translation에 있어 SOTA(a new state of the art)를 달성하였으며 NLP분야 뿐만 아니라 Computer Vision에서도 높은 성능을 기록하고 있는 방식이다.

## 2. Attention

'attention'은 '주목'이라는 뜻으로 attention 매커니즘의 주요 아이디어는 단어의 연관성에 주목을 하는 것이다.

예를 들어 '나는 고양이다'라는 문장을 'I am a cat'으로 번역하고자 한다. '나'라는 단어는 'I am a cat'이라는 문장에서 'I'와 연관이 가장 높을 것이고 '고양이'라는 단어는 'cat'이라는 단어와 연관이 가장 높을 것이다.

attention은 decoder에서 출력 단어를 예측하는 매 시점마다 encoder에서 처리된 입력 단어 정보를 확인해서 decoder의 해당 시점과 가장 연관성이 높은 단어에 주목한다. 이렇게 파악한 연관성을 바탕으로 출력 단어를 예측하게 된다.

두 단어(embedding 벡터)간의 연관성을 비교하는 방법 중 하나는 dot product 연산을 수행하는 것이다. dot product는 두 벡터가 얼마나 동일한가를 나타내는데 그 값이 양수일수록 같은 방향을 가리킨다는 뜻이 되고 음수일수록 서로 반대방향이 되어 전혀 다른 의미를 지니게 된다.

### self-attention

서로 다른 단어가 아닌 한 sequence내에서 서로 다른 위치의 단어들끼리 연관성을 파악하는 방법이다.
transformer는 일부분 self-attention을 반복해서 쌓은 형태를 활용한다.

## 3. SDPA(Scaled Dot-Product Attention)

<center><img src='/assets/image/transformer/transformer1.png'></center>
<center><a src='https://arxiv.org/abs/1706.03762'>Attention Is All You Need</a></center>

### 과정

transformer에서는 attention 매커니즘으로 dot product를 활용한 attention에 scaling을 적용한 형태를 사용한다.

첫번째 단어 벡터에 대한 attention을 계산하는 과정을 이용해 SDPA가 어떤 방식으로 일어나는 지 확인해보자.

#### (1) query, key, value 벡터 생성

input을 embedding layer에 통과시키고 나온 embedding vector 각각에 대해 query, key, value 벡터를 생성한다.

-   embedding vector: $d_{model}$(=512) 차원의 벡터
-   query vector: $d_k$(=64) 차원의 벡터, embedding vector에 $W^q$ 활용해서 생성
-   key vector: $d_k$(=64) 차원의 벡터, embedding vector에 $W^k$ 활용해서 생성
-   value vector: $d_v$(=64) 차원의 벡터, embedding vector에 $W^v$ 활용해서 생성

query 벡터와 key 벡터는 dot product 연산을 취하기 때문에 $d_k$차원으로 동일하다.
value 벡터는 $d_v$차원으로 query 벡터와 key 벡터 동일한 차원으로 만들어도 되고 그렇지 않아도 상관없다.

#### (2) query 벡터와 key 벡터간 dot product 연산

각 단어의 query 벡터와 각 단어의 key 벡터들끼리 dot product 연산을 진행하여 각 단어의 연관된 정도를 계산한다.

-   $q_1 \cdot k_i \, (for \, 1 \leq i \leq sequence \, length)$
-   첫번째 단어 벡터와 나머지 벡터들 간의 연관된 정도 계산

#### (3) 계산된 dot product 값에 scaling

dot product 값에 $\frac{1}{\sqrt{d_k}}$ 을 곱해주어 scaling을 진행한다.

scaling을 해주는 이유는 $d_k$의 값이 커질수록 역전파시 softmax 연산에 대한 기울기 값이 작아지는 것을 방지하기 위해서다.

-   $\frac{1}{\sqrt{d_k}}(q_1 \cdot k_i)$

#### (4) scaled된 dot product에 softmax 연산

softmax를 이용하여 연관된 정도를 확률로 표현

-   $softmax(\frac{q_1 \cdot k_i}{\sqrt{d_k}})$
-   즉, 첫번째 단어와 나머지 단어들 간의 연관된 정도를 확률로 표현

이렇게 구한 값을 'attention weight' 이라고 부른다.
이제 첫번째 단어와 모든 단어 벡터들 간의 연관된 정도를 표현하는 'attention weight'을 구하였다.

이 attention weight을 이용하여 시각화하면 각 단어들이 얼마나 연관이 있는지 한 눈에 확인할 수 있다. attention을 활용하면 어느정도 설명하기 쉬운 모델이 된다고 볼 수 있겠다.

#### (5) softmax 출력값을 value벡터와 곱

softmax 연산을 이용해 최종적으로 구한 attention weight을 각 단어의 value 벡터에 곱한 후 합하여 첫번째 단어에 대한 context vector를 계산한다.

context vector에 첫번째 단어가 나머지 단어와 얼마나 연관되어 있는지에 대한 정보가 모두 담겨있다고 볼 수 있다.

이제 이렇게 구한 context vector가 다음 층의 입력 벡터가 된다.

-   $weighted\,sum(v_k \, softmax(\frac{q_1 \cdot k_i}{\sqrt{d_k}}))$

### 통합

위 과정을 모든 단어 벡터들에 대해 수행해주면 된다.
모든 벡터들을 각각의 행렬을 이용해 합친 후 계산하면 편하다.

$Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$

-   $Q$: query matrix, 각 단어 벡터들에 대한 query 벡터가 담겨 있다.
-   $K$: key matrix, 각 단어 벡터들에 대한 key 벡터가 담겨 있다.
-   $V$: value matrix, 각 단어 벡터들에 대한 value 벡터가 담겨 있다.

## 4. Multi-head Attention

<center><img src='/assets/image/transformer/transformer2.png'></center>
<center><a src='https://arxiv.org/abs/1706.03762'>Attention Is All You Need</a></center>

Transformer는 SDPA를 $h$번 적용한 Multi-head Attention을 이용한다.

Multi-head Attention을 사용하는 이유는 서로 다른 h개의 representaion 공간을 만들어 이를 이용한 정보를 활용할 수 있기 때문이다. SDPA를 1번만 적용해도 다른 vector에 대한 연관된 정도를 확인할 수 있지만 사실 자기 자신에 대해 높은 확률로 표현하기 때문에 이 의미가 퇴색될 수 있다. 따라서 여러번 적용해주어 여러 개의 representation을 활용한다.

$MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O$

-   $where\, head_i = Attention(QW_i^Q, KW_i^K,VW_i^V)$

각각의 SDPA에 적용된 결과 값을 concatentation한 뒤 $W^O$를 행렬 곱하여 다음 layer의 input 값으로 활용될 수 있게 한다.

### Transformer에 적용

encoder, decoder내에서 각각 Multi-head Attention을 활용한다.

#### decoder 내 encoder-decoder attention layer

encoder-decoder attention layer는 decoder 내에서 이전 layer의 output을 query로, encoder의 output을 key, value로 활용하는 Multi-attention layer를 뜻한다. key와 value를 encoder의 output을 이용함로써 decoder의 각 단어들이 input sequence와의 연관성을 계산할 수 있게 해준다.

#### encoder 내 self-attention layer

이 경우 self-attention 내의 query, key, value들은 이전 layer의 output이 쓰인다.

## 5. Transformer 구조

<center><img src='/assets/image/transformer/transformer3.png'></center>
<center><a src='https://arxiv.org/abs/1706.03762'>Attention Is All You Need</a></center>

### Encoder 구조

#### (1) positional encoding

attention은 모든 단어들끼리 dot product 연산을 사용하기 때문에 순서(position)에 대한 정보가 거의 없다. 이를 보완하기 위해 Embedding vector에 position vector를 더하여 순서를 입력해준다.

#### (2) Multi-head Attention

self-attention을 진행한다.

#### (3) Add & Norm

encoder, decoder 모두 residual connection을 활용한다.

position encoding을 통과한 input x와 Multi-head Attention 출력값을 더한다.

-   $x = x + Multi-head Attention Layer(x)$

residual connection을 이후 layer normalization을 적용해준다.

layer normalization은 마지막 차원에 대해 $\mu$와 $\sigma$를 구한 뒤 각 원소에 대해 normalization을 적용해준다.

-   $x_{i,k} = \frac{x_{i,k} - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}}$

#### (4) Feed Forward

activation function으로 ReLU를 활용한 신경망을 활용한다.

$FFN(x) = max(0, xW_1 + b1)W_2 + b_2$

-   input, output feature 수: $d_{model}$

### Decoder 구조

#### (1) Masked Multi-head Attention

지금 이후 시점에 대한 정보를 참고하지 못하도록 masking 해놓은 Multi-head Attention이다.

#### (2) Multi-head Attention

query에 decoder 이전 layer의 출력값이 key, vector는 encoder의 output을 활용하는 'encoder-decoder' attention을 활용한다.

#### (3) Linear, Softmax

최종적으로 decoder에서 나온 출력값을 활용하여 linear layer와 softmax를 통과시킨 뒤 가장 높은 확률을 가진 단어를 최종 결과물로 선택한다.
