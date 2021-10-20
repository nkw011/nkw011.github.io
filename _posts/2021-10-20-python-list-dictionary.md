---
title: "[Python] List, Dictionary"
excerpt: "Python List, Dictionary 활용"
toc: true
toc_sticky: true
categories:
    - python
tags:
    - python
sidebar:
    nav: sidebarTotal
---

## 1. List

List는 가장 기본적인 자료형이다.

-   ordered: indexing이 가능하다
-   mutable: 데이터 추가/삭제/삽입 등이 가능하다.

기본적인 사용방법은 건너뛰고 이번에는 Comprehension, filter함수, map함수 같이 사용하는 방법을 알아볼 것이다.

### 1.1 Comprehension

Comprehesion이란 sequence 자료형을 만드는 방법 중 하나로 간결하다는 것과 빠르다는 장점이 있다.
Python에서는 크게 4가지 Comprehension이 있다.

-   List Comprehension
-   Set Comprehension
-   Dictionary Comprehension
-   Generator Expression

Generator의 경우 Comprehension과 동일하지만 특별히 expression이라고 부른다.
여기서는 List Comprehension과 Generator Expression에 대해 알아볼 것이다.

List Comprehesion의 기본적인 사용방법이다.

```py
l1 = []
for i in range(10):
    l1.append(i)
print(l1)

l2 = [i for i in range(10)]
print(l2)
```

```
결과

[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

for-loop가 `[]`안으로 들어왔다. 따라서 코드의 길이가 어느정도 줄었다는 것을 알 수 있다.

다음은 2로 나누었을 때 나머지가 0인 수로 이루어진 list를 만드는 방법이다.

```py
l2 = [i for i in range(10) if i % 2 == 0]
print(l2)
```

```
결과

[0, 2, 4, 6, 8]
```

if-condition도 list안에 넣어서 코드가 줄어든 것을 알 수 있다.
이 외에도 double for-loop문을 사용한다던지 등의 List Comprehension을 적용할 수 있다.

Generator Expression에 대해 알아보기 전에 Generator에 대해 알아보자.
Generator는 iterator를 생성해주는 함수이다. iterator는 `next()` 메소드를 이용하여 순차적으로 접근이 가능한 object이다.

list와 가장 큰 차이점은 list는 한 번에 여러가지 항목에 접근할 수 있지만 generator는 iterator를 반환하기 때문에 `next()`를 이용해 한 번에 한가지 항목만 접근할 수 있다.

generator는 함수 안에서 yield를 사용하여 구현하며 yield에는 값(변수)를 지정한다.

```py
def generator(n):
    for i in range(n):
        yield i

g = generator(5)
print(next(g))
print(next(g))
print(next(g))
print(next(g))
print(next(g))
```

```
결과

0
1
2
3
4
```

기본적으로 함수를 이용해서 구현하지만 Generator Expression을 이용하면 더 간단하게 구현할 수 있다.

```py
g = (i for i in range(5))
print(next(g))
print(next(g))
print(next(g))
print(next(g))
print(next(g))

g = (i for i in range(5))
for x in g:
    print(x)
```

```
결과

0
1
2
3
4
0
1
2
3
4
```

generator는 for-loop로 접근할 수 있다.
generator는 나중에 조금 더 자세히 다룰 것이다.

### 1.2. filter, map 함수와의 사용

filter 함수는 특정 조건으로 걸러진 요소들로 iterator객체를 만들어 반환하는 함수이다.
map 함수는 반복가능한 객체를 받아서 각 요소에 함수를 적용해준다.

filter(function,object)

-   function: 조건으로 쓰일 함수
-   object: filter함수가 쓰일 대상

map(function,object)

-   funtion: 적용할 함수
-   object: function을 적용할 대상

```py
# filter 사용 예시
a = [1,2,3,4,5]

for i in filter(lambda x : x % 2 == 0, a):
    print(i,end=' ')
print()

# map 사용 예시
a = ['1','2','3','4','5']
for i in map(int,a):
    print(type(i),end=' ')
print()

# filter와 map을 같이 사용한 예시
a = ['1','2','3','4','5']

b = list(filter(lambda x : x % 2 == 0,map(int,a)))
print(b)
```

```
결과

2 4
<class 'int'> <class 'int'> <class 'int'> <class 'int'> <class 'int'>
[2, 4]
```

### 1.3. 깊은 복사와 얕은 복사

List를 사용할 때 깊은 복사와 얕은 복사의 차이점을 알아두는 것이 좋다.
종종 복사한 객체의 값을 수정하려고 할 때 깊은 복사와 얕은 복사의 차이를 잘 몰라서 의도치 않은 상황이 발생할 수 있다.

얕은 복사(Shallow Copy): 객체를 복사할 때 해당 객체의 주소값만을 복사
깊은 복사(Deep Copy): 객체를 복사할 때 객체가 지닌 인스 턴스 변수 등 값 자체를 복사

둘의 차이점은 얕은 복사는 복사된 객체가 원본 객체와 같다는 것이고 깊은 복사는 복사된 객체가 원본 객체와 전혀 다르다는 차이점이 있다.

다음 예시를 보자.

```py
l1 = [[10] * 3 for _ in range(3)] # 깊은 복사
l2 = [[10] * 3] * 3 # 얕은 복사

l1[0][1] = 3
l2[0][1] = 3
print(l1)
print(l2)

for i in l1:
    print(id(i),end = ' ')
print()

for i in l2:
    print(id(i),end = ' ')
print()
```

```
결과

[[10, 3, 10], [10, 10, 10], [10, 10, 10]]
[[10, 3, 10], [10, 3, 10], [10, 3, 10]]
140728217315712 140728217304320 140728197417280
140728217313664 140728217313664 140728217313664
```

l2는 얕은 복사를 적용했기 때문에 `l2[0][1] = 3`이 의도치 않게 l2의 모든 원소 리스트의 1번째 원소를 3으로 변경하였다.
각 원소의 id값을 출력해보면 알 수 있듯이 깊은 복사는 모든 원소의 id값이 다르지만 얕은 복사는 모든 원소의 id값이 동일하여 l2의 모든 원소는 동일한 list임을 알 수 있다.

## 2. Dictionary

Python에서 Dictionary는 Hashtable을 구현한 자료구조이다.
Hashtable은 key에 value를 저장하는 구조로 key를 이용하여 value에 빠르게 접근할 수 있다는 장점이 있다.
key는 유일한 값이어야하며 중복을 허용하지 않는다.
그러면 어떤 자료형이 key가 될 수 있을까?

### 2.1. key가 될 수 있는 자료형

key가 될 수 있는 자료형은 기본적으로 str, int, float등 우리가 알고있는 기본적인 자료형이다.
여기까지만 된다면 Dictionary가 특별한 것처럼 보이지 않는데 추가로 immutable type 즉, 수정이 불가능한 자료형도 될 수 있다.
우리가 보통 알고 있는 자료형을 수정 가능한지 아닌지 나누어보았다

-   가변형 : list, bytearray, array.array, meomoryview, collections.deque
-   불변형 : tuple, str, bytes

더 구체적으로 확인하는 방법은 `hash()`를 활용해서 알아보는 방법이 있다.

```py
t1 = (10,20,(30,40,50))
t2 = [10,20,[30,40,50]]

print(hash(t1))
print(hash(t2))
```

```
결과
465510690262297113

TypeError: unhashable type: 'list'
```

`hash()`는 hashable이 가능한 객체만을 받는다. unhashable type은 key가 될 수 없다.

### 2.2 setdefault와 Dictionary Comprehension

Dictionary를 만드는 기본적인 방법은 건너뛰고 `setdefault()`와 Dictionary Comprehension를 활용하여 dictionary를 만드는 방법에 대해 알아보자.

Dictionary는 key의 중복을 허락하지 않는다고 하였다.
그렇다면 중복된 key가 들어올다면 dictionary는 과연 어떻게 동작할까?

```py
source = [('k1','val1'),('k1','val2'),('k2','val3'),('k2','val4'),('k2','val5')]
d1 = {}
for k,v in source:
    d1[k] = v

# Dictionary Comprehension
d2 = {k:v for k, v in source}
print(d1)
print(d2)
```

```
결과

{'k1': 'val2', 'k2': 'val5'}
{'k1': 'val2', 'k2': 'val5'}
```

먼저 Dictionary Comprehension을 이용해 구성한 Dictionary를 보자.
코드의 길이가 훨씬 줄어든 것을 볼 수 있다.

다시 본론으로 돌아가서 key가 중복된다면 나중에 들어온다는 값으로 변경됨을 알 수 있다.
그러면 중복된 key가 들어오는 모든 value를 하나의 list로 만드는 dictionary를 만들려면 어떻게 해야할까?

다음과 같이 구성할 수 있을 것이다.

```py
source = (('k1','val1'),('k1','val2'),('k2','val3'),('k2','val4'),('k2','val5'))

newDict1 = {}

for k,v in source:
    if k in newDict1.keys():
        newDict1[k].append(v)
    else:
        newDict1[k] = [v]

print(newDict1)
```

```
결과

{'k1': ['val1', 'val2'], 'k2': ['val3', 'val4', 'val5']}
```

그런데 setdefault 메소드를 이용하면 더 간단하게 구현할 수 있다.

setdefault(key,default) : key가 dictionary에 있다면 해당 value를 반환, key가 없으면 default를 반환하는 함수

-   key: key로 쓰이는 변수
-   default: key가 dictionary에 없을 때 쓰이는 초기값

```py
source = (('k1','val1'),('k1','val2'),('k2','val3'),('k2','val4'),('k2','val5'))

newDict2 = {}
for k,v in source:
    newDict2.setdefault(k,[]).append(v)
print(newDict2)
```

```
결과

{'k1': ['val1', 'val2'], 'k2': ['val3', 'val4', 'val5']}
```
