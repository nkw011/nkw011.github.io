---
title: "[Python] Named Tuple"
excerpt: "collections안에 있는 namedtuple 정리"
toc: true
toc_sticky: true
categories:
    - python
tags:
    - python
sidebar:
    nav: sidebarTotal
---

Python은 기본 데이터 타입인 tuple 이외에 collections내 정의된 namedtuple이라는 타입을 제공한다.

## 1. Named Tuple 정의하기

이름, 나이, 혈액형을 가지는 Named Tuple을 만들어보자.

```py
from collections import namedtuple

# 띄어쓰기
Person1 = namedtuple('Person1', 'name age blood')

# ,로 구분하기
Person2 = namedtuple('Person2', 'name, age, blood')

# 리스트 활용하기
Person3 = namedtuple('Person3', ['name' ,'age', 'blood'])

# 같은 field값 쓰고 싶다면 ename을 쓰면 사용할 수 있다.
Person4 = namedtuple('Person4', 'name age blood age' rename=True)
```

위 방법 중 원하는 방법으로 Named Tuple을 먼저 정의해야한다.

## 2. Named Tuple 사용하기

클래스를 이용해 객체를 만드는 것처럼 필요한 parameter수를 잘 지켜 사용하면 된다.

```py
p1 = Person1("Kim",28,'A')
print(p1[0],p1[1],p1[2])
print(p1.name, p1.age, p1.blood)
```

```
결과값

Kim 28 A
Kim 28 A
```

indexing을 이용하거나 정의한 field name을 이용해 만든 Named Tuple의 값을 사용할 수 있다.

## 3. Named Tuple 메소드, 변수

[Python Reference](https://docs.python.org/ko/3.7/library/collections.html#collections.namedtuple)에 나온 것 중 몇가지만 간단하게 살펴본다.

-   `_make()` : Named Tuple로 만들어준다.
-   `_asdict()` : Ordered Dict로 변환해준다.
-   `_fields` : 정의된 field name을 보여준다.

```py
a = ['Park',25,'AB']

p2 = Person2._make(a)
print(p2)

print(p2._asdict())
print(p2._fields)

```

```
결과값

Person2(name='Park', age=25, blood='AB')
{'name': 'Park', 'age': 25, 'blood': 'AB'}
('name', 'age', 'blood')
```
