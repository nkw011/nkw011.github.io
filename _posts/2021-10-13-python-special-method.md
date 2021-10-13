---
title: "[Python] Special Method"
excerpt: "Python에서 쓰이는 special method에 관한 간단한 정리"
toc: true
toc_sticky: true
categories:
    - python
tags:
    - python
sidebar:
    nav: sidebarTotal
---

## 1. Special Method (Magic Method)

Python은 기본적으로 대부분의 데이터 타입이 class로 이루어진 구조이다.
흔히 사용하는 `int`, `float`, `str` 모두 class로 정의된 타입이다.

```py
print(int)
print(float)
print(str)
```

```
결과값

<class 'int'>
<class 'float'>
<class 'str'>
```

따라서 `1+1`, `2.5 * 3.5` 등의 사칙연산을 활용한 연산 모두 클래스 내에 정의된 메소드에 의해 작동된다.

```py
n = 10

print(n + 100)
print(n.__add__(100))
print(n*100,n.__mul__(100))
```

```
결과값

110
110
1000 1000
```

이 때 사용된 `__add__`, `__mul__` 등 따로 정의하지 않아도 내장된 메소드를 special method라고 한다.
\_\_(double underscore)가 메소드 이름에 포함되어있다.

`__add__`, `__mul__`을 포함한 다양한 special method를 [Python Reference](https://docs.python.org/3/reference/datamodel.html#special-method-names)에서 찾아볼 수 있다.

## 2. 예시: Vector 클래스 만들기

다음과 같은 연산이 가능한 Vector 클래스를 만들어보자

```
(1,2) + (2,3) : (3,5)
3 * (1,2) : (3,6)
not (1,2) : False
not (0,0) : True
```

```py
class Vector:

    def __init__(self,x,y):
        self._x = x
        self._y = y

    def __add__(self,other):
        return Vector(self._x+other._x, self._y + other._y)

    def __mul__(self,c):
        return Vector(self._x * c, self._y* c)

    def __bool__(self):
        return self._x != 0 or self._y != 0

    def __str__(self):
        return "({},{})".format(self._x, self._y)


v1 = Vector(1,2)
v2 = Vector(2,3)
v3 = Vector(0,0)
print(v1 + v2)
print(v1 * 3)
print(not v3)
```

```
결과값

(3,5)
(3,6)
True
```
