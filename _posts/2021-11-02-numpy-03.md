---
title: "[Numpy] 03. ndarray 기본 연산과 broadcasting"
excerpt: "Python Numpy Library - ndarray 기본 연산과 broadcasting"
toc: true
toc_sticky: true
categories:
    - numpy
tags:
    - python
    - numpy
sidebar:
    nav: sidebarTotal
---

```python
import numpy as np
```

## 3.1. ndarray 기본 연산

+, -, \*, /, //, %, \*\* python 기본 연산자를 numpy의 ndarray에 적용할 수 있는데 모두 원소끼리 연산이 이루어지는 형태이다.

즉, 일반적으로 알고있는 행렬 연산처럼 생각을 하면 안된다.

```python
a = np.arange(12).reshape((3,4))
b = np.arange(11,23).reshape((3,4))

print("a:\n",a)
print("b:\n",b)

print("a+b:\n",a+b)
print("a-b:\n",a-b)
print("a*b:\n",a*b)
print("a/b:\n",a/b)
```

    a:
     [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    b:
     [[11 12 13 14]
     [15 16 17 18]
     [19 20 21 22]]
    a+b:
     [[11 13 15 17]
     [19 21 23 25]
     [27 29 31 33]]
    a-b:
     [[-11 -11 -11 -11]
     [-11 -11 -11 -11]
     [-11 -11 -11 -11]]
    a*b:
     [[  0  12  26  42]
     [ 60  80 102 126]
     [152 180 210 242]]
    a/b:
     [[0.         0.08333333 0.15384615 0.21428571]
     [0.26666667 0.3125     0.35294118 0.38888889]
     [0.42105263 0.45       0.47619048 0.5       ]]

## 3.2 broadcasting

numpy는 ndarray의 shape이 일치하지 않을때도 연산을 지원한다.

서로다른 shape을 가진 ndarray를 맞추는 과정을 broadcasting이라고 하며 다음과 같이 이루어진다.

### Case1. dimension이 일치할 때

shape이 작은 쪽을 큰 쪽에 맞춰 확장한다.

-   (3,3) + (1,3) : (1,3)을 (3,3)으로 확장시켜 + 연산을 수행한다.
-   (3,3) + (3,1) : (3,1)을 (3,1)으로 확장시켜 + 연산을 수행한다.
-   (2,3,1) + (1,3,1): (1,3,1)을 (2,3,1)로 확장시켜 + 연산을 수행한다.
-   (2,1,2) + (2,3,1): (2,1,2)을 (2,3,2)로 (2,3,1)을 (2,3,2)로 확장시켜 + 연산을 수행한다.

```python
a = np.arange(9).reshape((3,3))
b = np.arange(3).reshape((1,3))

print("a:\n",a)
print("b:\n",b)
print("a+b:\n",a+b)
```

    a:
     [[0 1 2]
     [3 4 5]
     [6 7 8]]
    b:
     [[0 1 2]]
    a+b:
     [[ 0  2  4]
     [ 3  5  7]
     [ 6  8 10]]

```python
a = np.arange(9).reshape((3,3))
b = np.arange(3).reshape((3,1))

print("a:\n",a)
print("b:\n",b)
print("a+b:\n",a+b)
```

    a:
     [[0 1 2]
     [3 4 5]
     [6 7 8]]
    b:
     [[0]
     [1]
     [2]]
    a+b:
     [[ 0  1  2]
     [ 4  5  6]
     [ 8  9 10]]

```python
a = np.arange(2*3*2).reshape((2,3,2))
b = np.arange(1*3*2).reshape((1,3,2))

print("a:\n",a)
print("b:\n",b)
print("a+b:\n",a+b)
```

    a:
     [[[ 0  1]
      [ 2  3]
      [ 4  5]]

     [[ 6  7]
      [ 8  9]
      [10 11]]]
    b:
     [[[0 1]
      [2 3]
      [4 5]]]
    a+b:
     [[[ 0  2]
      [ 4  6]
      [ 8 10]]

     [[ 6  8]
      [10 12]
      [14 16]]]

```python
a = np.arange(2*3*2).reshape((2,3,2))
b = np.arange(2*3*1).reshape((2,3,1))

print("a:\n",a)
print("b:\n",b)
print("a+b:\n",a+b)
```

    a:
     [[[ 0  1]
      [ 2  3]
      [ 4  5]]

     [[ 6  7]
      [ 8  9]
      [10 11]]]
    b:
     [[[0]
      [1]
      [2]]

     [[3]
      [4]
      [5]]]
    a+b:
     [[[ 0  1]
      [ 3  4]
      [ 6  7]]

     [[ 9 10]
      [12 13]
      [15 16]]]

```python
a = np.arange(2*1*2).reshape((2,1,2))
b = np.arange(2*3*1).reshape((2,3,1))

print("a:\n",a)
print("b:\n",b)
print("a+b:\n",a+b)
```

    a:
     [[[0 1]]

     [[2 3]]]
    b:
     [[[0]
      [1]
      [2]]

     [[3]
      [4]
      [5]]]
    a+b:
     [[[0 1]
      [1 2]
      [2 3]]

     [[5 6]
      [6 7]
      [7 8]]]

### Case2. dimension이 다를 때

dimension이 작은 쪽을 큰 쪽에 맞춰서 dimension을 확장한다.

-   (2,3) + (3,) : (3,)을 (1,3)으로 만든다음 (2,3)으로 확장시켜 + 연산을 수행한다.
-   (2,3,2) + (3,2) : (3,2)를 (1,3,2)로 만든다음 (2,3,2)로 확장시켜 + 연산을 수행한다.
-   (2,3,2) + (2,): (2,)을 (2,3,2)순서로 확장한다음 + 연산을 수행한다.

즉 차원이 작은 쪽의 shape이 큰 쪽의 오른쪽 shape과 일치해야함을 알 수 있다.

```python
a = np.arange(2*3).reshape((2,3))
b = np.arange(3)

print("a:\n",a)
print("b:",b)
print("a+b:\n",a+b)
```

    a:
     [[0 1 2]
     [3 4 5]]
    b: [0 1 2]
    a+b:
     [[0 2 4]
     [3 5 7]]

```python
a = np.arange(2*3*2).reshape((2,3,2))
b = np.arange(3*2).reshape((3,2))

print("a:\n",a)
print("b:\n",b)
print("a+b:\n",a+b)
```

    a:
     [[[ 0  1]
      [ 2  3]
      [ 4  5]]

     [[ 6  7]
      [ 8  9]
      [10 11]]]
    b:
     [[0 1]
     [2 3]
     [4 5]]
    a+b:
     [[[ 0  2]
      [ 4  6]
      [ 8 10]]

     [[ 6  8]
      [10 12]
      [14 16]]]

```python
a = np.arange(2*3*2).reshape((2,3,2))
b = np.arange(2)

print("a:\n",a)
print("b:",b)
print("a+b:\n",a+b)
```

    a:
     [[[ 0  1]
      [ 2  3]
      [ 4  5]]

     [[ 6  7]
      [ 8  9]
      [10 11]]]
    b: [0 1]
    a+b:
     [[[ 0  2]
      [ 2  4]
      [ 4  6]]

     [[ 6  8]
      [ 8 10]
      [10 12]]]
