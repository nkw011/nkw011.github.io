---
title: "[Numpy] 04. ndarray Indexing, Slicing, Assignment"
excerpt: "Python Numpy Library - 04. ndarray Indexing, Slicing, Assignment"
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

## 4.1. Vector Indexing, Slicing, Assignment

Python list를 접근하듯이 똑같이 ndarray에 접근할 수 있다.

### Vector Indexing

-   ndarray[-1]: 마지막 원소 접근
-   ndarray[-n]: 뒤에서 n번째 원소 접근

### Vector Slicing

-   ndarray[start:]: 뒷부분이 생략되면 '마지막 원소까지' 라는 의미 / ndarray[start]부터 마지막 원소까지 추출
-   ndarray[:end]: 앞부분이 생략되면 '처음부터' 라는 의미 / 처음부터 ndarray[end-1] 원소까지 추출
-   ndarray[start:end]: ndarray[start] 부터 ndarray[end-1] 까지 원소가 추출된다.
-   ndarray[start:end:step] : ndarray[start:end] 원소 중 step간격의 원소가 추출된다. (ndarray[start], ndarray[start+step], ...)
    (단, step이 음수라면 start에서부터 거꾸로 출력되므로 start가 end보다 커야한다. 주의 필요!)

```python
a = np.arange(10)
print("a[-1]:",a[-1],"a[-2]:",a[-2])

# 처음부터 마지막 원소전까지 출력
print("a[:-1]:",a[:-1])

# 뒤에서 5번째 원소부터 출력
print("a[-5:]:",a[-5:])

# 짝수번째 인덱스 원소들 출력
print("a[::2]:",a[::2])

# 홀수번째 인덱스 원소를 출력
print("a[1::2]:",a[1::2])

# 세번째 인덱스 원소부터 마지막 원소전까지 2간격마다의 원소들을 출력
print("a[3:-1:2]:",a[3:-1:2])
```

    a[-1]: 9 a[-2]: 8
    a[:-1]: [0 1 2 3 4 5 6 7 8]
    a[-5:]: [5 6 7 8 9]
    a[::2]: [0 2 4 6 8]
    a[1::2]: [1 3 5 7 9]
    a[3:-1:2]: [3 5 7]

### Assignment with Vector Slicing

-   ndarray[start:end:step] = a : ndarray[start:end:step]이 모두 a로 바뀐다. (step 생략하고 사용가능)
-   ndarray[start:end:step] = otherArray : ndarray[step:end:step]과 otherArray의 shape이 일치하면 otherArray로 바뀐다. (step 생략하고 사용가능)

```python
a = np.arange(12)
print("a:",a)

a[::2] = 100
print("a:",a,'\n')

b = np.arange(12)
print("b:",b)

b[:-3:2] = np.arange(11,23)[:-3:2]
print("b:",b)
```

    a: [ 0  1  2  3  4  5  6  7  8  9 10 11]
    a: [100   1 100   3 100   5 100   7 100   9 100  11]

    b: [ 0  1  2  3  4  5  6  7  8  9 10 11]
    b: [11  1 13  3 15  5 17  7 19  9 10 11]

## 4.2. Matrix Indexing, Slicing

### Matrix Indexing, Slicing

Python List에서는 M[a][b]와 같이 대괄호 2개를 이용하여 접근이 가능했지만 ndarray는 M[a,b]와 같이 ','로 구분하여 접근한다.

전체를 뜻하는 ':'는 '...'로 바꿔서 사용이 가능하다.

나머지 Indexing, Slicing 방법은 Vector Indexing, Slicing 사용하듯이 Matrix Indexing, Slicing하여 사용가능하다.

```python
a = np.arange(2*3).reshape((2,3))
print("a:\n",a)
print("a[1,2]:",a[1,2],'\n')

# 첫번째열 출력
print("a[:,0]:",a[:,0])
print("a[...,0]:",a[...,0],'\n')

# 각 행의 두번째 3번째 원소 출력
print("a[:,1:]",a[:,1:])
```

    a:
     [[0 1 2]
     [3 4 5]]
    a[1,2]: 5

    a[:,0]: [0 3]
    a[...,0]: [0 3]

    a[:,1:] [[1 2]
     [4 5]]

```python
a = np.arange(4*4).reshape((4,4))
print("a:\n",a)

# 수평선 뒤집기
b = a[::-1,:]
print("b:\n",b)

# 수직선 뒤집기
c = a[:,::-1]
print("c:\n",c)
```

    a:
     [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]
     [12 13 14 15]]
    b:
     [[12 13 14 15]
     [ 8  9 10 11]
     [ 4  5  6  7]
     [ 0  1  2  3]]
    c:
     [[ 3  2  1  0]
     [ 7  6  5  4]
     [11 10  9  8]
     [15 14 13 12]]

## 4.3. 3차원 이상 ndarray Indexing, Slicing

2차원 ndarray인 Matrix를 indexing, slicing 하는 것과 동일한 방법으로 3차원 이상의 ndarray에 접근가능하다.

```python
a = np.arange(2*3*3).reshape((2,3,3))
print("a:\n",a,'\n')

print("a[:,:-1,:]:\n",a[:,:-1,:])
```

    a:
     [[[ 0  1  2]
      [ 3  4  5]
      [ 6  7  8]]

     [[ 9 10 11]
      [12 13 14]
      [15 16 17]]]

    a[:,:-1,:]:
     [[[ 0  1  2]
      [ 3  4  5]]

     [[ 9 10 11]
      [12 13 14]]]

## 4.4. int ndarray, boolean ndarray를 이용한 Indexing

indices: int ndarray라고 가정

-   a[indices]: shape - indices / 원소 - indices의 원소에 해당하는 index를 가진 a의 원소로 구성됨
-   a (>,%,!=,==, bool값을 반환하는 연산자) 0 : ndarray 각 원소를 연산자로 판별하여 맞으면 True, 아니면 False로 구성된 bool ndarray가 반환됨
-   a[a>=0]: ndarray의 원소중 0이상인 것만 추출됨

```python
a = np.random.randint(-6,6,(6,))
indices = np.array([0,2,4])
print("a:\n",a,'\n')

print("indices:",indices)
print("a[indices]:",a[indices])
print("a>=0:",a>=0)
print("a[a>=0]:",a[a>=0])
```

    a:
     [-4 -2 -6  2 -6  2]

    indices: [0 2 4]
    a[indices]: [-4 -6 -6]
    a>=0: [False False False  True False  True]
    a[a>=0]: [2 2]

```python
a = np.random.randint(0,10,(6,))
indices = np.random.randint(0,6,(2,3))

print("a:",a)
print("indices:\n",indices)
print("a[indices]:\n",a[indices])
```

    a: [8 4 9 9 7 1]
    indices:
     [[2 3 1]
     [5 2 2]]
    a[indices]:
     [[9 9 4]
     [1 9 9]]

indices1, indices2: int ndarray로 가정

-   a[indices1,indices2]: shape - indices / 원소 - a[indices1의 첫번째원소, indices2의 첫번째 원소], a[indices1의 두번째 원소, indices2의 세번째 원소], a[indices1의 n번째 원소, indices2의 n번째 원소]로 구성된 ndarray를 반환함.

```python
a = np.arange(3*4).reshape((3,4))
indices1 = np.random.randint(0,3,(2,3))
indices2 = np.random.randint(0,4,(2,3))

print("a:\n",a,'\n')

print("indices1:\n",indices1)
print("indices2:\n",indices2,'\n')

print("a[indices1, indices2]:\n",a[indices1,indices2])
```

    a:
     [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]

    indices1:
     [[1 0 0]
     [1 2 0]]
    indices2:
     [[3 1 2]
     [0 0 2]]

    a[indices1, indices2]:
     [[7 1 2]
     [4 8 2]]

-   np.nonzero(a) : a 원소 중에서 0이 아닌 원소의 index를 가진 ndarray를 반환한다.
-   np.where(condition,[x,y]): condition이 True이면 x를, False이면 y를 가진 ndarray를 반환한다. x,y 생략시 해당 조건문에 True인 index가 반환된다.

```python
a = np.random.randint(-3,4,(6,))
print("a:",a)
print("nonzero:",np.nonzero(a),'\n')

b = np.random.randint(0,6,(6,))
print("b:",b)
print("where(condition,x,y):",np.where(b%2==0,b,-1))
print("where(condition):",np.where(b%2==0))
```

    a: [-2  2 -2  2  1  3]
    nonzero: (array([0, 1, 2, 3, 4, 5]),)

    b: [0 3 0 4 3 4]
    where(condition,x,y): [ 0 -1  0  4 -1  4]
    where(condition): (array([0, 2, 3, 5]),)
