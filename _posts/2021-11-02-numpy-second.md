---
title: "[Numpy] 02. ndarray 변경하기"
excerpt: "Python Numpy Library - ndarray 변경하기"
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

## 2.1. ndim, shape, size, dtype, itemsize, nbytes

-   ndarray.ndim: ndarray가 지닌 차원을 나타낸다. len(ndarray.shape)과 동일하다.
-   ndarray.size: ndarray의 전체 원소 수를 반환한다.
-   ndarray.dtype: ndarray 원소의 data type을 반환한다. (자세한 타입은 [Numpy Reference](https://numpy.org/doc/stable/reference/arrays.dtypes.html) 참고)
-   ndarray.itemsize: ndarray 원소의 byte 크기를 반환한다.
-   ndarray.nbytes: ndarray 전체 원소의 byte 크기를 반환한다. ndarray.size \* ndarray.itemsize와 동일하다.

```python
a = np.array([[1,2,3],[4,5,6]])
print("a.ndim:",a.ndim,",len(a.shape):",len(a.shape))
print("a.size:",a.size)
print("a.dtype:",a.dtype)
print("a.itemsize:",a.itemsize)
print("a.nbytes:",a.nbytes)
```

    a.ndim: 2 ,len(a.shape): 2
    a.size: 6
    a.dtype: int64
    a.itemsize: 8
    a.nbytes: 48

## 2.2. reshape ndarray, resize ndarray

-   np.reshape(ndarray,newshape), ndarray.reshape(newshape): ndarray의 shape을 newshape으로 변경한 새로운 ndarray를 반환한다. (단, newshape의 총 원소의 수는 기존 ndarray의 총 원소의 수와 동일해야한다.)

### -1 shape?

shape을 표현할 때 -1이 포함되어 있는 경우가 있다. -1은 더이상 계산하지 않아도 확실하게 shape을 알 수 있을 때 쓴다.

size가 12인 ndarray가 존재한다고 가정해보자. shape을 (2,6), (3,4), (4,3), (1,12) 등으로 다양하게 구성할 수 있을 것이다.
이 때 (2,-1)이라고 shape을 표현해보자. size가 12이고 shape중 첫번째 자리가 2로 결정되어있기 때문에 나머지 2번째 자리는 6인것을 쉽게 알 수 있다.
(-1,2)이라고 shape을 표현해보자. size가 12이고 두번째 자리가 2로 결정되어있기 때문에 첫번째 자리가 6인것을 쉽게 알 수가 있다.

이렇듯 -1은 더이상 계산하지 않아도 나머지 shape 자리를 알 수 있을 때 쓴다. 그 자리를 -1로 대체해서 쓴다고 보면 된다.

-1은 특정 shape 부분을 강조하거나 size가 클 때 자주 쓰인다.

```python
a = np.array([[1,2,3],[4,5,6]])
b = np.reshape(a,(3,2))

print("a:\n",a)
print("b:\n",b)

c = np.arange(12).reshape((3,4))
print("c:\n",c)
```

    a:
     [[1 2 3]
     [4 5 6]]
    b:
     [[1 2]
     [3 4]
     [5 6]]
    c:
     [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]

-   np.resize(ndarray,newshape): ndarray의 shape을 newshape으로 변경한 새로운 ndarray를 반환한다.
-   ndarray.resize(newshape): ndarray의 shape을 newshape으로 변경한다. (새로운 ndarray를 반환하지 않는다.)

이 때 newshape의 총 원소의 수는 기존 ndarray의 총 원소의 수와 동일하지 않아도 된다. 따라서 사용에 주의가 필요하다.

만약 newshape의 원소의 수가 기존 원소의 수보다 많다면 부족한 만큼 첫 원소부터 반복해서 채워 넣는다.

```python
a = np.array([[1,2,3],[4,5,6]])
b = np.resize(a,(3,2))
print("a:\n",a)
print("b:\n",b,'\n')

c = np.resize(a,(3,4))
print("c:\n",c,'\n')

a.resize((3,4))
print("a:\n",a)
```

    a:
     [[1 2 3]
     [4 5 6]]
    b:
     [[1 2]
     [3 4]
     [5 6]]

    c:
     [[1 2 3 4]
     [5 6 1 2]
     [3 4 5 6]]

    a:
     [[1 2 3 4]
     [5 6 0 0]
     [0 0 0 0]]

## 2.3. dtype 변경하기

-   dtype parameter 추가: ndarray를 만드는 method parameter에 dtype=newtype을 추가하면 newtype으로 dtype이 변경된다.
-   ndarray.astype(newtype): ndarray의 dtype을 newtype으로 변경해준다.

```python
a = np.array([1.5,2.5,3.5],dtype=np.int8)
print("a:",a)

b = np.arange(6,dtype=np.float32)
print("b:",b)

c = a.astype(np.bool8)
print("c:",c)
```

    a: [1 2 3]
    b: [0. 1. 2. 3. 4. 5.]
    c: [ True  True  True]

## 2.4. copy, view

-   copy: ndarray 객체를 깊은 복사한다. (원본에 영향을 미치지 않음)
-   view: ndarray 객체를 얕은 복사한다. (원본에 영향을 미침)

copy를 하면 원본에 영향을 미치지 않지만 메모리를 그만큼 더 차지하고 view는 원본에 영향을 미치지만 메모리를 공유하기 때문에 메모리를 덜 차지한다.

```python
a = np.arange(6).reshape((2,3))
b = a.copy()
c = a.view()

b[1,2] = 10
print("b:\n",b)
print("a:\n",a,'\n')

c[1,2] = 10
print("c:\n",c)
print("a:\n",a)
```

    b:
     [[ 0  1  2]
     [ 3  4 10]]
    a:
     [[0 1 2]
     [3 4 5]]

    c:
     [[ 0  1  2]
     [ 3  4 10]]
    a:
     [[ 0  1  2]
     [ 3  4 10]]

## 2.5. flatten, ravel

-   ndarray.flatten(): ndarray를 vector로 만든 새로운 ndarray를 반환한다.
-   ndarray.ravel(): ndarray를 vector로 만든 새로운 ndarray를 반환한다.

두 메소드 모두 벡터로 만든다는 공통점이 있지만 flatten은 copy하고 ravel은 view이다.

```python
a = np.arange(6).reshape((2,3))
b = a.flatten()

print("a:\n",a)
print("b:",b,'\n')

c = np.arange(12).reshape((4,3))
d = c.ravel()

print("c:\n",c)
print("d:",d)
```

    a:
     [[0 1 2]
     [3 4 5]]
    b: [0 1 2 3 4 5]

    c:
     [[ 0  1  2]
     [ 3  4  5]
     [ 6  7  8]
     [ 9 10 11]]
    d: [ 0  1  2  3  4  5  6  7  8  9 10 11]
