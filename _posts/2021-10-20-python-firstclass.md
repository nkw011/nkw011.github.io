---
title: "[Python] First Class, Closure, Decorator"
excerpt: "Python First Class, Closure, Decorator 정리"
toc: true
toc_sticky: true
categories:
    - python
tags:
    - python
sidebar:
    nav: sidebarTotal
---

## 1. 일급 함수 (First Class)

Python에서 함수는 객체(object)이다. 이는 `type()`을 써보면 알 수 있다.

```py
# 객체로서의 함수

def factorial(n):
    if n <= 2 :
        return n
    return n * factorial(n-1)

print(type(factorial))
```

```
결과

<class 'function'>
```

또한 Python에서 함수는 일급 함수(first-class function)인데 일급 함수는 함수가 다음과 같은 특성을 만족한다는 것을 뜻한다.

-   함수를 변수에 할당할 수 있다.
-   함수를 다른 함수의 인자로 전달 가능하다.
-   함수를 다른 함수의 반환값으로 쓸 수 있다.

### 1.1. 함수를 변수에 할당할 수 있다.

```py

def factorial(n):
    if n <= 2 :
        return n
    return n * factorial(n-1)

varFunc = factorial
print(varFunc(5))
print(factorial(5))
```

```
결과

120
120
```

`factorial`함수를 `varFunc`이라는 변수에 할당하였다.
할당된 `varFunc`변수는 `factorial` 함수와 똑같이 동작한다는 것을 확인할 수 있다.

### 1.2. 함수를 다른 함수의 인자로 전달 가능하다.

```py
def factorial(n):
    if n <= 2 :
        return n
    return n * factorial(n-1)

# 1! ~ 5!까지 출력
print(list(map(factorial,range(1,6))))

# 1 ~ 10까지의 수 중 짝수 출력
print(list(filter(lambda x : not(x % 2),range(1,11))))

# reduce(function, iterable,[초기값]): [초기값]이 있다면 초기값으로 없다면 첫 원소를 기준으로 iterable 객체에 하나씩 funciton을 누적으로 적용하고 나온 결과를 반환하는 함수
# 인자가 2개인 function에만 적용이 된다.
from functools import reduce
print(reduce(lambda x, y: x + y,range(1,6))) # ((((1 + 2) + 3) + 4) + 5)
```

```
결과

[1, 2, 6, 24, 120]
[2, 4, 6, 8, 10]
15
```

이번에는 `factorial`함수를 `map`함수의 인자로 전달을 하였고 정확히 동작한다는 것을 확인할 수 있다.
lambda를 이용한 익명 함수도 `filter`함수에 잘 전달되는 것을 확인할 수 있다.

다른 예제를 한 번 살펴보자.

```py
from functools import partial

ten = partial(lambda x,y: x * y,10)
print(ten(5))
```

```
결과

50
```

`partial(function, value)` 메소드는 함수에서 parameter를 1개 이상 고정하는 역할을 한다. 즉 value가 이미 function의 parameter로 들어가있다.

이 `partial` 메소드에서도 익명함수가 인자로 전달된 것을 확인할 수 있다.

### 1.3. 함수를 다른 함수의 반환값으로 쓸 수 있다.

```py
def factorial(n):
    if n <= 2 :
        return n
    return n * factorial(n-1)

def func1():
    return factorial

fact2 = func1()
print(fact2(5))
```

```
결과

120
```

`factorial`함수를 다른 함수의 반환값으로 전달할 수 있는 것을 확인할 수 있다.

## 2. Closure

Closure(클로저)란 네임스페이스의 상태값을 기억하는 함수이다.
보통 함수에서 지역변수는 함수가 종료되면 같이 사라진다.
하지만 Closure를 이용하면 이러한 변수 등의 상태값을 기억하고 활용할 수 있다.

### 2.1. 외부 함수와 내부 함수

Python은 함수 안에서 또다른 함수를 정의하는 것을 허용한다.
다음과 같은 형태로 주로 사용되는데 Closure도 이와 같은 형태를 지닌다.

```py
def outerFuntion():
    def innerFunction(arg):
        # code block
        # ...
    return innerFunction
```

바깥에 있는 함수를 외부 함수라고 하고 안쪽에 있는 함수를 내부 함수라고 지칭한다.
외부 함수는 내부 함수를 반환값으로 쓴다.

Closure는 외부 함수에 변수를 두어 내부 함수에서 사용된 결과값을 저장한다.
이 때 이러한 변수를 자유변수(free variable)라고 부르며 보통 다음과 같은 형태로 쓴다.

```py
def outerFuntion():
    freeVar1 = []
    freeVar2 = 0
    # ...
    def innerFunction(arg):
        # code block exmaple
        nonlocal freeVar2
        freeVar1.append(arg)
        freeVar2 += 1
        # ...
    return innerFunction
```

자유변수를 내부 함수에서 쓸 때는 함수의 scope를 잘 확인하여서 써야한다.
자유변수를 데이터 타입에 따라 `nonlocal`로 지정하지않고 내부 함수에서 쓰게 되면 내부 함수의 지역 변수로 사용되어 `UnboundLocalError`가 나올 수 있기 때문에 주의해서 써야한다.

### 2.2 Closure 예시

다음은 평균값을 구하는 Closure이다.

```py
def averageClosure():
    cache = []
    def averager(num):
        cache.append(num)
        return sum(cache) / len(cache)
    return averager

avg1 = averageClosure()
print(avg1(5))
print(avg1(7)) # 이전 값인 5를 기억해서 (5 + 7) / 2 = 6.0 이 출력된다.
```

```
결과

5.0
6.0
```

위 Closure는 다음과 같이 변경할 수도 있다.

```py
def anotherClosure():
    cnt,total = 0,0
    def averager(n):
        nonlocal cnt,total
        cnt += 1
        total += n
        return total/cnt
    return averager

avg3 = anotherClosure()
print(avg3(1))
print(avg3(3)) # (1+3) / 2 = 2.0
```

```
결과

1.0
2.0
```

위 예시들 모두 처음에 쓰였던 값을 기억하기 때문에 올바른 평균값을 구한다.

## 3. Decorator

Decorator는 장식가, 인테리어 디자이너라는 뜻을 가지고 있다.
Python에서도 이런 의미와 비슷하게 쓰이는데 Decorator는 주어진 상황 및 용도에 맞게 기존의 코드에 여러가지 기능을 추가하는 문법이다.

다음 예시는 Decorator로 쓰일 함수를 하나 만든 것이다.

```py
import time

def checkPerformance(function):
    def clocked(*args):
        start = time.time()
        result = function(*args)
        end = time.time()
        funcName = function.__name__
        print('%s: %fs' % (funcName, end-start))
        return result
    return clocked

def average(*args):
    cnt, total = 0,0
    for arg in args:
        cnt += 1
        total += arg
    return total / cnt

def summation(*args):
    return sum(args)

avgCheck = checkPerformance(average)
sumCheck = checkPerformance(summation)

avgCheck(*range(1,51))
sumCheck(*range(1,51))
```

```
결과

average: 0.000007s
summation: 0.000002s
```

위 예시에서의 Closure는 함수의 실행시간을 간단히 측정한 것이다.
이 Closure를 Decorator로 쓰일 함수로 사용할 것인데 Decorator로 쓰는 방법은 간단하다.
사용하려는 대상 함수 위에 `@만든함수이름`을 붙이면 Decorator로 쓸 수 있다.

```py
import time

def checkPerformance(function):
    def clocked(*args):
        start = time.time()
        result = function(*args)
        end = time.time()
        funcName = function.__name__
        print('%s: %fs' % (funcName, end-start))
        return result
    return clocked

@checkPerformance
def average(*args):
    cnt, total = 0,0
    for arg in args:
        cnt += 1
        total += arg
    return total / cnt

@checkPerformance
def summation(*args):
    return sum(args)

average(*range(1,51))
summation(*range(1,51))
```

```
결과

average: 0.000007s
summation: 0.000002s
```

`average`함수와 `summation`함수 위에 `@checkPerformance`를 써서 `checkPerformance`함수를 Decorator로 활용하였다.

Decorator는 주로 로깅, 프레임워크, 유효성을 체크할 때 쓰인다.
