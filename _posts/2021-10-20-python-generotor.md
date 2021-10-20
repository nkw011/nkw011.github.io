---
title: "[Python] Generator"
excerpt: "Python Generator, Iterator 정리"
toc: true
toc_sticky: true
categories:
    - python
tags:
    - python
sidebar:
    nav: sidebarTotal
---

generator는 동시성을 위해 사용하는 python 비동기 프로그래밍에 쓰이는 중요한 문법이다.
특히 Coroutine을 이해하기 위해 필수로 알아야한다.

## 1. iterator

먼저 iterator에 대해 살펴보자.

iterator는 list같은 sequence 자료형과 비슷해보이지만 다른 점이 존재한다.
list는 필요한 요소를 먼저 한 번에 생산하기 때문에 메모리가 iterator에 비해 많이 사용된다.
그러나 iterator는 필요할 때(e.g. 메소드를 사용할 때) 요소를 생산하기 때문에 메모리 사용이 list같은 자료형보다는 적다고 말할 수 있다.

반복가능한 자료형은 iterator로 만들 수 있다.
조금 더 구체적으로 말하면 클래스에 `__iter__` 메소드가 구현되어 있다면 iterator로 만들 수 있다.
`__iter__` 메소드를 구현한 클래스의 객체를 iterator로 만드는 방법은 `iter()`를 사용하면 된다.
`iter()`를 사용하면 해당 객체의 `__iter__` 메소드를 호출하게 되고 이 `__iter__` 메소드는 iterator를 반환하여 넘겨준다. `iter()`는 넘겨받은 iterator를 반환한다.

```py
t = list(range(1,11))

print('__iter__' in dir(t))

it1 = iter(t)
print(it1)

print(next(it1))
print(next(it1))
print(next(it1))
print(next(it1))
print(next(it1))

print()

for num in it1:
    print(num)

# 위 for-loop는 아래 while문과 동일하게 동작한다.
while True:
    try:
        print(next(it1)) # 더이상 반환할 다음 원소가 없다면 StopIteration을 발생시킨다.
    except StopIteration:
        break
print()
```

```
결과

True
<list_iterator object at 0x7fbd4d5a21f0>
1
2
3
4
5

6
7
8
9
10
```

iterator는 더이상 반환할 다음 원소가 없다면 StopIteration을 발생시킨다.

### 1.1. class 기반 iterator 만들기

class를 iterator로 활용하려면 class 내부에 `__iter__`,`__next__`가 구현되어 있어야한다.

다음은 띄어쓰기로 된 문장을 단어별로 분리하는 클래스이다.

```py
class SetenceSplitter:

    def __init__(self,sentence):
        self._sentence = sentence.split(' ')
        self._index = 0

    def __next__(self):
        try:
            result = self._sentence[self._index]
        except IndexError:
            raise StopIteration('더 이상 문장에 단어가 존재하지 않습니다.')
        self._index += 1
        return result

s1 = SetenceSplitter('I think therefore I am')
print(next(s1))
print(next(s1))
print(next(s1))
print(next(s1))
print(next(s1))
print(next(s1))
```

```
결과

I
think
therefore
I
am

StopIteration: 더 이상 문장에 단어가 존재하지 않습니다.
```

`__next__`메소드는 더이상 반환할 원소가 없으면 StopIteration을 발생하도록 하였다.

```py
class SetenceSplitter:

    def __init__(self,sentence):
        self._sentence = sentence.split(' ')
        self._index = 0

    def __next__(self):
        try:
            result = self._sentence[self._index]
        except IndexError:
            raise StopIteration('더 이상 문장에 단어가 존재하지 않습니다.')
        self._index += 1
        return result

    # __iter__ method는 iterator를 반환해야한다.
    def __iter__(self):
        return iter(self._sentence)

s2 = SetenceSplitter('I think therefore I am')

for s in s2:
    print(s)

```

```
결과

I
think
therefore
I
am
```

`__iter__`메소드를 구현하여 iterator처럼 동작하게하였다.

### 1.2. itertools

다양한 용도의 iterator를 반환하는 메소드를 가지고 있는 라이브러리이다.

예시를 통해 몇가지를 소개해본다.

```py
# iterator 중요 library

# 출력을 위한 메세지
msg = "\n{}: {}"

import itertools

# 조합 가지 수 반환
gen1 = itertools.combinations(range(4),3)
print(msg.format("combination",list(gen1)))

# 순열 가지 수 반환
gen2 = itertools.permutations(range(4),3)
print(msg.format("permutation",list(gen2)))

# count(start, step) -> iifinite iterators
gen3 = itertools.count(1,2.5)
print("\ncount 결과")
print(next(gen3))
print(next(gen3))

# takewhile(predicate, iterable): predicate is true일 때까지 iterable에서 원소를 만들어내는 메소드
gen4 = itertools.takewhile(lambda x :x < 100,itertools.count(1,2.5))
print("\ntakewhile 결과")
for num in gen4:
    print(num,end=' ')
print()

# accumulate(iterable[,func])
# 기본적으로 iterable의 누적된 합을 보여준다.
# 만약 func에 함수가 들어온다면 iterable에 func이 누적된 결과값을 보여준다.
gen5 = itertools.accumulate(range(1,11))
print(msg.format("accumulate",list(gen5)))

gen6 = itertools.accumulate([3,2,5,1,6,4,4,9],max)
print(msg.format("accumulate with func",list(gen6)))


# product(*iterable, repeat=1): 길이가 repeat인 cartesian product를 생산한다.
gen7 = itertools.product('ABCDE')
print(msg.format("product with repeat=1",list(gen7)))

gen8 = itertools.product('ABCDE', repeat=2)
print(msg.format("product with repeat=2",list(gen8)))


# chain(*iterable): 서로다른 iterable을 연결해준다.
gen9 = itertools.chain("ABCDE","123")
print(msg.format("chain",list(gen9)))

# groupby(iterable,key=None): 같은 그룹끼리 묶어준다.
gen10 = itertools.groupby('AAABBCCCCDDEEE')
print(msg.format("groupby",list(gen10))) # 같은 그룹으로 묶인 것이 class로 나온 것을 알 수 있다.

print()

gen10 = itertools.groupby('AAABBCCCCDDEEE')
for c, group in gen10:
    print("{}: {}".format(c,list(group)))
```

```
결과

combination: [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]

permutation: [(0, 1, 2), (0, 1, 3), (0, 2, 1), (0, 2, 3), (0, 3, 1), (0, 3, 2), (1, 0, 2), (1, 0, 3), (1, 2, 0), (1, 2, 3), (1, 3, 0), (1, 3, 2), (2, 0, 1), (2, 0, 3), (2, 1, 0), (2, 1, 3), (2, 3, 0), (2, 3, 1), (3, 0, 1), (3, 0, 2), (3, 1, 0), (3, 1, 2), (3, 2, 0), (3, 2, 1)]

count 결과
1
3.5

takewhile 결과
1 3.5 6.0 8.5 11.0 13.5 16.0 18.5 21.0 23.5 26.0 28.5 31.0 33.5 36.0 38.5 41.0 43.5 46.0 48.5 51.0 53.5 56.0 58.5 61.0 63.5 66.0 68.5 71.0 73.5 76.0 78.5 81.0 83.5 86.0 88.5 91.0 93.5 96.0 98.5

accumulate: [1, 3, 6, 10, 15, 21, 28, 36, 45, 55]

accumulate with func: [3, 3, 5, 5, 6, 6, 6, 9]

product with repeat=1: [('A',), ('B',), ('C',), ('D',), ('E',)]

product with repeat=2: [('A', 'A'), ('A', 'B'), ('A', 'C'), ('A', 'D'), ('A', 'E'), ('B', 'A'), ('B', 'B'), ('B', 'C'), ('B', 'D'), ('B', 'E'), ('C', 'A'), ('C', 'B'), ('C', 'C'), ('C', 'D'), ('C', 'E'), ('D', 'A'), ('D', 'B'), ('D', 'C'), ('D', 'D'), ('D', 'E'), ('E', 'A'), ('E', 'B'), ('E', 'C'), ('E', 'D'), ('E', 'E')]

chain: ['A', 'B', 'C', 'D', 'E', '1', '2', '3']

groupby: [('A', <itertools._grouper object at 0x7fbd4d68c3d0>), ('B', <itertools._grouper object at 0x7fbd4d68c4c0>), ('C', <itertools._grouper object at 0x7fbd4d68c400>), ('D', <itertools._grouper object at 0x7fbd4d68c490>), ('E', <itertools._grouper object at 0x7fbd4d68c4f0>)]

A: ['A', 'A', 'A']
B: ['B', 'B']
C: ['C', 'C', 'C', 'C']
D: ['D', 'D']
E: ['E', 'E', 'E']
```

## 2. generator

generator는 iterator를 반환하는 함수이다.
일반적인 함수처럼 작성되지만 yield 구문을 사용한다는 점에서 차이가 있다.

```py
def geneEx1():
    print('start')
    yield 1
    print('middle')
    yield 2
    print('end')

g1 = geneEx1()
print(next(g1))
print(next(g1))

print()

l1 = [x*3 for x in geneEx1()]
print(l1)

print()

g2 = (x*3 for x in geneEx1())
for i in g2:
    print(i)
```

```
결과

start
1
middle
2

start
middle
end
[3, 6]

start
3
middle
6
end
```

### 2.1. generator 작동 방식

함수에 yield 구문을 사용하게되면 그 함수는 generator가 된다. 이 함수는 제네레이터 함수(generator function)라고 한다.
제네레이터 함수가 호출되면 제네레이터 이터레이터(genenrator iterator)를 반환한다.

반환된 제네레이터 이터레이터는 제네레이터 함수의 실행을 통제한다.
제네리이터 함수는 제네레이터의 메소드가 호출이 되었을 때 내부를 실행하기 시작한다.
(제네레이터 메소드는 제네레이터 함수가 아니다.)

제네레이터의 메소드가 호출이 되면 제네레이터 함수는 첫번째 yield 구문까지 진행된고 다시 중단된다.
첫번째 yield 구문은 yield 구문의 expression을 실행한다. 그리고 실행한 결과값(expression이 실행되고 반환되는 값)을 호출한 제네레이터 메소드로 반환한다.(즉, 제네레이터 메소드로 넘겨준다.)

중단이 될 때 지금까지 사용된 지역 변수를 포함한 모든 local state는 유지된다.

제네레이터 함수는 제네레이터의 메소드가 호출이 되면 다시 실행되는데 중단된 지점부터 다음번 yield 구문까지 진행된다.
이 때 다음번 yield 구문이 없다면 StopIteration을 발생시킨다.

yield 구문 자체의 반환값(yield 구문 옆에 있는 expression이 실행된 반환되는 값 x)은 실행을 재개한 제네레이터의 메소드에 의존한다.

만약 next()라면 yield 구문의 반환값은 None이 된다.
먄약 send()가 사용이 되었다면 반환값은 send()에 의해 넘겨진 값이 된다.

```
[yield 구문과 메소드 정리]

1. yield {expression}
yield {expression}: expression에 있는 것을 생성하여 제네레이터 메소드로 넘겨준다.
반환값은 기본적으로 None이다. 단, send()같이 값을 넘겨주는 함수가 있다면 send()에 의해 넘겨주는 값을 반환한다.

2. next(generator)
제네레이터 메소드이기 때문에 호출하면 제네레이터 함수가 재개된다.
iterable의 `__next__`를 호출하여 얻어낸 결과값을 반환한다.
`__next__`는 보통 iterable 객체의 다음 item을 반환한다.(즉, 아직 반환되지 않은 것 중 첫번째 아이템을 반환한다.)
정리하자면 결국 next(iterable)은 iterable 객체의 다음 item을 반환하는 역할을 한다.

3. generator.send(value)
제네레이터 메소드이기 때문에 호출하면 제네레이터 함수가 재개된다.
제네레이터 함수를 재개하는 동시에 인자로 넣은 값을 제네레이터 함수로 넘겨주는데 이 때 인자로 넣은 값은 현재 yield구문의 반환값이 된다.
send() 자체 반환 값은 next()와 동일하게 yield expression의 실행 결과값이다.

p.s. send()와 next() 모두 다음 차례 yield 구문까지 실행이 된다. 값을 넘겨주는 것은 다음번 next()나 send()에 의해 결정된다.
```

다음은 예시를 이용해서 작동원리를 설명한 것이다.

```py
def gen1():
    print("start")
    y = yield 1 # 사실상 generator 역할을 한다.(expression이 item이라고 생각하면 된다.)
    print("y: ",y)
    yield print("yield 안에 값이 아닌 함수를 넣었다.")
    print("end")

t1 = gen1() # genearator를 생성한다.
print(t1)
next(t1)
print(next(t1)) # 두번째 next(t1)으로 인해 y에 None이 전달되고 y: None이 출력된다.
next(t1)

```

```
결과

<generator object gen1 at 0x7fce3248a660>
start
y:  None
yiled 안에 값이 아닌 함수를 넣었다.
None
end

StopIteration
```

-   print(t1): `gen1()`은 generator iterator를 반환하기 때문에 t1은 generator object가 되어 generator object임이 출력된다.
-   next(t1): 다음번 yield 구문인 `yield 1`까지 실행된다. yield 구문을 실행한 결과값인 1을 `next(t1)`으로 넘겨준다. 이 때 y에 반환값을 넘겨주지는 않는다. y에 값을 넘겨주는 것은 다음번 generator method에 의해 넘겨진다. 쉽게 생각하면 `y = yield 1`에서 `yield 1`까지 실행된다고 생각하면 된다.
-   print(next(t1)): y에 None값을 넘겨준다. 따라서 `gen1()`에서 `y: None`이 출력된다. `yield print("yield 안에 값이 아닌 함수를 넣었다.")`을 실행해서 `"yield 안에 값이 아닌 함수를 넣었다."`이 출력된다. `print()`에서 반환되는 값이 없기 때문에 `next(t1)`은 None 값을 가진다. 따라서 `print(next(t1))`에서 None이 출력된다.
-   next(t1): `yield print("yield 안에 값이 아닌 함수를 넣었다.")` 구문에 None 값을 보내지만 따로 받는 변수가 없기 때문에 출력이 되지 않는다. `end`가 출력되고 다음번 yield 구문이 없기 때문에 `StopIteration`이 발생한다.

```py
def gen1():
    print("start")
    y = yield 1
    print("y:",y)
    yield print("yield 안에 값이 아닌 함수를 넣었다.")
    print("end")

t1 = gen1()
next(t1)
t1.send(100)
next(t1)
```

```
결과

start
y: 100
yiled 안에 값이 아닌 함수를 넣었다.
end

StopIteration
```

`t1.send(100)`는 100을 yield 구문의 반환값으로 넘겨주기 때문에 y가 100이 된다.
