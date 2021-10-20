---
title: "[Python] Cocurrency"
excerpt: "Python Cocurrency(동시성) 정리"
toc: true
toc_sticky: true
categories:
    - python
tags:
    - python
sidebar:
    nav: sidebarTotal
---

**p.s. 처음 배우는 내용을 정리한 것이라 오류가 많이 있을 수 있습니다.(추후 복습시 반영 예정)**

만약 은행에 창구가 하나라고 생각해보자.
사람들은 줄을 서서 자신의 차례가 올 때까지 기다릴 것이며 은행 직원의 처리 능력에 따라 대기 시간이 달라질 것이다.
어쩌면 오래걸릴 수도 있고 그렇지 않을 수도 있다.
은행이 창구를 2개 더 늘려 3개로 만든다고 생각해보자.
일단 사람들이 3개의 창구로 분산될 것이다. 대기시간도 1개의 창구만 있을 때보다 줄어들 것이고 보다 많은 사람들의 일을 처리할 수 있을 것이다.
이렇게 은행 일을 여러 창구가 동시에 맡아 처리하는 것처럼 여러가지 일을 동시에 처리하는 것을 동시성이라고 할 수 있다.

## 1. 동기와 비동기

동시성은 동기와 비동기라는 단어와도 자주 언급되어있다.
여러가지 일을 동시에 처리해서 결과를 받아내기 위해서는 비동기적인 특성이 필요하다.
따라서 동기와 비동기에 대해 설명하기 전에 먼저 동기와 비동기에 대해 먼저 언급을 해보겠다.

### 1.1. 동기(synchronous)

동기는 요청과 결과가 한 자리에서 동시에 일어난다는 뜻이다.
동기적인 특성은 메인루틴과 서브루틴에서 자주 드러난다.
메인 루틴에서 서브 루틴을 실행한다고 해보자. 서브 루틴을 실행하게되면 서브루틴에서 일을 처리하고 결과가 나올 때까지 메인루틴은 대기하고 있어야한다. 따라서 메인루틴은 한 번에 하나의 서브루틴을 실행할 수 밖에 없다.
대부분의 일반적인 프로그램들의 코드들은 순차적으로 진행이 된다. 한 번에 한가지 작업만을 발생시키고 처리한다.
만약 어떤 함수의 결과가 다른 함수에 영향을 받는다면 그 함수는 영향을 받는 함수가 끝나고 결과를 낼 때까지 기다려야한다.
요즘같이 CPU의 성능이 좋아지고 멀티 프로세서가 발달하는 시대에서는 동기방식의 프로그램은 컴퓨터 성능을 효율적으로 쓰지 못하게되는 결과를 낳을 수도 있다. (물론 무조건 그렇다는 것은 아니다.)

### 1.2. 비동기(asynchronous)

비동기는 이러한 동기가 가지고 있는 '하나의 결과를 처리할 때까지 기다려야한다.' 특성을 깬 방식이라고 생각하면된다.
만약 당신이 일을 처리하고 있을 때 일을 도와줄 수 있는 사람이 여러명이라고 생각해보자.
이 사람들이 작업을 나눠받고 작업에 대한 결과를 당신에게 알려줄 수 있다면 당신은 그 사람들에게 일을 나눠주고 결과를 받을 때까지 기다릴 필요없이 자신의 다른 일을 수행하면 될 것이다.
프로그램이 비동기적인 특성이 있다면 여러가지 일을 동시에 수행하고 처리할 수 있다.
이렇게 되면 하나의 작업이 끝날때까지 기다리지 않고 다른 작업을 수행하면서 먼저 시작한 작업 결과를 완료하는대로 받으면 된다.
비동기는 Thread에 의한 방법과 Process에 의한 방법이 있다.

## 2. concurrent.futures

프로그래밍을 오래하지 않은 사람이 비동기적 방식으로 프로그래밍을 하려면 상당히 복잡한 수준의 프로그래밍을 해야했었다.
Python은 비동기적 프로그래밍을 비교적 간단히 구현할 수 있도록 concurrent module을 구현해놓았다.
concurrent.futures에는 동시성을 위한 멀티프로세싱과 멀티스레딩 API를 제공한다.
크게 3가지 요소로 나누어 위 모듈에 대해 살펴보도록한다.

### 2.1. Executor

비동기적으로 함수 호출을 할 수 있도록 하는 메소드를 구현한 추상클래스이다.
처리를 Thread로 할 것인지 Process로 할 것인지에 따라 Executor 클래스를 구현한 ThreadPoolExecutor와 ProcessPoolExecutor가 존재한다.
두 클래스가 제공하는 API는 거의 동일하며 주로 다음 메소드를 제공한다.

-   submit(fn,\*args) : fn에는 작업을 처리할 함수를 args에는 작업들을 넣는다. 리턴값은 병렬로 실행되는 작업을 래핑한 Future 클래스의 객체이다.

-   map(func,*args,timeout=None): 반환되는 값은 Future 객체가 아닌 함수가 적용된 결과에 대한 generator가 만들어진다. 보통의 map함수와 비슷하게 동작하지만 timeout이 존재한다. 작업을 처리할 함수를 func에 넣고 작업들을 *args에 넣는다. timeout은 작업들을 기다리기까지 제한시간이다. timeout이 없으면 작업을 끝날 때까지 계속기다리지만 timeout에 시간을 넣게되면 해당 제한시간까지 작업을 완료하지 못한 경우 예외를 발생시킨다.

### 2.2. Future

비동기로 호출된 함수들을 캡슐화 해놓은 클래스이다.
이 Future 객체들은 위에서 언급했듯이 Executor의 submit()에 의해 만들어진다.

주로 다음과 같은 메소드를 지닌다.

-   cancel(): 작업을 취소한다. 취소할 수 있으면 True를 반환하고 없으면 False를 반환한다.
-   running(): 함수가 실행중이고 취소할 수 없으면 True를 반환한다.
-   done(): 실행이 완료되었거나 함수가 성공적으로 취소되었으면 True를 반환한다.
-   result(): 함수가 실행되고 나온 결과값을 돌려준다.

### 2.3. 모듈함수

비동기 처리로인해 병렬화된 작업을 동기화 하기 위한 메소드를 2가지 제공한다.

-   concurrent.futures.wait(fs,timeout=None,return_when=ALL_COMPLETED): fs에는 Futures로 만든 객체를 넣는다. 지정된 대기 시간을 넣고 싶은 경우 timeout에 값을 넣어주면된다. 2가지 종류의 set을 반환한다. 하나는 done이며 작업이 완료된(또는 지정된 시간에 완료된) Future 객체를 포함한다. 다른 하나는 not_done이며 완료되지않은(또는 지정된 시간에 완료되지않은) Future 객체를 포함한다.

-   concurrent.futures.as_completed(fs,timeout=None): 먼저 작업이 완료된 Future객체를 순서대로 담은 iterator를 반환한다. 시간을 지정해주고 싶으면 timeout에 값을 넣으면 된다.

### 2.4. 예시

보통 다음의 순서를 지닌다.

```
[순서]

1. Executor클래스를 구현한 ThreadPoolExecurtor와 ProcessPoolExecutor중 원하는 방식의 Executor를 사용하여 작업들을 Futures객체로 만들어준다.

2. wait(), as_completed() 메소드를 이용해서 작업이 완료된 Future 객체를 받는다.

3. Future 객체의 결과를 출력한다.
```

위 순서에 따라
1에서 주어진 숫자까지의 합을 구하는 작업을 동시에 처리하려고 한다.
멀티스레드 방식을 사용헤서 구현할 것이다.

먼저 concurrent.futures를 import 해온다.

```py
import concurrent.futures # 전체를 받아오거나
# 또는
from concurrent.furture import ThreadPoolExecutor # 필요한 부분만 받아온다.
```

다음으로 작업할 것을 한번에 모아둔다.

```py
WORK_LIST = [int(1e5), int(1e6), int(1e7), int(1e8)]
```

작업을 처리할 함수를 정의한다.

```py
def sumGen(n):
    return sum(list(range(1,n+1)))
```

ThreadPoolExecutor를 사용하여 작업을 Future객체로 바꾸어준다.
바꿔준 Future 객체를 한 번에 모아두는 변수를 초기화한다.
보통 Executor는 with 구문을 이용하여 사용한다.

```py
future_list = []

with ThreadPoolExecutor() as executor:
    for work in WORK_LIST:
        future = executor.submit(sum_generator, work)
        future_list.append(future)
```

동기화를 위한 메소드를 작성한다.

```py
future_list = []

with ThreadPoolExecutor() as executor:
    for work in WORK_LIST:
        future = executor.submit(sum_generator, work)
        future_list.append(future)

results = wait(future_list,timeout=7)
print(results.done) # timeout안에 완료된 작업들 받아
print(results.not_done) # timeout안에 완료되지않은 작업들 받아오기

for f in results.done:
    print(f.result()) # 완료된 작업의 결과 받아오기
```

```py
future_list = []

with ThreadPoolExecutor() as executor:
    for work in WORK_LIST:
        future = executor.submit(sum_generator, work)
        future_list.append(future)

for f in as_completed(future_list,timeout=7):
    print(f.result()) # timeout안에 먼저 완료된 결과 받아오기
```
