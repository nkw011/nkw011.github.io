---
title: "[Matplotlib.pyplot] 03. ticks, limit, spine, grid"
excerpt: "데이터 시각화를 위한 Matplotlib.pyplot 사용법 - 03. ticks, limit, spine, grid"
toc: true
toc_sticky: true
categories:
    - matplotlib
tags:
    - python
    - matplotlib
sidebar:
    nav: sidebarTotal
---

```python
import matplotlib.pyplot as plt
import numpy as np
```

## 1. Ticks

tick은 눈금을 가리킨다.

아무것도 없이 ax를 그렸을 때 각각의 축은 기본적으로 0~1까지 5개의 간격이 존재하며 따라서 총 6개의 tick이 존재한다.
이 tick을 조절하는 방법을 알아보자.

### 1.1. set_xticks(), set_yticks()

tick을 설정할 때 쓰인다.

**[Axes.set_xticks(ticks)](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xticks.html)**

-   ticks: list of floats, 리스트에 담긴 원소가 tick이 된다

**[Axes.set_yticks(ticks)](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_yticks.html)**

-   ticks: list of floats, 리스트에 담긴 원소가 tick이 된다

반대로 tick을 얻어오려면 set대신 get을 사용하면 된다.

```python
fig,ax = plt.subplots()

xticks = np.arange(0,60,10)
yticks = np.arange(0,60,10)

ax.set_xticks(xticks)
ax.set_yticks(yticks)
```

    [<matplotlib.axis.YTick at 0x7fdcad028b20>,
     <matplotlib.axis.YTick at 0x7fdcac8a14f0>,
     <matplotlib.axis.YTick at 0x7fdcac8f1580>,
     <matplotlib.axis.YTick at 0x7fdcacf68e20>,
     <matplotlib.axis.YTick at 0x7fdcacf4e460>,
     <matplotlib.axis.YTick at 0x7fdcacf71e80>]

<img src="/assets/image/matplotlib-pyplot-03_files/matplotlib-pyplot-03_4_1.png">

### 1.2. Tick Label

각 tick에 label을 설정해줄 수 있다.

**[Axes.set_xticklabels(labels)](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xticklabels.html)**

-   labels: tick에 쓰일 label을 담는다. tick의 갯수와 동일해야한다.
-   label이기 때문에 font property를 적용할 수 있다.

**[Axes.set_yticklabels(labels)](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_yticklabels.html)**

-   labels: tick에 쓰일 label을 담는다. tick의 갯수와 동일해야한다.
-   label이기 때문에 font property를 적용할 수 있다.

set대신 get을 사용하면 tick label을 가져올 수 있다.

```python
fig, ax = plt.subplots()

ticks = np.arange(6)
tick_labels = ["tick" + str(i) for i in range(6)]

ax.set_xticks(ticks)
ax.set_yticks(ticks)

ax.set_xticklabels(tick_labels)
ax.set_yticklabels(tick_labels)
```

    [Text(0, 0, 'tick0'),
     Text(0, 1, 'tick1'),
     Text(0, 2, 'tick2'),
     Text(0, 3, 'tick3'),
     Text(0, 4, 'tick4'),
     Text(0, 5, 'tick5')]

<img src="/assets/image/matplotlib-pyplot-03_files/matplotlib-pyplot-03_6_1.png">

### 1.3. major, minor

기본 tick(major)외에 추가로 표시할 수 있는 tick을 minor라고 한다.
주로 새부 간격이나 라인을 표시할 때 많이 쓰인다.

major인지는 따로 표시를 하지 않아도 되고 tick을 minor로 쓰고싶다면 parameter에 minor=True 를 추가하면 된다.

```python
fig, ax = plt.subplots(figsize=(7,7))

major_ticks = np.arange(0,21,5)
minor_ticks = np.arange(21)

ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks,minor=True)

ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks,minor=True)
```

    [<matplotlib.axis.YTick at 0x7fdcad024580>,
     <matplotlib.axis.YTick at 0x7fdcac8d1be0>,
     <matplotlib.axis.YTick at 0x7fdcac7ee4f0>,
     <matplotlib.axis.YTick at 0x7fdcac7ee6d0>,
     <matplotlib.axis.YTick at 0x7fdcacd0d370>,
     <matplotlib.axis.YTick at 0x7fdcac944790>,
     <matplotlib.axis.YTick at 0x7fdcac7ee7f0>,
     <matplotlib.axis.YTick at 0x7fdcac7ee910>,
     <matplotlib.axis.YTick at 0x7fdcacaf6eb0>,
     <matplotlib.axis.YTick at 0x7fdcac8fec10>,
     <matplotlib.axis.YTick at 0x7fdcad1956d0>,
     <matplotlib.axis.YTick at 0x7fdcad195f10>,
     <matplotlib.axis.YTick at 0x7fdcad195f40>,
     <matplotlib.axis.YTick at 0x7fdcad195b50>,
     <matplotlib.axis.YTick at 0x7fdcacaf6e50>,
     <matplotlib.axis.YTick at 0x7fdcacd0d280>,
     <matplotlib.axis.YTick at 0x7fdcac7c1040>,
     <matplotlib.axis.YTick at 0x7fdcac82bfa0>,
     <matplotlib.axis.YTick at 0x7fdcac82bb80>,
     <matplotlib.axis.YTick at 0x7fdcac82bf70>,
     <matplotlib.axis.YTick at 0x7fdcac815fa0>]

<img src="/assets/image/matplotlib-pyplot-03_files/matplotlib-pyplot-03_8_1.png">

### 1.4. tick_params()

tick의 appearence나 label을 설정할 때 쓰인다.

**[Axes.tick_params()](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.tick_params.html)**

-   axis: {'x', 'y', 'both'}, default: 'both'
-   which: {'major', 'minor', 'both'}, default: 'major'
-   direction: {'in', 'out', 'inout'} / tick의 방향을 결정한다. 축의 안쪽으로 할 것인지 바깥쪽으로 할 것인지 정한다.
-   length: 길이
-   width : 두께
-   color: 색상
-   labelsize: label 크기
-   labelcolor: label 색상
-   bottom, top, left, right: 각 축마다 tick을 표시하는지 하지 않는지 설정
-   labelbottom, labeltop, labelleft, labelright: 각 축마다 label을 표시하는지 하지 않는지 설정

```python
fig, ax = plt.subplots(figsize=(7,7))

major_ticks = np.arange(0,21,5)
minor_ticks = np.arange(21)

ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks,minor=True)

ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks,minor=True)

ax.tick_params(which='major',length=5,width=2,
               labelsize=10,labelcolor='b')
ax.tick_params(which='minor',length=3)
```

<img src="/assets/image/matplotlib-pyplot-03_files/matplotlib-pyplot-03_10_0.png">

```python
fig, ax = plt.subplots(figsize=(7,7))

ticks = np.arange(0,21,5)

ax.set_xticks(ticks)
ax.set_yticks(ticks)

ax.tick_params(left=False,labelleft=False,top=True,right=True,
               direction='in', length=5)
```

<img src="/assets/image/matplotlib-pyplot-03_files/matplotlib-pyplot-03_11_0.png">

## 2. limit

각 축에 limit을 설정할 수 있다.

**[Axes.set_xlim(left,right)](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xlim.html)**

-   left: 시작지점. 즉, 축이 가질 수 있는 최솟값
-   right: 마지막지점. 즉, 축이 가질 수 있는 최댓값

**[Axes.set_ylim(left,right)](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_ylim.html)**

-   left: 시작지점. 즉, 축이 가질 수 있는 최솟값
-   right: 마지막지점. 즉, 축이 가질 수 있는 최댓값

보통 left,rigth를 list형태로 묶어서 parameter로 넘겨주기도 한다.

각 축에 설정된 값을 가져오려면 set대신 get을 쓰면 된다.

```python
fig = plt.figure(figsize=(7,7))

ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

ticks = np.arange(0,0.6,0.1)
lims = [0,0.5]

ax1.set_xticks(ticks)

ax2.set_xlim(lims)
ax2.set_ylim(0,0.5)

ax2.set_xticks(ticks)
ax2.set_yticks(ticks)

```

    [<matplotlib.axis.YTick at 0x7fdcac7ff850>,
     <matplotlib.axis.YTick at 0x7fdcac7ff430>,
     <matplotlib.axis.YTick at 0x7fdcac7dd280>,
     <matplotlib.axis.YTick at 0x7fdcacbaed00>,
     <matplotlib.axis.YTick at 0x7fdcacb9d250>,
     <matplotlib.axis.YTick at 0x7fdcacb9d760>]

<img src="/assets/image/matplotlib-pyplot-03_files/matplotlib-pyplot-03_13_1.png">

## 3. spine

축의 선을 [spine](https://matplotlib.org/stable/api/spines_api.html?highlight=spine#module-matplotlib.spines)이라고 한다.

spine을 불러오기위해서는 axes.spines을 사용해야하며 dictionary는 아니지만 dictionary처럼 각 축을 'left','right','top','bottom'와 같은 key로 접근할 수 있다. items를 이용해 key와 spine을 모두 불러올 수 있다.

**Methods**

-   spine.set_visible(bool_value): default=True, False를 사용하면 축이 보이지 않게 할 수 있다.
-   spine.set_linewidth(width): line의 width를 설정
-   spine.set_color(c): color 설정
-   spine.set_position(position): 축의 위치를 조정한다.
    -   position: 2-tuple로 이루어져있다. 첫번째는 position type, 두번재는 amount이다.
    -   position type: 'axes'-spine 위치를 axes coordinate를 이용해 설정, 'data'-spine 위치를 data coordinate를 이용해 설정
    -   amount: 위치할 값
    -   'center', 'zero'를 쓰기도 하는데 'center'=('axes',0.5), 'zero'=('data',0.0)을 가리킨다.

```python
fig,axes = plt.subplots(2,1,figsize=(7,7))

axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
```

<img src="/assets/image/matplotlib-pyplot-03_files/matplotlib-pyplot-03_15_0.png">

```python
fig, ax = plt.subplots(figsize=(7,7))

ax.set_xlim([-1,1])
ax.set_ylim([-1,1])

for loc, spine in ax.spines.items():
    if loc in ['top', 'right']:
        spine.set_visible(False)
    if loc in ['left','bottom']:
        spine.set_position(('data',0))
```

<img src="/assets/image/matplotlib-pyplot-03_files/matplotlib-pyplot-03_16_0.png">

## 4. grid

grid형태의 line을 그려주는 method이다.

**[Axes.grid(which='major',axis='both',\*\*kwargs)](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.grid.html)**

-   which: {'major', 'minor', 'both'}
-   axis: {'both', 'x', 'y'}
-   \*\*kawargs: Line2D properties, line에 적용될 수 있는 property를 적용할 수 있다.
    -   alpha: 투명도
    -   color: 색상
    -   linestyle: ('-', 일직선), ('-.',일직선과 점이 번갈아 나타남), (':', 점선)
    -   linewidth: line의 width 설정

```python
fig, axes = plt.subplots(2,1,figsize=(7,7))

ticks = np.arange(0,21,5)

for ax in axes.flat:
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

axes[0].grid(axis='x')
axes[1].grid(axis='y')
```

<img src="/assets/image/matplotlib-pyplot-03_files/matplotlib-pyplot-03_18_0.png">

```python
fig, ax = plt.subplots(figsize=(7,7))

major_ticks = np.arange(0,21,5)
minor_ticks = np.arange(21)

ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks,minor=True)

ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks,minor=True)

ax.tick_params(which='major',length=5,width=2,
               labelsize=10,labelcolor='b')
ax.tick_params(which='minor',length=3)

ax.grid(which='major', linewidth=2)
ax.grid(which='minor', linewidth=1,alpha=0.5)
```

<img src="/assets/image/matplotlib-pyplot-03_files/matplotlib-pyplot-03_19_0.png">
