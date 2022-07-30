---
title: "Colab에 torchdata 설치하기 & pip's depedency 관련 ERROR 해결방법"
excerpt: "colab에 torchdata 설치하는 방법과 설치할 때 발생하는 ERROR 해결하기"
use_math: true
toc: true
toc_sticky: true
categories:
    - nlp
tags:
    - colab
    - torchdata
sidebar:
    nav: sidebarTotal
---

torchtext의 데이터셋을 불러오려면 먼저 torchdata를 설치해야합니다.
* 호환성의 문제가 있을 수 있으므로 PyTorch의 version에 적합한 version을 설치해야합니다.
* [torchdata Version 호환성 확인하기](https://github.com/pytorch/data#version-compatibility)

torchdata를 설치할 때 아래와 같은 ERROR가 발생할 수 있습니다.
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
datascience 0.10.6 requires folium==0.2.1, but you have folium 0.8.3 which is incompatible.
```
호환성 관련 ERROR 표시로 folium 0.2.1. 버전을 설치하면 해결할 수 있습니다. 이 때 호환성 관련 ERROR 표시가 나오더라도 torchdata가 설치될 수 있으니 삭제한 후 재설치하면 됩니다.

* folium==0.2.1을 먼저 설치한 다음 torchdata를 재설치합니다.
    * 설치된 경우 torchdata를 삭제한 후 folium부터 다시 설치합니다.
* colab을 사용하고 있는 경우 런타임을 재실행한 다음 순서대로 설치하면 됩니다.
