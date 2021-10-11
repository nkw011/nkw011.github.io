---
title: "[Python] Class(클래스) 구조"
excerpt: "Python Class(클래스) 구조 정리"
toc: true
toc_sticky: true
categories:
    - python
tags:
    - python
sidebar:
    nav: sidebarTotal
---

## 1. Class 구조와 생성자

Class 구조를 사용하면 재사용성, 코드 반복 최소화, 메소드 활용 등의 장점이 있다.

### 1.1. Class 구조

```
# 클래스가 상속받지 않는 경우
class Name:
    ...

# 클래스가 상속받는 경우
class Name(상속받는 클래스):
    ...
```

정의된 Class를 활용해 객체를 만드는 것은 다음과 같다.

```
# 필요한 parameter가 존재하지 않는경우
variable = Name()

# 필요한 parameter가 존재하는 경우
variable = Name(필요한 parameter)
```

Class 이름과 생성할 때 필요한 parameter를 채워 변수에 할당하면 된다.
이 때 필요한 parameter는 Class에서 정의된 생성자에 나와있다.

### 1.2. 생성자(Constructor)

생성자는 Class를 활용해 객체를 생성할 때 사용된다.

```py
class Name:
    def __init__(self):

# 위와 같은 생성자에 self 이외에 parameter가 없는 경우
a = Name()

class Name:
    def __init__(self,param1):
        self._param1 = param1

# 생성자에 paramter가 존재하는 경우
a = Name(a)
```

parameter에 self만 있는 경우는 객체 생성시 paramter가 따로 필요하지 않다는 뜻이고
self이외에 다른 parameter가 존재하는 경우 객체 생성시 paramter를 꼭 적어주어야한다는 뜻이다.

self는 class내에서 생성된 객체 스스로를 가리킬 때 쓰는 것으로 생성자에 필요한 parameter와는 관련이 없으므로 일단은 모른체로 넘겨두도록 하겠다.

생성자에 아무런 코드가 없는 경우 생성자를 만들지 않아도 객체를 생성할 수 있다.
다음과 같은 코드가 있다고 하자.

```py
class Name:
    pass

a = Name()
```

Name Class안에 생성자가 없는데도 a에는 Name을 활용해 만들어진 객체가 할당이 된다.
그 이유는 다음과 같다.
class안에 생성자를 명시해주지 않으면 자동적으로 아무런 parameter도 코드 블록도 없는 default constuctor가 생성이 되기 때문이다.
따라서 생성자를 명시해주지 않아도 class를 활용해 객체를 만들 수 있다.

## 2. 클래스 변수와 인스턴스 변수

변수의 종류에 대해 배울 것인데 Python 클래스 구조에서 변수는 크게 2가지가 존재한다.
바로 클래스 변수와 인스턴스 변수이다.
클래스 변수는 클래스가 공통으로 사용하는 변수이기 때문에 클래스를 이용해 생성된 객체가 모두 공통된 값을 지닌다.
하지만 인스턴스 변수는 객체마다 다른 값을 지닌다.

### 2.1. 인스턴스 변수

객체마다 필요한 속성값을 넣어줄 때 필요한 변수라고 생각을 하면 편하다.
다음과 같은 내용이 들어간 학생기록부를 만든다고 가정하자.

```
학생기록부
1. 이름
2. 나이
3. 혈액형
4. 점수
```

학생기록부라는 클래스를 만들기 위해서 이름, 나이, 혈액형, 점수와 같은 속성값이 필요하다.
이 때 이름, 나이, 혈액형, 점수는 학생마다 다르므로 인스턴스 변수를 이용해 만들어줄 것이다.
인스턴스 변수는 생성자 안에 self라는 키워드를 이용해 만들 수 있다.

```py
class Student:
    def __init__(self,name,age,blood,score):
        self._name = name
        self._age = age
        self._blood = blood
        self._score = score
```

이름, 나이, 혈액형, 점수는 모두 학생마다 다른 값을 가지고 있기 때문에 객체를 만들 때 입력하도록 생성자에 parameter를 추가하였다.
그리고 각각 객체 안에서 사용이 될 수 있도록 인스턴스 변수에 할당을 하였다.

```
self keyword에 대해서

self는 객체 자신을 가리키는 키워드라고 생각하면 된다.
클래스 내부에서 객체마다 다르게 동작하는 인스턴스 변수를 만들기 위해서 self라는 키워드를 사용하고 또 인스턴스변수를 사용하기 위해서는
메소드 첫 parameter에 self를 명시해주어야한다.

생성자는 인스턴스 변수를 생성하기 때문에 self를 parameter에 명시해주었다.
```

### 2.2 클래스 변수

학생기록부에 총 학생 수를 기록해야한다고 하자.
총 학생 수는 학생마다 다르지 않고 모두 공통된 값을 지니기 때문에 클래스 변수를 이용해서 만든다.
클래스 변수는 클래스 내부에 별다른 키워드를 사용하지 않고 변수를 만드는 것처럼 만들어주면 된다.

```py
class Student:

    # 총 학생수를 나타내기 위해 쓰이는 클래스 변수 명시
    total = 0

    def __init__(self,name,age,blood,score):
        self._name = name
        self._age = age
        self._blood = blood
        self._score = score
        Student.total += 1
```

객체를 생성할 때마다 학생 수가 하나씩 증가하도록 생성자 안에 코드를 추가하였다.
클래스 변수를 사용하기 위해서는 클래스 내에서는 `클래스이름.클래스변수` 사용한다.
클래스 밖에서는 `클래스이름.클래스변수`, `객체이름.클래스변수`로 사용해주어야한다.

클래스 변수와 인스턴스 변수는 동일한 이름을 사용해도 된다.
이 때 객체로 접근하면 인스턴스 변수가 출력이 되고 클래스이름을 사용해 접근하면 클래스 변수로 접근이 된다.

## 3. `__str__`, `__doc__`, `__dict__`, `dir()`

### 3.1. `__str__`

Student 클래스를 활용해 만든 객체를 print()를 활용해 출력을 해보자

```py
class Student:

    total = 0

    def __init__(self,name,age,blood,score):
        self._name = name
        self._age = age
        self._blood = blood
        self._score = score
        Student.total += 1

s1 = Student("Kim",18,'A',100)

print(s1)
```

```
결과 값

<__main__.Student object at 0x7fb3cad80b50>
```

각 객체를 가리키는 고유한 주소값인 id가 나오게 된다.
객체를 print를 이용해 출력할 때 주소값이 아닌 이름, 나이, 혈액형, 점수같은 정보를 출력할려면
미리 내장된 함수인 `__str__`을 이용해주어야한다.

```py
class Student:

    total = 0

    def __init__(self,name,age,blood,score):
        self._name = name
        self._age = age
        self._blood = blood
        self._score = score
        Student.total += 1

    def __str__(self):
        return "Student {} - age:{} blood type: {} score: {}".format(self._name,self._age,self._blood,self._score)


s1 = Student("Kim",18,'A',100)
print(s1)
```

```
결과 값

Student Kim - age:18 blood type: 100 score: 100
```

### 3.2. `__doc__`

클래스 내부에는 클래스를 설명하기 위해 주석을 별도로 만들어 줄 수 있다.

```py
class Student:
    '''
    Student class
    Author: Kum
    '''
    total = 0

    def __init__(self,name,age,blood,score):
        self._name = name
        self._age = age
        self._blood = blood
        self._score = score
        Student.total += 1

    def __str__(self):
        return "Student {} - age:{} blood type: {} score: {}".format(self._name,self._age,self._blood,self._score)


s1 = Student("Kim",18,'A',100)
print(s1.__doc__)
```

```
결과 값

    Student class
    Author: Kum
```

클래스 내부에 docstring을 이용해 주석을 달게 되면 `__doc__`를 이용해 출력할 수 있다.

### 3.3. `__dict__`

객체가 사용하는 속성 값을 dictionary로 바꾸어서 보여준다.

```py
class Student:
    '''
    Student class
    Author: Kum
    '''
    total = 0

    def __init__(self,name,age,blood,score):
        self._name = name
        self._age = age
        self._blood = blood
        self._score = score
        Student.total += 1

    def __str__(self):
        return "Student {} - age:{} blood type: {} score: {}".format(self._name,self._age,self._blood,self._score)


s1 = Student("Kim",18,'A',100)
print(s1.__dict__)
```

```
결과 값

{'_name': 'Kim', '_age': 18, '_blood': 100, '_score': 100}
```

### 3.4. `dir()`

`dir()`은 parameter로 객체를 받는다.
해당 객체가 어떤 변수와 메소드를 가지고 있는 지 나열해준다.

```py
class Student:
    '''
    Student class
    Author: Kum
    '''
    total = 0

    def __init__(self,name,age,blood,score):
        self._name = name
        self._age = age
        self._blood = blood
        self._score = score
        Student.total += 1

    def __str__(self):
        return "Student {} - age:{} blood type: {} score: {}".format(self._name,self._age,self._blood,self._score)


s1 = Student("Kim",18,'A',100)
print(dir(s1))
```

```
결과 값

['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_age', '_blood', '_name', '_score', 'total']
```

## 4. instance method, class method, static method

클래스 메소드는 3가지 종류가 있다.
self 키워드를 사용하는 instance method, cls 키워드를 사용하는 class method, 아무런 키워드를 사용하지 않는 static method이다.

### 4.1. instance method

인스턴스 변수를 사용하기 위해 parameter에 self를 사용하는 method를 instance method라고 한다.
instance method는 객체 이름만 이용해 사용할 수 있다.

```py
class Student:
    '''
    Student class
    Author: Kum
    '''
    total = 0

    def __init__(self,name,age,blood,score):
        self._name = name
        self._age = age
        self._blood = blood
        self._score = score
        Student.total += 1

    def __str__(self):
        return "Student {} - age:{} blood type: {} score: {}".format(self._name,self._age,self._blood,self._score)

    def getScore(self):
        return self._score


s1 = Student("Kim",18,'A',100)
print(s1.getScore())
```

```
결과 값

100
```

학생 점수를 반환하는 메소드인 `getScore` 작성하였다.
학생 점수는 인스턴스 변수이기 때문에 parameter에 self를 사용하였다.

### 4.2. class method

클래스 변수를 사용하는 메소드이다.
`@classmethod`를 메소드 위에 명시해주고 parameter에 cls를 추가해주어야한다.
class method는 클래스 이름, 객체를 이용해 모두 접근할 수 있다.

```py
class Student:
    '''
    Student class
    Author: Kum
    '''
    total = 0

    def __init__(self,name,age,blood,score):
        self._name = name
        self._age = age
        self._blood = blood
        self._score = score
        Student.total += 1

    def __str__(self):
        return "Student {} - age:{} blood type: {} score: {}".format(self._name,self._age,self._blood,self._score)

    def getScore(self):
        return self._score

    @classmethod
    def getTotalNumber(cls):
        return cls.total


s1 = Student("Kim",18,'A',100)
s2 = Student("Park",19,'AB',80)
print(Student.getTotalNumber())
```

```
결과 값

2
```

### 4.3. static method

static method를 사용하기 위해서는 `@staticmethod`를 메소드 위에 명시해주어야한다.
static method는 class method와 같이 클래스 이름, 객체를 이용해 모두 사용할 수 있다.

```py
class Student:
    '''
    Student class
    Author: Kum
    '''
    total = 0

    def __init__(self,name,age,blood,score):
        self._name = name
        self._age = age
        self._blood = blood
        self._score = score
        Student.total += 1

    def __str__(self):
        return "Student {} - age:{} blood type: {} score: {}".format(self._name,self._age,self._blood,self._score)

    def getScore(self):
        return self._score

    @classmethod
    def getTotalNumber(cls):
        return cls.total

    @staticmethod
    def isBloodTypeA(stu):
        return stu._blood == 'A'


s1 = Student("Kim",18,'A',100)
s2 = Student("Park",19,'AB',80)


print(Student.isBloodTypeA(s1))
print(Student.isBloodTypeA(s2))

print(s1.isBloodTypeA(s1))
print(s1.isBloodTypeA(s2))
```

```
결과 값

True
False
True
False
```
