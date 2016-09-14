
Weighted Mean with 1 decimal place


```python
n = int(raw_input())
X = raw_input()
W = raw_input()
X = X.split(' ')
W = W.split(' ')
elements = []
weights = []
for i in range(n):
    elements.append(int(X[i]))
    weights.append(int(W[i]))
sums = 0.0
for elem, weigh in zip(elements, weights):
    sums+=elem*weigh
print(("%.1f")%(sums/sum(weights)))
```

    10
    10 40 30 50 20 10 40 30 50 20
    1 2 3 4 5 6 7 8 9 10
    31.1


Standard Deviation (Can use np.std(a))


```python
from math import sqrt
N = input()
X = map(float,raw_input().split())
mean = sum(X)/N
print round(sqrt(sum([(x-mean)**2 for x in X])/N),1)
```

    3
    1 2 3
    0.8


Quantiles


```python
from math import ceil
N = int(raw_input())
X = map(int,raw_input().split())
X = sorted(X)
if not N%2==0:
    index = int(round(N/2))
    q2 = X[index]
    first = X[:index]
    second = X[index+1:]
    
else:
    half = int(len(X)/2)
    q2 = (X[half-1]+X[half])/2
    first = X[:half]
    second = X[half:]

if int(len(first)%2)==0:
    half = int(len(first)/2)
    q1 = (first[half-1]+first[half])/2
else:
    q1 = first[int(round(len(first)/2))]
    
if int(len(second)%2)==0:
    half = int(len(second)/2)
    q3 = (second[half-1]+second[half])/2
else:
    q3 = second[int(round(len(second)/2))]
    
print q1
print q2
print q3
```

    10
    3 7 8 5 12 14 21 15 18 14
    7
    13
    15


Binomial Distribution


```python
import math

def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

prob_boy = 1.09/(1.09+1)
sums = 0
for i in range(3,7):
    sums+= nCr(6,i)*(prob_boy)**i*(1-prob_boy)**(6-i)
print round(sums,3)
6
6 12 8 10 20 16
5 4 3 2 1 5
```

    0.696


Interquartile Range


```python
M = int(raw_input())
X = map(int,raw_input().split())
F = map(int,raw_input().split())
final = []
for i, f in zip(X, F):
    while f>0:
        final.append(i)
        f = f-1
final = sorted(final)
N = len(final)
if not N%2==0:
    index = int(round(N/2))
    q2 = final[index]
    first = final[:index]
    second = final[index+1:]
    
else:
    half = int(len(final)/2)
    q2 = (final[half-1]+final[half])/2.0
    first = final[:half]
    second = final[half:]

if int(len(first)%2)==0:
    half = int(len(first)/2)
    q1 = (first[half-1]+first[half])/2.0
else:
    q1 = first[int(round(len(first)/2))]
    q1 = float(q1)
    
if int(len(second)%2)==0:
    half = int(len(second)/2)
    q3 = (second[half-1]+second[half])/2.0
else:
    q3 = second[int(round(len(second)/2))]
    q3 = float(q3)

print round(q3-q1,1)
```

    30
    10 40 30 50 20 10 40 30 50 20 1 2 3 4 5 6 7 8 9 10 20 10 40 30 50 20 10 40 30 50
    1 2 3 4 5 6 7 8 9 10 1 2 3 4 5 6 7 8 9 10 10 40 30 50 20 10 40 30 50 20
    30.0


Pearson's Correlation
10
10 9.8 8 7.8 7.7 7 6 5 4 2
200 44 32 24 22 17 15 12 8 4


```python
from math import sqrt
N = int(raw_input())
X = map(float,raw_input().split())
Y = map(float,raw_input().split())

mean_x = sum(X)/len(X)
mean_y = sum(Y)/len(Y)

std_x = sqrt(sum([(x-mean_x)**2 for x in X])/N)
std_y = sqrt(sum([(y-mean_y)**2 for y in Y])/N)

num_x = 0.0
num_y = 0.0
num = 0.0
for x, y in zip(X, Y):
    num_x = x-mean_x
    num_y = y-mean_y
    num += num_x * num_y

cov = num / (std_x * std_y * N)

print round(cov, 3)

```

    10
    10 9.8 8 7.8 7.7 7 6 5 4 2
    200 44 32 24 22 17 15 12 8 4
    530.394
    19.034
    -7.366
    -14.766
    -15.326
    -5.616
    16.644
    44.634
    81.354
    159.874
    808.86
    0.612


Spearman's Rank Correlation
10
10 9.8 8 7.8 7.7 1.7 6 5 1.4 2 
200 44 32 24 22 17 15 12 8 4


```python
import copy
N = int(raw_input())
X = map(float,raw_input().split())
Y = map(float,raw_input().split())

temp_x = list(X)
temp_y = list(Y)

temp_x.sort(reverse = True)
temp_y.sort(reverse = True)

rankval_x = {}
rankval_y = {}
vals = copy.copy(N)
for x,y in zip(temp_x, temp_y):
    rankval_x[x] = N
    rankval_y[y] = N
    N = N-1

rx = 0
ry = 0
d = 0.0

for x, y in zip(X, Y):
    if x in rankval_x:
        rx = rankval_x[x]
    if y in rankval_y:
        ry = rankval_y[y]
    d += (rx-ry)**2
    
rxy = 1-((6*d)/(vals*((vals**2)-1)))

print round(rxy,3)
```

    10
    10 9.8 8 7.8 7.7 1.7 6 5 1.4 2 
    200 44 32 24 22 17 15 12 8 4



```python
from sklearn import linear_model
import numpy as np
math = []
stats = []
for x in range(5):
    N = map(int, raw_input().split())
    math.append(N[0])
    stats.append(N[1])
x = np.asarray(math).reshape(-1, 1)
lm = linear_model.LinearRegression()
lm.fit(x, stats)
print(lm.intercept_)
print(lm.coef_[0]*80+lm.intercept_)
```

    95 85
    85 95
    80 70
    70 65
    60 70
    26.7808219178
    0.643835616438


    /Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/scipy/linalg/basic.py:884: RuntimeWarning: internal gelsd driver lwork query error, required iwork dimension not returned. This is likely the result of LAPACK bug 0038, fixed in LAPACK 3.2.2 (released July 21, 2010). Falling back to 'gelss' driver.
      warnings.warn(mesg, RuntimeWarning)


Multiple Linear Regression:

0.18 0.89 109.85
1.0 0.26 155.72
0.92 0.11 137.66
0.07 0.37 76.17
0.85 0.16 139.75
0.99 0.41 162.6
0.87 0.47 151.77


```python
import numpy as np
from numpy.linalg import inv
m = map(int, raw_input().split())
n = m[1]
m = m[0]
X = []
Y = []
for x in range (n):
    X.append(map(float, raw_input().split()))

    
t = int(raw_input())

for i in range(t):
    Y.append(map(float, raw_input().split()))

x = np.zeros((n,m+1))
x[:,[0]] = 1
    
ctr1 = 0
y = np.zeros((n,1))
for i in X:
    x[ctr1][1] = i[0]
    x[ctr1][2] = i[1]
    y[ctr1][0] = i[2]
    ctr1 += 1

temp = np.dot(x.T, x)
temp1 = inv(temp)
temp = np.dot(temp1, x.T)
pred = np.dot(temp, y)

for i in range(t):
    print round(pred[0]+Y[i][0]*pred[1]+Y[i][1]*pred[2],4)
```
