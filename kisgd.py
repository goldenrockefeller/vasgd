import numpy as np

a = 0.251
b = 0.25

def h(x):
    if x > 0:
        return a * x
    return b * x

def dh(x):
    if x > 0:
        return a
    return b

def df(x, t):
    dhx = dh(x)
    return -dhx * (dhx * x - t)

def opt_scale(x, t):
    dht = dh(t)
    dhx = dh(x)
    return (dht * x - t)/(dhx * x - t) / dhx/dht

# Not good as a state estimator for nonlinear functions

x = 0.2
t = -1
df = a * (a*x - t)
e = a*x - t

u = 1.1
dg = b * (b*x - u)
e2 = b*x - u

top = 1 # a**2 + b**2 estimate
bottom = 1 # v estimate

for i in range(100):
    top = np.sqrt((df + dg)**2 / bottom ** 2)
    bottom = np.sqrt((e**2 + e2**2)/top)
    print(bottom, top)

print(f"a**2 + b**2 = {a**2 + b**2}")

