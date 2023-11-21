# %%
import numpy as np
import matplotlib.pyplot as plt
# %%
# pixel activation curve
xMax = 5
a = np.log(0.5)
x = np.linspace(0, xMax)
y = 1 - np.exp(x * a)
plt.vlines(np.arange(xMax + 1), ymin = 0, ymax=1, color='red', linestyles='dashed')
plt.plot(x, y)
plt.plot([0, xMax], [0.5, 0.5])
plt.xlabel('Coverage counts')
plt.ylabel('Quality recovery')
plt.title(f'Quality-Coverage: y= 1 - exp^(ax), a < 0')
plt.show()
# %% scalar average
from Common import *
x = np.random.randn(10)
u = [0]
v = [0]
for i in range(1, 10):
    u.append( scalarMeanUpdate(x[i-1], u[i-1], i) )
    v.append( scalarVarUpdate(v[i-1], u[i], u[i-1], i) )
    print(np.mean(x[:i]), np.var(x[:i]))
    print(u[-1], v[-1])
    print('=======')
# u, s = u[1:], s[1:]
for i in range(0, 10):
    for j in range(i + 1, 10):
        print(f'================={i}-{j}=================')
        u_ij = scalarMeanMN(u, i, j)
        s_ij = scalarVarMN(v, u, i, j)
        assert np.abs(np.mean(x[i:j]) - u_ij) < 1e-3 and np.abs(np.var(x[i:j]) - s_ij) < 1e-3
        print(f'{np.mean(x[i:j])} == {u_ij}')
        print(f'{np.var(x[i:j])} == {s_ij}')
#%%
# sclar MN
from Common import *
x = np.random.randn(100)
uu = np.zeros((x.shape[0]+1,))
vv = np.zeros((x.shape[0]+1,))
u = np.zeros((x.shape[0]+1,))
v = np.zeros((x.shape[0]+1,))
for i in range(1, x.shape[0] + 1):
    u[i] = scalarMeanUpdate(x[i - 1], u[i - 1], i)
    v[i] = scalarVarUpdate(v[i-1], u[i], u[i-1], i)
    uu[i] = np.mean(x[:i])
    vv[i] = np.var(x[:i])
print(np.allclose(u, uu), np.allclose(v, vv))
umn = np.zeros((x.shape[0]+1, x.shape[0]+1))
vmn = np.zeros((x.shape[0]+1, x.shape[0]+1))
uumn = np.zeros((x.shape[0]+1, x.shape[0]+1))
vvmn = np.zeros((x.shape[0]+1, x.shape[0]+1))
for i in range(umn.shape[0]):
    for j in range(i + 1, umn.shape[1]):
        umn[i, j] = scalarMeanMN(u, i, j)
        uumn[i, j] = np.mean(x[i:j])
        vmn[i, j] = scalarVarMN(v, u, i, j)
        vvmn[i, j] = np.var(x[i:j])
print(np.allclose(umn, uumn), np.allclose(vmn, vvmn))
#%% angle average
from Common import *
# print(np.random.randint(-1, 2, size=10) * 2 * np.pi)
a = np.random.randn(100)
# x = np.random.randn(100) + np.random.randint(-1, 2, size=100) * 2 * np.pi
# x = np.random.randn(100) + 2 * np.pi
eju = [1]
ejv = [1]
means = []
logMeans = []
for i in range(1, 10):
    print(i)
    eju.append(angularMeanUpdate(a[i-1], eju[-1], i))
    ejv.append(angularVarUpdate(ejv[-1], eju[-1], eju[-2], i))
    print('my:', eju[-1], ejv[-1])
    print('ans:', np.exp(1j * np.mean(a[:i])), np.exp(1j * np.var(a[:i])))
    
    m = np.mean(a[:i])
    while m < -np.pi:
        m += 2*np.pi
    while m >= np.pi:
        m -= 2*np.pi
    means.append(m)
    logMeans.append(np.angle(eju[-1]))
    print('=======')
# print(means, logMeans)
print(np.allclose(means, logMeans))
m, n = 2, 5
u_mn = angularMeanMN(eju, m, n)
s_mn = angularVarMN(ejv, eju, m, n)
# while st < -np.pi:
#     st += 2*np.pi
# while st >= np.pi:
#     st -= 2*np.pi
print('my:', u_mn, s_mn)
print('ans:', np.exp(1j * np.mean(a[m:n])), np.exp(1j * np.var(a[m:n])))
# %%
# Angular MN
from Common import *
x = np.random.randn(100)
uu = np.ones((x.shape[0]+1,), dtype=np.complex128)
vv = np.ones((x.shape[0]+1,), dtype=np.complex128)
u = np.ones((x.shape[0]+1,), dtype=np.complex128)
v = np.ones((x.shape[0]+1,), dtype=np.complex128)
for i in range(1, x.shape[0] + 1):
    u[i] = angularMeanUpdate(x[i - 1], u[i - 1], i)
    v[i] = angularVarUpdate(v[i-1], u[i], u[i-1], i)
    uu[i] = np.mean(x[:i])
    vv[i] = np.var(x[:i])
print(np.allclose(u, uu), np.allclose(v, vv))
umn = np.zeros((x.shape[0]+1, x.shape[0]+1), dtype=np.complex128)
vmn = np.zeros((x.shape[0]+1, x.shape[0]+1), dtype=np.complex128)
uumn = np.zeros((x.shape[0]+1, x.shape[0]+1), dtype=np.complex128)
vvmn = np.zeros((x.shape[0]+1, x.shape[0]+1), dtype=np.complex128)
for i in range(umn.shape[0]):
    for j in range(i + 1, umn.shape[1]):
        umn[i, j] = angularMeanMN(u, i, j)
        uumn[i, j] = np.mean(x[i:j])
        vmn[i, j] = angularVarMN(v, u, i, j)
        vvmn[i, j] = np.var(x[i:j])
print(np.allclose(umn, uumn), np.allclose(vmn, vvmn))
# %%
# pesimistic prob 
m, h, P = 0.1, 0.5, 10
# m*P >= 1
k = np.linspace(1 + 1e-12, h/m, 20000)
for P in [25, 50, 100, 200, 500, 1000]:
# for P in [1000]:
    f = ((m * (k-1) + (1/P)) ** (m*P)) * ((( 1/k + ((1-1/k) * (h - m*k)) / (m*(k-1)) )) ** (m*P))
    plt.plot(k, f[f != np.nan]/np.max(f[f != np.nan]), label=f'P={P}')
    # plt.vlines([
    #     # (m - 1/P)/m,
    #     # (P*(h + m)*(P*m - 1)) ** ((1/2)/(P*m)),
        
    #     # (P*(h + m)*(P*m - 1)) ** (1/2)/(P*m),
    #     np.sqrt((m+h)*P/m),
        
    #     # -(P*(h + m)*(P*m - 1)) ** (1/2)/(P*m)
    # ], ymin=0, ymax=1, color='r', linestyles='dashed')
    plt.vlines([
        # (m - 1/P)/m,
        # (P*(h + m)*(P*m - 1)) ** ((1/2)/(P*m)),
        np.sqrt((m+h)/m),
        # -(P*(h + m)*(P*m - 1)) ** (1/2)/(P*m)
    ], ymin=0, ymax=1, color='g', linestyles='dashed')
    print((P*(h + m)*(P*m - 1)) ** (1/2)/(P*m), np.sqrt((m+h)/m))
# plt.vlines([(m-2)*h/(m**2), (m-1)*h/(m**2), h/m], ymin=0, ymax=1, color='r')
# plt.vlines([h/m], ymin=0, ymax=1, color='r')
plt.legend()
plt.show()
# print(np.max(f))
# print(k[np.argmax(f1 * f2)])
#%%
# Outside decay, inside hold
k = 1
d = 10
x = np.linspace(0.1, 10*d, 20001)
y1 = d/(x)
y2 = 1/(1 + np.exp(-2 * k * (-x + d)))
# d = np.arange(10, 50, 10)
# plt.plot(x, y1 * (1 - y2))
plt.plot(x, y1 * (1 - y2) + 1 * y2)
# plt.vlines(d, ymin=np.min(y), ymax=np.max(y), color='red')
plt.show()
#%%
# Double-sided windowing
k = 10
d = np.sqrt(3)/2
x = np.linspace(-1, 1, 20001)
y1 = 1/(1 + np.exp(-2 * k * (x + (d))))
y2 = 1/(1 + np.exp(-2 * k * (-x + d)))
y = (y1 + y2) / 2
plt.plot(x, y)
# plt.vlines([-d, d], ymin=np.min(y), ymax=np.max(y), color='red')
plt.show()

# target = np.pi/8
# x = np.linspace(-np.pi/2, np.pi/2, 20000)
# sig = 0.4
# 1/(sig * np.sqrt(2*np.pi))
# y = np.exp(((np.sin(x))/sig)**2 * (-1/2))
# plt.plot(np.sin(x), y)
# plt.vlines(np.sin([np.pi/4, -np.pi/4]), ymin=np.min(y), ymax=np.max(y), color='red')
# plt.show()
#%%
# inner product windowing
import torch
x = torch.linspace(-0.4, 1, 20001)
y = torch.log(x + (1 - np.cos(np.pi/4)))
print(np.log(np.cos(np.pi/4) + (1 - np.cos(np.pi/4))))
plt.plot(x, y, label='4')

y = torch.log(x + (1 - np.cos(np.pi/8)))
print(np.log(np.cos(np.pi/4) + (1 - np.cos(np.pi/8))))
plt.plot(x, y, label='8')
plt.vlines([np.cos(np.pi/4)], ymin=-1, ymax=1, color='red')
plt.plot([x[0], x[-1]], [0, 0])
plt.legend()
plt.show()
#%%
import torch
x = torch.linspace(-1, 1, 20001)
k = 50
y = 1/(1 + torch.exp(-2 * k * (x-np.cos(torch.pi/4))))
plt.plot(x, y, label='4')
plt.show()
#%%
import torch
x = torch.tensor([1, 2, 3], dtype=torch.float)
x.requires_grad_()
y = torch.Tensor([x[2], x[1]]) ** 2
yy = torch.sum(y)
yy.backward()
print(x.grad)
#%%
# Double-sided windowing
import torch
def win(x, k):
    return 1/(1 + torch.exp(-2 * k * x))
def innerWindow(x, k=0.01, fov=np.pi/2):
    return ( win(x + torch.cos(fov/2), k) + win(-x + torch.cos(fov/2), k)) / 2
# print(torch.inner(torch.zeros((4, 3)), torch.zeros((5, 3))).shape)
x = torch.linspace(-1, 1, 2000, requires_grad=False)
p0 = torch.Tensor([0, 0, 0])
d0 = torch.Tensor([0, 0, 1])
dnp = np.array([
    [1, 2, 2],
    [1, 1, 2],
    [1, 3, 2],
    [2, 4, 2]
])
dNorm = dnp / np.sqrt(np.sum(dnp ** 2, axis=1)).reshape((-1, 1))
print(dNorm.shape)

dNorm = torch.from_numpy(dNorm)
d = torch.from_numpy(dnp)
k = torch.Tensor([1])
k.requires_grad_(True)
print(y)
optimizer = torch.optim.SGD([k], lr=1e-2)
for i in range(100):
    optimizer.zero_grad()
    dis = d - (p0 + k*d0).reshape((-1, 3))
    gain = torch.sum(dis * d0, axis=1)
    # print(gain.shape)
    penalty = torch.sum(dis * dis, axis=1)
    y = -gain/penalty
    y = torch.sum(y)
    y.backward(create_graph=True, retain_graph=True)    
    optimizer.step()
    print(k.detach().item())
    print(y.detach().item())