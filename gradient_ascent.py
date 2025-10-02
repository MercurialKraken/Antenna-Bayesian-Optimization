import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import copy
import random


def getGradients(x, function, h=1e-5):
    x = x.astype(np.float64)
    x_close = np.zeros(x.shape)
    y = function(x)
    # print(y)
    for i in range(len(x)):
        xph = copy.deepcopy(x)
        xph[i, 0] += h
        grad = (function(xph) - function(x))/h
        # print(grad)
        x_close[i, 0] = grad
    return x_close


def gradientAscent(function, xshape, learning_rate=0.1,
                   learning_rate_decay=0.95, min_x=-99999.,
                   max_x=99999., max_iterations=100, h=1e-5):
    x_star = np.zeros(xshape)
    x_star += [random.uniform(min_x, max_x)+i for i in x_star]
    # print(x_star)
    x_history = []
    y_history = []
    # print(x_star)
    for i in range(max_iterations):
        x_history.append(copy.deepcopy(x_star))
        y_history.append(function(x_star))
        grads = getGradients(x_star, function)
        step = learning_rate*grads
        x_star += step
        # print(x_star)
    plt.plot(np.linspace(0, len(y_history), len(y_history)),
             np.array(y_history).reshape((100,)))
    for i in range(len(x_star)):
        if x_star[i, 0] > max_x:
            x_star[i, 0] = max_x
    for i in range(len(x_star)):
        if x_star[i, 0] < min_x:
            x_star[i, 0] = min_x
    # print(x_history)
    return x_star, x_history, y_history


#maxima, _, _ = gradientAscent(function, xshape=(8, 1), learning_rate=0.1)
'''
maxi, xhis, yhis = gradientAscent(function, xshape=(
    2, 1), learning_rate=0.1, min_x=-99, max_x=99)
'''


'''var1 = np.linspace(-100, 100, num=100).reshape(100, 1)
var2 = np.linspace(-100, 100, num=100).reshape(100, 1)



X1, X2 = np.meshgrid(var1, var2)
res = [-1*(X1[i]**2)-X2[i]**2 for i in range(len(var1))]
Z = np.array(res).reshape(X1.shape)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X1, X2, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("Y")
#ax.view_init(45, 10)


X1, X2 = np.meshgrid(var1, var2)
res = [-1*(X1[i]**2)-X2[i]**2 for i in range(len(var1))]
Z = np.array(res).reshape(X1.shape)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
xhis = np.array([np.ravel(xh) for xh in xhis])
xxy = np.concatenate((xhis, np.array(yhis).reshape(100, 1)), axis=1)
ax.scatter3D(xxy[:,0], xxy[:,1], xxy[:,2], color='green')
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("Y")
plt.show()'''
