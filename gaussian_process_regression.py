import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import math
import random

#X = [[3], [7], [15]]
#X = np.array(X)

#y = [[4], [7], [3]]
#y = np.array(y)


def squared_exponential(xi, xj):
    diff = np.subtract(xi, xj)
    # print(diff)
    ndiff = np.linalg.norm(diff)
    # print(ndiff)
    kij = np.exp(-0.5*math.pow(ndiff, 2))
    return kij


def kernel(X, kernel_function):
    cov = np.zeros((len(X), len(X)))
    for i in range(len(cov)):
        for j in range(len(cov[i])):
            cov[i, j] = kernel_function(X[i], X[j])
    return cov


def inference_kernel(x_new, X, kernel_function):
    k_star = np.zeros((len(X), 1))
    for i in range(len(k_star)):
        k_star[i, 0] = kernel_function(x_new, X[i])
    return k_star


def inference(x_new, X, y, kernel_function, initial_mean=0, x_range=1):
    K = kernel(X, kernel_function)
    # print(K.shape)
    K_inv = np.linalg.inv(K)
    # print(K_inv)
    # print(K_inv.shape)
    k_star = inference_kernel(x_new, X, kernel_function)

    k_star_star = kernel_function(x_new, x_new)
    # print(k_star.shape)
    k_star_t = np.transpose(k_star)
    # print(k_star_t.shape)
    mu = initial_mean + np.matmul(k_star_t,
                                  np.matmul(K_inv,
                                            (y-initial_mean)))
    sigma = np.subtract(1, np.matmul(
        k_star_t, np.matmul(K_inv, k_star)))*x_range
    return mu, sigma


def plot_2d_inference(X, y, kernel_function, xmin, xmax,
                      initial_mean=0, x_range=1, step=50):
    var1 = np.linspace(xmin, xmax, num=step)
    var2 = np.linspace(xmin, xmax, num=step)
    X1, X2 = np.meshgrid(var1, var2)
    y_news = []
    y_minus_sigma = []
    y_plus_sigma = []
    ymoos = np.zeros((step, step))
    for xi in range(len(var1)):
        for xj in range(len(var2)):
            y_mu, y_sigma = inference(
                [X1[xi, xj], X2[xi, xj]], X, y, kernel_function, initial_mean, x_range)
            y_std = math.sqrt(y_sigma[0][0])
            # print(y_mu)
            y_minus_sigma.append(y_mu[0][0]-y_std)
            y_news.append(y_mu[0][0])
            y_plus_sigma.append(y_mu[0][0]+y_std)
            ymoos[xi, xj] = y_mu[0][0]
    Z = ymoos.reshape(X1.shape)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X1, X2, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)


def plot_1d_inference(X, y, kernel_function, xmin, xmax, initial_mean=0, x_range=1, step=50):
    x_news = np.linspace(xmin, xmax, num=step)
    y_news = []
    y_minus_sigma = []
    y_plus_sigma = []
    for x_new in x_news:
        y_mu, y_sigma = inference(
            x_new, X, y, kernel_function, initial_mean, x_range)
        y_std = math.sqrt(y_sigma[0][0])
        # print(y_mu)
        y_minus_sigma.append(y_mu[0][0]-y_std)
        y_news.append(y_mu[0][0])
        y_plus_sigma.append(y_mu[0][0]+y_std)
    plt.scatter(X, y)
    plt.plot(x_news, y_news)
    plt.fill_between(x_news, y_minus_sigma, y_plus_sigma, alpha=0.2)


#original_x = np.linspace(0, 4*math.pi, 100)
#original_y = np.sin(original_x)
#plt.plot(original_x, original_y)
'''
X = [[1,1], [0.5,0.7], [-0.7, -0.5]]
y = [[0.875], [0.527], [0.495]]

plot_2d_inference(np.array(X), np.array(y), squared_exponential, -1, 1, x_range=1, step=100)

'''
