import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def predict(X, theta):
    return X*theta[1] + theta[0]

def fit(X, y, theta, alpha, num_iters):
    m = len(X)
    for _ in range(num_iters):
        error = predict(X, theta) - y
        theta[0] = theta[0] - (alpha/m) * np.sum(error)
        theta[1] = theta[1] - (alpha/m) * np.dot(error, X)
    return theta

def visualizeRegression(X, y, theta):
    ax = plt.axes()
    ax.set_xlim([0,25])
    ax.set_ylim([0,25])
    ax.scatter(X, y)
    line_x = np.linspace(0,22.5, 20)
    line_y = theta[0] + line_x * theta[1]
    ax.plot(line_x, line_y)
    plt.title('Function de regression', fontsize=18, fontweight='bold')
    plt.xlabel('population', fontsize=14, fontweight='bold')
    plt.ylabel('profit', fontsize=14, fontweight='bold')
    ax.grid(color='black', linestyle='-', linewidth=0.2)
    plt.show()


def cost(X, y, theta):
    m = len(X)
    error = predict(X, theta) - y
    cost = (1/(2*m))*(np.sum(error**2))
    return cost

def fit_with_cost(X, y, theta, alpha, num_iters):
    m = len(X)
    J_history = []
    for _ in range(num_iters):
        error = predict(X, theta) - y
        theta[0] = theta[0] - (alpha/m) * np.sum(error)
        theta[1] = theta[1] - (alpha/m) * np.dot(error, X)
        J_history.append(cost(X, y, theta))
    return theta, J_history

def visualizeCost (J_history) :
    ax = plt.axes()
    plt.title('Cost evolution', fontsize=18, fontweight='bold')
    plt.xlabel('Iteration', fontsize=14, fontweight='bold')
    plt.ylabel('Cout', fontsize=14, fontweight='bold')
    ax.plot(J_history)
    plt.show()

if __name__ == '__main__':
    data = pd.read_csv("ex1data1.csv")
    data.plot.scatter('population', 'profit')
    X = np.array(data['population'])
    y = np.array(data['profit'])
    theta = np.zeros(2)
    z = X*theta[1] + theta[0]

    theta = np.zeros(2)
    #theta = fit(X, y, theta, 0.02, 1000)
    theta, J_history = fit_with_cost(X, y, theta, 0.02, 1000)
    print(theta)
    visualizeRegression(X, y, theta)
    visualizeCost(J_history)

