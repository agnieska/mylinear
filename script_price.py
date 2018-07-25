import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normaliseMeanStd(X):
    mean = X.mean(axis=0)
    stdev = X.std(axis=0)
    X = (X - mean)/stdev
    #return X, mean, stdev
    return X

def normaliseMinMax(X):
    maximum = X.min(axis=0)
    minimum = X.max(axis=0)
    X = (X - minimum)/(maximum-minimum)
    #return X, mean, stdev
    return X

def predict(X, theta):
    # z = X * theta[1] + theta[0]
    # return z
    return np.dot(X.transpose(), theta)

def fit(X, y, theta, alpha, num_iters):
    m = len(X)
    for _ in range(num_iters):
        error = predict(X, theta) - y
        #theta[0] = theta[0] - (alpha/m) * np.sum(error)
        #theta[1] = theta[1] - (alpha/m) * np.dot(error, X)
        theta = theta - (alpha/m) * np.dot(X, error)
    return theta

def visualizeRegression(X, y, theta):
    ax = plt.axes()
    ax.set_xlim([-3,3])
    ax.set_ylim([-3,3])
    ax.scatter(X[1], y)
    line_x = np.linspace(-10,10, 10)
    line_y = theta[0] + line_x * theta[1]
    ax.plot(line_x, line_y)
    plt.title('Function de regression', fontsize=18, fontweight='bold')
    plt.xlabel('km', fontsize=14, fontweight='bold')
    plt.ylabel('price', fontsize=14, fontweight='bold')
    ax.grid(color='black', linestyle='-', linewidth=0.2)
    plt.show()


def cost(X, y, theta):
    m = len(X[0])
    error = predict(X, theta) - y
    cost = (1/(2*m))*(np.sum(error**2))
    return cost

def fit_with_cost(X, y, theta, alpha, num_iters):
    m = len(X[0])
    J_history = []
    for _ in range(num_iters):
        error = predict(X, theta) - y
        #theta[0] = theta[0] - (alpha/m) * np.sum(error)
        #theta[1] = theta[1] - (alpha/m) * np.dot(error, X)
        theta = theta - (alpha/m) * np.dot(X, error)
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
    data = pd.read_csv("data.csv")
    data.plot.scatter('km', 'price')
    y = np.array(data['price'])
    y = normaliseMinMax(y)

    X = []
    x0 = np.ones(len(y))
    x1 = np.array(data['km'])
    x1 = normaliseMinMax(x1)
    X.append(x0)
    X.append(x1)

    theta = np.zeros(2)
    #theta = fit(X, y, theta, 0.02, 1000)
    theta, J_history = fit_with_cost(X, y, theta, 0.02, 1000)
    print(theta)
    visualizeRegression(X, y, theta)
    visualizeCost(J_history)
    pd.DataFrame.to_csv(theta)
