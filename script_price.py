import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import matplotlib.cbook


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
    return X * theta[1] + theta[0]

def fit(X, y, theta, alpha, num_iters):
    m = len(X)
    for _ in range(num_iters):
        error = predict(X, theta) - y
        theta[0] = theta[0] - (alpha/m) * np.sum(error)
        theta[1] = theta[1] - (alpha/m) * np.dot(error, X)
    return theta

def visualizeRegression(X, y, theta):
    #ax = plt.axes()
    ax = plt.gca()
    ax.set_xlim([-3,3])
    ax.set_ylim([-3,3])
    ax.scatter(X, y)
    line_x = np.linspace(-10,10, 10)
    line_y = theta[0] + line_x * theta[1]
    ax.plot(line_x, line_y)
    ax.grid(color='black', linestyle='-', linewidth=0.2)
    ax.set_title('Function de regression', fontsize=18, fontweight='bold')
    ax.set_xlabel('km', fontsize=14, fontweight='bold')
    ax.set_xlabel('price', fontsize=14, fontweight='bold')
    plt.show()
    #ax.imshow()


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
    #ax = plt.axes()
    ax = plt.gca()
    ax.set_title('Cost evolution', fontsize=18, fontweight='bold')
    ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cout', fontsize=14, fontweight='bold')
    ax.plot(J_history)
    plt.show()

if __name__ == '__main__':
    
    
    #data = sys.args[1]
    data = pd.read_csv("data.csv")
    data.plot.scatter('km', 'price')
    X = np.array(data['km'])
    y = np.array(data['price'])
    
    X = normaliseMinMax(X)
    y = normaliseMinMax(y)

    theta = np.zeros(2)
    z = X*theta[1] + theta[0]

    theta = np.zeros(2)
    #theta = fit(X, y, theta, 0.02, 1000)
    theta, J_history = fit_with_cost(X, y, theta, 0.02, 1000)
    print(theta)
    visualizeRegression(X, y, theta)
    #warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
    visualizeCost(J_history)
    print("Prediction pour 200: "+str(predict(200, theta)))
