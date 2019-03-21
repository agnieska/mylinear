import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import matplotlib.cbook
from pprint import pprint


def normaliseMeanStd(X):
    mean = X.mean(axis=0)
    stdev = X.std(axis=0)
    X = (X - mean)/stdev
    return X, mean, stdev

def normaliseMinMax(X):
    maximum = X.min(axis=0)
    minimum = X.max(axis=0)
    X = (X - minimum)/(maximum-minimum)
    return X, minimum, maximum

def predict(X, theta):
    return X*theta[1] + theta[0]

def fit(X, y, theta, alpha, num_iters):
    m = len(X)
    for _ in range(num_iters):
        error = predict(X, theta) - y
        theta[0] = theta[0] - (alpha/m) * np.sum(error)
        theta[1] = theta[1] - (alpha/m) * np.dot(error, X)
    return theta

def cost(X, y, theta):
    m = len(X)
    error = predict(X, theta) - y
    cost = (1/(2*m))*(np.sum(error**2))
    return cost

def fit_with_cost(X, y, theta, alpha, num_iters):
    m = len(X)  
    C_history = []
    T_history = []
    for _ in range(num_iters):
        error = predict(X, theta) - y
        theta[0] = theta[0] - (alpha/m) * np.sum(error)
        theta[1] = theta[1] - (alpha/m) * np.dot(error, X)
        T_history.append(theta.copy())
        #C_history.append(cost(X, y, theta))
    return theta, T_history, C_history

def visualizeRegression(X, y, theta):
    plt.figure(1)
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
    #plt.show()
    #ax.imshow()
    
def visualizeCost (J_history) :
    plt.figure(2)
    #ax = plt.axes()
    ax = plt.gca()
    ax.set_title('Cost evolution', fontsize=18, fontweight='bold')
    ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cout', fontsize=14, fontweight='bold')
    ax.plot(J_history)
    #plt.show()

def visualizeTheta (T_history) :
    plt.figure(3)
    ax = plt.axes()
    plt.title('Theta evolution', fontsize=18, fontweight='bold')
    plt.xlabel('Iteration', fontsize=14, fontweight='bold')
    plt.ylabel('Theta', fontsize=14, fontweight='bold')
    ax.plot(T_history)
    plt.show()
    
def main(argv):    
    data = pd.read_csv(argv)
    #data.plot.scatter('km', 'price')
    X = np.array(data['km'])
    print("X : " + str(type(X)))
    pprint(X)
    y = np.array(data['price'])
    print("y : " + str(type(y)))
    pprint(y)
    
    """
    X, mini, maxi = normaliseMinMax(X)
    print("normalized X : " + str(type(X)))
    pprint(X)
    y = normaliseMinMax(y)
    print("normalized y : " + str(type(y)))
    pprint(y)
    """
    theta = np.zeros(2)
    z = X * theta[1] + theta[0]
    print("Z : " + str(type(z)))
    pprint(z)
    theta = np.zeros(2)
    #theta = fit(X, y, theta, 0.02, 1000)
    theta, T_history, J_history = fit_with_cost(X, y, theta, 0.01, 10000)
    print("theta 0 : " + str(theta[0]))
    print("theta 1 : " + str(theta[1]))
    
    visualizeTheta (T_history)
    visualizeRegression(X, y, theta)
    visualizeCost(J_history)
    
    print("Prediction pour 60000: " + str(predict(240000, theta)))
    plt.show()

if __name__ == '__main__':
    #main(sys.argv[1])
    main("data.csv")