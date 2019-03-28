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
        print("error : " + str(error))
        theta[0] = theta[0] - (alpha/m) * np.sum(error)
        theta[1] = theta[1] - (alpha/m) * np.dot(error, X)
        t = theta.copy()
        T_history.append(t)
        C_history.append(cost(X, y, theta))
    return theta, T_history, C_history

def visualizeRegression(X, y, theta):
    plt.figure(1)
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

def visualizeCost (C_history) :
    plt.figure(2)
    ax = plt.axes()
    plt.title('Cost evolution', fontsize=18, fontweight='bold')
    plt.xlabel('Iteration', fontsize=14, fontweight='bold')
    plt.ylabel('Cout', fontsize=14, fontweight='bold')
    ax.plot(C_history)
    plt.show()

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
    data.plot.scatter('population', 'profit')
    X = np.array(data['population'])
    y = np.array(data['profit'])
    
    theta = np.zeros(2)
    print("main theta : " + str(theta))
    Z = X*theta[1] + theta[0]
    print("main Z : " + str(Z))
    theta = np.zeros(2)
    print("main theta : " + str(theta))
    #theta = fit(X, y, theta, 0.02, 1000)
    theta, T_history, J_history = fit_with_cost(X, y, theta, 0.02, 10)
    print(theta)
    visualizeRegression(X, y, theta)
    visualizeCost(J_history)
    visualizeTheta (T_history) 
    print("Prediction pour 5734: "+str(predict(5734, theta)))

if __name__ == '__main__':
    #main(sys.argv[1])
    main("ex1data1.csv")