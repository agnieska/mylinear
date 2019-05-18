import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics
import sys


filename = "price_data.csv"
name_X = "km"
name_Y = "price"

def hipothesis(X, theta):
    return X*theta[1] + theta[0]

def fit(X, y, theta, alpha, num_iters):
    m = len(X)
    #m = X.shape[0]
    print("m = " , m)
    print("alpha = " , alpha)
    #gradient = [0,0]
    #X0 = np.ones(len(X))
    for _ in range(num_iters):
        errors = hipothesis(X, theta) - y
        print("errors : " , errors)
        theta[0] = theta[0] - (alpha * np.sum(errors))/m
        theta[1] = theta[1] - (alpha * np.dot(errors, X))/m
        #theta[0] = theta[0] - (alpha * np.dot(errors, X0))/m
        #theta[1] = theta[1] - (alpha * np.dot(errors, X))/m
        print("theta :", theta)
        #gradient[0] = alpha * np.dot(errors, X0)/ m
        #gradient[1] = alpha * np.dot(errors, X)/ m
        #print("gradient : " , gradient)
        #theta[0] = theta[0] - gradient[0] 
        #theta[1] = theta[1] - gradient[1]
#       theta[1] = theta[1] - (alpha/m) * np.sum(error * X.T)
        print("theta iter :" , theta)
    return theta

def centrer_reduire (X):
    stdev = statistics.stdev(X)
    mean = statistics.mean(X)
    A = []
    for x in X :
        a = float((x - mean)/stdev)
        A.append(a)
    return np.array(A), stdev, mean


def calcul_cost(errors, m):
    cost = (1/(2*m))*(np.sum(errors**2))
    return cost

def fit_with_cost(X, y, theta, alpha, num_iters):
    m = len(X)  
    C_history = []
    T_history = []
    for _ in range(num_iters):
        errors = hipothesis(X, theta) - y
        #print("errors : " + str(errors))
        theta[0] = theta[0] - (alpha * np.sum(errors))/m
        theta[1] = theta[1] - (alpha * np.dot(errors, X))/m
        t = theta.copy()
        T_history.append(t)
        total_cost = calcul_cost(errors, m)
        C_history.append(total_cost)
    return theta, T_history, C_history

def visualizeRegression(X, y, theta):
    print("minX : " , min(X))
    print("max X : " , max(X))
    print("min y : " , min(y))
    print("max y : " , max(y))
    print("theta : " , theta)
    plt.figure(1)
    ax = plt.axes()
    ax.set_xlim([0 ,max(X)*1.1])
    ax.set_ylim([min(y)*0.9,max(y)*1.1])
    ax.scatter(X, y)
    line_x = np.linspace(0,max(X)*1.1, 20)
    print(line_x)

    line_y = theta[0] + line_x * theta[1]
    print(line_y)
    ax.plot(line_x, line_y)
    plt.title('Function de regression', fontsize=18, fontweight='bold')
    plt.xlabel(name_X, fontsize=14, fontweight='bold')
    plt.ylabel(name_Y, fontsize=14, fontweight='bold')
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
'''
def predict (x, mean, stdev, theta):
    x = (x-mean)/stdev
    y = hipothesis(x, theta)
    if y < 0 : y = 0
    return y
'''
def save_parameters (theta, mean, stdev) :
    line = str(theta[0])+","+ str(theta[1]) + "," + str(mean)+","+ str(stdev)
    with open ("parameters.txt", "w" , encoding="utf-8") as file :
        file.write(line)
    file.close

def main(argv):  
    data = pd.read_csv(argv, sep=",")
#    print(type(data[indep][0]))
    #data.plot.scatter(indep, dep)
    X_raw = np.array(data[name_X].astype(float))
    y = np.array(data[name_Y].astype(float))
    print("X raw = ", X_raw)
    print("y = ", y)
    X_norm, mean, stdev = centrer_reduire(X_raw)
    print("X normalized = ", X_norm)
    #X = (500000 - X)/1000
    #print("X reversed = ", X)
    theta = np.zeros(2)
    print("main theta : " + str(theta))
#    Z = X*theta[1] + theta[0]
#    print("main first occurence de Z : " + str(Z))
#    theta = [0, 0]
#    print("main theta : " + str(theta))
    theta = fit(X_norm, y, theta, 0.2, 100)
    print("theta apres le fit" , theta)
    #theta, T_history, J_history = fit_with_cost(X_norm, y, theta, 0.2, 100)
    visualizeRegression(X_norm, y, theta)
    #visualizeCost(J_history)
    #visualizeTheta (T_history) 
    #print("Prediction pour 5734: "+str(predict(5734, theta)))
    #x = X_raw[2] 
    #print("predic for" , x)
    #x = 54370
    #prediction = round(predict(x, mean , stdev , theta),2)
    #print("Estimated", name_Y,  "for", x, name_X ," = ", prediction , "units")
    #saving results
    save_parameters (theta, mean, stdev)

if __name__ == '__main__':
    #main(sys.argv[1])
    main(filename)