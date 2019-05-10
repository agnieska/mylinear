import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics


filename = "profit_data1.csv"
indep = 'population'
dep = "profit"
'''
filename = "price_data.csv"
indep = "km"
dep = "price"
'''
def hipothesis(X, theta):
    return X*theta[1] + theta[0]

def fit(X, y, theta, alpha, num_iters):
    m = len(X)
    #m = X.shape[0]
    print("m = " , m)
    print("alpha = " , alpha)
    gradient = [0,0]
    X0 = np.ones(len(X))
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
    d = statistics.stdev(X)
    m = statistics.mean(X)
    A = []
    for x in X :
        a = float((x - m) / d)
        A.append(a)
    return np.array(A), d, m


def calcul_cost(errors, m):
    cost = (1/(2*m))*(np.sum(errors**2))
    return cost

def fit_with_cost(X, y, theta, alpha, num_iters):
    m = len(X)  
    C_history = []
    T_history = []
    for _ in range(num_iters):
        errors = hipothesis(X, theta) - y
        print("errors : " + str(errors))
        theta[0] = theta[0] - (alpha * np.sum(errors))/m
        theta[1] = theta[1] - (alpha * np.dot(errors, X))/m
        t = theta.copy()
        T_history.append(t)
        total_cost = calcul_cost(errors, m)
        C_history.append(total_cost)
    return theta, T_history, C_history

def visualizeRegression(X, y, theta):
    print("minX : " , min(X))
    print("max X : " ,max(X))
    print("min y : " , min(y))
    print("max y : " , max(y))
    print("theta : " , theta)
    plt.figure(1)
    ax = plt.axes()
    ax.set_xlim([0 ,max(X)* 2])
    ax.set_ylim([min(y)*0.9,max(y)*1.1])
    ax.scatter(X, y)
    line_x = np.linspace(0,max(X)*1.1, 20)
    print(line_x)
    line_y = theta[0] + line_x * theta[1]
    print(line_y)
    ax.plot(line_x, line_y)
    plt.title('Function de regression', fontsize=18, fontweight='bold')
    plt.xlabel(indep, fontsize=14, fontweight='bold')
    plt.ylabel(dep, fontsize=14, fontweight='bold')
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

def predict(x, theta):
    return x*theta[1] + theta[0]

    
def main(argv):  
    data = pd.read_csv(argv, sep=",")
#    print(type(data[indep][0]))
    #data.plot.scatter(indep, dep)
    RawX = np.array(data[indep].astype(float))
    y = np.array(data[dep].astype(float))
    print("X = ", RawX)
    print("y = ", y)
    X, m, d = centrer_reduire(RawX)
    print("X = ", X)
    #X = (500000 - X)/1000
    #print("X reversed = ", X)
    theta = np.zeros(2)
    print("main theta : " + str(theta))
#    Z = X*theta[1] + theta[0]
#    print("main first occurence de Z : " + str(Z))
#    theta = [0, 0]
#    print("main theta : " + str(theta))
    #theta = fit(X, y, theta, 0.02, 1000)
    print("theta apres le fit" , theta)
    theta, T_history, J_history = fit_with_cost(X, y, theta, 0.02, 1000)
    visualizeRegression(X, y, theta)
    visualizeCost(J_history)
    visualizeTheta (T_history) 
    print("Prediction pour 5734: "+str(predict(5734, theta)))
    

if __name__ == '__main__':
    #main(sys.argv[1])
    main(filename)