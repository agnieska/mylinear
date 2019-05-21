import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics
import sys

def centrer_reduire (X):
    stdev = statistics.stdev(X)
    mean = statistics.mean(X)
    A = []
    for x in X :
        a = float((x - mean)/stdev)
        A.append(a)
    return np.array(A), stdev, mean

def hipothesis(X, theta):
    return X*theta[1] + theta[0]

def calcul_cost(errors, m):
    cost = (1/(2*m))*(np.sum(errors**2))
    return cost

def fit(X, y, theta, alpha, num_iters):
    
    #print("nombre d'enregistrements m = " , m)
    #print("pas d'apprentissage alpha = " , alpha)
    X0 = np.ones(len(X))
    m = len(X)   #m = X.shape[0]
    gradient = [0,0]
    C_history = []
    G_history = []
    T_history = []
    for _ in range(num_iters):
        
        # calcul error between hipothesis and real data (Y)
        errors = hipothesis(X, theta) - y
        
        # calcul total cost (sum of errors)
        total_cost = calcul_cost(errors, m)

        #calcul gradient descent avec pas d'apprentissage alpha    
        gradient[0] = alpha * np.dot(errors, X0)/ m
        gradient[1] = alpha * np.dot(errors, X)/ m
        
        #update theta 0 et theta 1
        theta[0] = theta[0] - gradient[0] 
        theta[1] = theta[1] - gradient[1]

        # memorise this iteration
        C_history.append(total_cost)
        gradient_copy = gradient.copy()
        G_history.append(gradient_copy)
        theta_copy = theta.copy()
        T_history.append(theta_copy)
    return theta, C_history, G_history, T_history 


def visualizeData(X, y, theta, name_X, name_Y):  
    plt.figure(1)
    ax = plt.axes()
    ax.set_xlim([0 ,max(X)*1.1])
    ax.set_ylim([min(y)*0.9,max(y)*1.1])
    ax.scatter(X, y)
    
    plt.title('Dataset visualisation', fontsize=18, fontweight='bold')
    plt.xlabel(name_X, fontsize=14, fontweight='bold')
    plt.ylabel(name_Y, fontsize=14, fontweight='bold')
    ax.grid(color='black', linestyle='-', linewidth=0.9)
    plt.show()


def visualizeRegression(X, y, theta, name_X, name_Y):
    plt.figure(2)
    ax = plt.axes()
    ax.set_xlim([0 ,max(X)*1.1])
    ax.set_ylim([min(y)*0.9,max(y)*1.1])
    ax.scatter(X, y)
    
    line_x = np.linspace(0,max(X)*1.1, 20)
    line_y = theta[0] + line_x * theta[1]
    ax.plot(line_x, line_y)
    
    plt.title('Function de regression', fontsize=18, fontweight='bold')
    plt.xlabel(name_X, fontsize=14, fontweight='bold')
    plt.ylabel(name_Y, fontsize=14, fontweight='bold')
    ax.grid(color='black', linestyle='-', linewidth=0.9)
    plt.show()
'''    
def visualizeHistory (history_list, history_name, figure_number) :
    plt.figure(figure_number)
    ax = plt.axes()
    plt.title("Bonus : " + history_name + ' evolution', fontsize=18, fontweight='bold')
    plt.xlabel('Iteration', fontsize=14, fontweight='bold')
    plt.ylabel(history_name, fontsize=14, fontweight='bold')
    ax.plot(history_list)
    plt.show()
'''
def visualizeCost (C_history) :
    plt.figure(4)
    ax = plt.axes()
    plt.title('Bonus : Cost evolution', fontsize=18, fontweight='bold')
    plt.xlabel('Iteration', fontsize=14, fontweight='bold')
    plt.ylabel('Cost', fontsize=14, fontweight='bold')
    ax.plot(C_history)
    plt.show()

def visualizeGradient (G_history) :
    plt.figure(4)
    ax = plt.axes()
    plt.title('Bonus : Gradient evolution', fontsize=18, fontweight='bold')
    plt.xlabel('Iteration', fontsize=14, fontweight='bold')
    plt.ylabel('Gradient', fontsize=14, fontweight='bold')
    ax.plot(G_history)
    plt.show()

def visualizeTheta (T_history) :
    plt.figure(4)
    ax = plt.axes()
    plt.title('Bonus : Theta evolution', fontsize=18, fontweight='bold')
    plt.xlabel('Iteration', fontsize=14, fontweight='bold')
    plt.ylabel('Theta', fontsize=14, fontweight='bold')
    ax.plot(T_history)
    plt.show()

def save_parameters (theta, mean, stdev) :
    line = str(theta[0])+","+ str(theta[1]) + "," + str(mean)+","+ str(stdev)
    with open ("parameters.txt", "w" , encoding="utf-8") as file :
        file.write(line)
    file.close

def main(argv):  
    data = pd.read_csv(filename, sep=",")
    name_X = "km"
    name_Y = "price"
    columns = list(data.columns.values)
    print(columns)
    name_X = columns[0]
    name_Y = columns[1]
    print(name_X, name_Y)
    
    print("Bonus : visualisation  des donnees ")
    data.plot.scatter(name_X, name_Y)
    
    X_raw = np.array(data[name_X].astype(float))
    y = np.array(data[name_Y].astype(float))
    print("X raw = ", X_raw)
    print("y = ", y)
    X_norm, mean, stdev = centrer_reduire(X_raw)
    print("X normalized = ", X_norm)
    theta = np.zeros(2)
    #theta = fit(X_norm, y, theta, 0.2, 100)
    theta, C_history, G_history, T_history = fit(X_norm, y, theta, 0.2, 100)
    print("theta apres le fit" , theta)
    
    print("bonus : visualise regression" )
    visualizeRegression(X_norm, y, theta, name_X, name_Y)
    
    print("bonus : visualise total cost iterations" )
    visualizeCost(C_history)
    #visualizeHistory(C_history, "Cost", 3)
    
    print("bonus : visualise gradient descend" )
    visualizeGradient (G_history) 
    
    print("bonus : visualise theta iterations" )
    visualizeTheta (T_history) 
    
    #saving results
    save_parameters (theta, mean, stdev)
    print("parameters saved")

if __name__ == '__main__':
    filename = "price_data.csv"
    if len(sys.argv) == 2 :
        filename = sys.argv[1]
    elif len(sys.argv) > 2 :
        sys.exit("Wrong number of parameters. expected one data file or nothing")
    main(filename)