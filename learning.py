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

def calcul_mean_error(errors, m):
    mean_error = (1/(2*m))*(np.sum(errors**2))
    return mean_error

def fit(X, y, theta, alpha, num_iters):
    
    #print("nombre d'enregistrements m = " , m)
    #print("pas d'apprentissage alpha = " , alpha)
    X0 = np.ones(len(X))
    m = len(X)   #m = X.shape[0]
    gradient = [0,0]
    E_history = []
    G_history = []
    T_history = []
    for _ in range(num_iters):
        
        # calcul error between hipothesis and real data (Y)
        errors = hipothesis(X, theta) - y
        
        # calcul total cost (sum of errors)
        mean_error = calcul_mean_error(errors, m)

        #calcul gradient descent avec pas d'apprentissage alpha    
        gradient[0] = alpha * np.dot(errors, X0)/ m
        gradient[1] = alpha * np.dot(errors, X)/ m
        
        #update theta 0 et theta 1
        theta[0] = theta[0] - gradient[0] 
        theta[1] = theta[1] - gradient[1]

        # memorise this iteration
        E_history.append(mean_error)
        gradient_copy = gradient.copy()
        G_history.append(gradient_copy)
        theta_copy = theta.copy()
        T_history.append(theta_copy)
    return theta, E_history, G_history, T_history 


def visualizeData(X, y, name_X, name_Y):  
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
    
def visualizeHistory (history_list, history_name, figure_number) :
    plt.figure(figure_number)
    ax = plt.axes()
    plt.title("Bonus : " + history_name + ' evolution', fontsize=18, fontweight='bold')
    plt.xlabel('Iteration', fontsize=14, fontweight='bold')
    plt.ylabel(history_name, fontsize=14, fontweight='bold')
    ax.plot(history_list)
    plt.show()

def visualizeError (E_history) :
    plt.figure(4)
    ax = plt.axes()
    plt.title('Bonus : Mean error evolution', fontsize=18, fontweight='bold')
    plt.xlabel('Iteration', fontsize=14, fontweight='bold')
    plt.ylabel('Error', fontsize=14, fontweight='bold')
    ax.plot(E_history)
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

def save_parameters (theta, mean, stdev, mean_error) :
    line = str(theta[0])+","+ str(theta[1]) + "," + str(mean)+","+ str(stdev)+","+ str(mean_error)
    with open ("parameters.txt", "w" , encoding="utf-8") as file :
        file.write(line)
    file.close

def main(argv):
    # lecture des donnees
    data = pd.read_csv(filename, sep=",")
    columns = list(data.columns.values)
    name_X = columns[0]
    name_Y = columns[1]
    X_raw = np.array(data[name_X].astype(float))
    y = np.array(data[name_Y].astype(float))
    
    print("Phase I  Lecture des donnees pour apprentissage")
    print("Found data 'X' to learn : ", name_X , X_raw)
    print("Found data 'y' to predict : ", name_Y, y)
    
    print("Bonus 1 : visualisation  des données bruts avant la normalisation")
    #data.plot.scatter(name_X, name_Y)
    visualizeData(X_raw, y, name_X, name_Y)
    
    print("Phase II Normalisation des donnees")
    X_norm, mean, stdev = centrer_reduire(X_raw)
    print("X normalized = ", X_norm)
    
    
    print("Phase III Apprentissage")
    theta = np.zeros(2)
    alpha = 0.2
    iterations = 100
    print("Coefficients  avant apprentissage" , theta)
    print("Try to learn with alpha = ", alpha," nombre d'iterations =", iterations )
    #theta = fit(X_norm, y, theta, 0.2, 100)
    theta, E_history, G_history, T_history = fit(X_norm, y, theta, alpha, iterations)
    print(E_history)
    mean_error = E_history[-1]
    print("Coefficients apres apprentissage" , theta)
    print("Bonus 2 : Erreur d'apprentissage : ", mean_error)
    
    # Learnig visualisations
    print("Bonus 3 : visualise regression" )
    visualizeRegression(X_norm, y, theta, name_X, name_Y)
    print("Bonus 4 : visualise all iterations of coeficients search" )
    visualizeHistory(T_history, "Coeficients Theta", 3)
    print("Bonus 5 : visualise all iterations of gradient descend " )
    visualizeHistory(G_history, "Gradient", 4)
    print("Bonus 6 : visualise all iterations of mean error" )
    visualizeHistory(E_history, "Mean Error", 5)
    
    #saving results
    print("Phase IV : Saving parameters ")
    save_parameters (theta, mean, stdev, mean_error)
    print("parameters saved to file parameters.txt")

if __name__ == '__main__':
    filename = "price_data.csv"
    if len(sys.argv) == 2 :
        filename = sys.argv[1]
    elif len(sys.argv) > 2 :
        sys.exit("Wrong number of parameters. expected one data file or nothing")
    main(filename)