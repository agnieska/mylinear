import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics
import sys

def read_file (filename) :
    try :
        data = pd.read_csv(filename, sep=",")
    except :
        sys.exit("Error : data file", filename, "not found. Try another file")
    columns = list(data.columns.values)
    name_X = columns[0]
    name_Y = columns[1]
    if (not name_X.isalpha() or not name_Y.isalpha() ):
        sys.exit("problem to read column labels : \nX column label = "+ name_X +" \nY column label = "+ name_Y + " \ntry another file")
    X_raw = np.array(data[name_X].astype(float))
    y = np.array(data[name_Y].astype(float))
    if (len(X_raw) < 10 or len(y < 10)) :
        sys.exit("not enough data to learn")
    return X_raw, y, name_X, name_Y

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
    mean_error = np.sum((errors**2)**0.5)/m
    total_cost = (1/(2*m))*(np.sum(errors**2))
    return mean_error, total_cost

# learning function
def learn(X, y, theta, alpha, num_iters):    
    #print("nombre d'enregistrements m = " , m)
    #print("pas d'apprentissage alpha = " , alpha)
    X0 = np.ones(len(X))
    m = len(X)   #m = X.shape[0]
    gradient = [0,0]
    E_history = []
    C_history = []
    G_history = []
    T_history = []
    for _ in range(num_iters):
        
        # calcul error between hipothesis and real data (Y)
        errors = hipothesis(X, theta) - y
        
        # calcul total cost (sum of errors) for gradient and mean error for prediction
        mean_error, total_cost = calcul_cost(errors, m)

        #calcul gradient descent avec pas d'apprentissage alpha    
        gradient[0] = alpha * np.dot(errors, X0)/ m
        gradient[1] = alpha * np.dot(errors, X)/ m
        
        #update theta 0 et theta 1
        theta[0] = theta[0] - gradient[0] 
        theta[1] = theta[1] - gradient[1]

        # memorise this iteration
        E_history.append(mean_error)
        C_history.append(total_cost)
        gradient_copy = gradient.copy()
        G_history.append(gradient_copy)
        theta_copy = theta.copy()
        T_history.append(theta_copy)
    return theta, E_history, C_history, G_history, T_history 


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
    plt.title("Bonus : " + history_name + ' evolution', fontsize=16, fontweight='bold')
    plt.xlabel('Iteration', fontsize=14, fontweight='bold')
    plt.ylabel(history_name, fontsize=14, fontweight='bold')
    ax.plot(history_list)
    plt.show()

def save_parameters (theta, mean, stdev, mean_error) :
    line = str(theta[0])+","+ str(theta[1]) + "," + str(mean)+","+ str(stdev)+","+ str(mean_error)
    try :
        with open ("parameters.txt", "w" , encoding="utf-8") as file :
            file.write(line)
        file.close
    except :
        sys.exit("Error : File could not be saved")

def main(filename):
    
    
    
    print("----------------------------------------------------------------------\n")
    print("Phase I  READ DATA FOR LEARNING\n")
    # lecture des donnees
    X_raw , y , name_X, name_Y = read_file (filename)
    print("\nFound data 'X' to learn : \ncolumn name =", name_X , X_raw)
    print("\nFound data 'y' to predict : \ncolumn name =", name_Y, y)
    
    # visualisation du dataset
    print("\nBonus 1 : Visualize  dataset before normalization")
    #data.plot.scatter(name_X, name_Y)
    visualizeData(X_raw, y, name_X, name_Y)
    
    print("\n----------------------------------------------------------------------")

    print("\n\nPhase II DATA NORMALIZATION")
    X_norm, mean, stdev = centrer_reduire(X_raw)
    print("\nX normalized = ", X_norm)
    
    print("\n----------------------------------------------------------------------")
    
    print("\n\nPhase III LARNING")
    
    #initialise 2 linear coefficients theta[1] et theta[0] a zero
    theta = np.zeros(2)
    alpha = 0.2
    iterations = 100
    print("\nLinear coefficients before learning" , theta)
    print("Try to learn with learning rate =", alpha," iterations number =", iterations )

    theta, E_history, C_history, G_history, T_history = learn(X_norm, y, theta, alpha, iterations)
    print("Learning acomplished")
    print("Linear coefficients after learning" , theta)
    mean_error = E_history[-1]
    print("\nBonus 2 : Mean error after learning: ", mean_error)
    
    # Learnig visualisations
    print("Bonus 3 : Visualize linear regression" )
    visualizeRegression(X_norm, y, theta, name_X, name_Y)
    print("Bonus 4 : Visualize all iterations of coeficients search" )
    visualizeHistory(T_history, "Linear coefficients", 3)
    print("Bonus 5 : Visualize all iterations of gradient descend " )
    visualizeHistory(G_history, "Gradient", 4)
    print("Bonus 6 : Visualize all iterations of mean error" )
    visualizeHistory(E_history, "Mean error", 5)
    visualizeHistory(C_history, "Total cost", 6)
    
    print("\n----------------------------------------------------------------------")

    #saving results
    print("\n\nPhase IV : SAVING PARAMETERS ")
    save_parameters (theta, mean, stdev, mean_error)
    print("\nParameters saved to file parameters.txt")
    print("Use predict.py [option:parameters.txt] to predict price")
    
    print("\n----------------------------------------------------------------------\n")

if __name__ == '__main__':
    filename = "data.csv"
    if len(sys.argv) == 2 :
        filename = sys.argv[1]
    elif len(sys.argv) > 2 :
        sys.exit("Error : Wrong number of parameters. expected one data file or nothing")
    main(filename)