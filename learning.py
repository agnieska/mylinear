import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics
import sys

def read_file (filename) :
    try :
        data = pd.read_csv(filename, sep=",")
    except :
        sys.exit("ERROR: Data file "+filename+" not found. Try another file")
    columns = list(data.columns.values)
    name_X = columns[0]
    name_Y = columns[1]
    if (not name_X.isalpha() or not name_Y.isalpha() ):
        sys.exit("ERROR: Problem to read column labels : \nX column label = "+ name_X +" \nY column label = "+ name_Y + " \ntry another file")
    X_raw = np.array(data[name_X].astype(float))
    y = np.array(data[name_Y].astype(float))
    if len(X_raw) < 24 and len(y) < 24 :
        sys.exit("WARNING: Not enough data to learn")
    return X_raw, y, name_X, name_Y

def centrer_reduire (X):
    stdev = statistics.stdev(X)
    mean = statistics.mean(X)
    A = []
    for x in X :
        a = float((x - mean)/stdev)
        A.append(a)
    return np.array(A), stdev, mean

def calcul_cost(loss_vector):
    sample_size = len(loss_vector)
    sum_loss_square = sum(loss_vector ** 2)
    double_size = 2 * sample_size
    cost = sum_loss_square / double_size
    return cost

def calcul_error(loss_vector):
    sample_size = len(loss_vector)
    loss_absolute = (loss_vector **2) **0.5
    max_error = max(loss_absolute)
    mean_error = sum(loss_absolute) / sample_size
    return mean_error, max_error

def learn(X, y, theta, alpha, num_iters):    
    X0 = X ** 0
    X1 = X ** 1
    sample_size = len(X)   #sample_size = X.shape[0]
    E_history = []
    C_history = []
    G_history = []
    T_history = []
    for _ in range(num_iters):
        # calcul delta (ecart) between hipothesis and real empirical data (Y)
        estimated_results = theta[0] * X0  +  theta[1] * X1
        empirical_results = y
        loss_vector = estimated_results - empirical_results
        #calcul gradient descent avec pas d'apprentissage alpha    
        gradient_X0 =  - alpha / sample_size * sum(loss_vector * X0)
        gradient_X1 =  - alpha / sample_size * sum(loss_vector * X1)
        #calcul new coefficients
        theta_X0 = theta[0] + gradient_X0 
        theta_X1 = theta[1] + gradient_X1
        #update theta 0 et theta 1
        theta = [ theta_X0, theta_X1 ]
        # calcul total cost (sum of errors) for gradient and mean error for prediction
        total_cost = calcul_cost(loss_vector)
        mean_error, max_error = calcul_error(loss_vector)
        # memorise each iteration
        E_history.append([mean_error, max_error])
        C_history.append(total_cost)
        G_history.append([ gradient_X0.copy(), gradient_X1.copy()])
        T_history.append([ theta_X0.copy(), theta_X1.copy() ])
    return theta, E_history, C_history, G_history, T_history 

def visualizeData(X, y, name_X, name_Y):  
    plt.figure(1)
    ax = plt.axes()
    ax.set_xlim([0 , max(X)*1.2])
    ax.set_ylim([min(y)*0.8,max(y)*1.2])
    ax.scatter(X, y)
    plt.title('Dataset visualisation', fontsize=18, fontweight='bold')
    plt.xlabel(name_X, fontsize=14, fontweight='bold')
    plt.ylabel(name_Y, fontsize=14, fontweight='bold')
    ax.grid(color='black', linestyle='-', linewidth=0.9)
    plt.show()

def visualizeRegression(X, y, theta, name_X, name_Y):
    plt.figure(2)
    ax = plt.axes()
    ax.set_xlim([min(X)*1.2,max(X)*1.2])
    ax.set_ylim([min(y)*0.8,max(y)*1.2])
    ax.scatter(X, y)
    line_x = np.linspace(min(X)*1.2 , max(X)*1.2, 26)
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

def save_parameters (theta, mean, stdev, mean_error, max_error) :
    line = str(theta[0])+","+ str(theta[1]) + "," + str(mean)+","+ str(stdev)+","+ str(mean_error)+","+ str(max_error)
    try :
        with open ("parameters.txt", "w" , encoding="utf-8") as file :
            file.write(line)
        file.close
    except :
        sys.exit("WARNING: File could not be saved")

def main(filename):
    print("----------------------------------------------------------------------\n")
    print("Phase I  READ DATA FOR LEARNING")
    # lecture des donnees
    X_raw , y , name_X, name_Y = read_file(filename)
    print("\nSUCCESS: Found data 'X' to learn. \ncolumn name =", name_X , X_raw)
    print("\nSUCCESS: Found data 'y' to predict. \ncolumn name =", name_Y, y)
    
    # visualisation du dataset
    print("\nBonus 1 : Visualize  dataset before normalization")
    visualizeData(X_raw, y, name_X, name_Y)
    print("\n----------------------------------------------------------------------")
    print("\nPhase II DATA NORMALIZATION")
    X_norm, mean, stdev = centrer_reduire(X_raw)
    print("\nSUCCESS: X normalized = ", X_norm)
    print("\n----------------------------------------------------------------------")
    print("\nPhase III LARNING")
    
    #initialise 2 linear coefficients theta[1] et theta[0] à zero
    theta = np.zeros(2)
    alpha = 0.2
    iterations = 50
    print("\n...Linear coefficients before learning" , theta)
    print("...Try to learn with learning rate =", alpha," iterations number =", iterations )

    #learning
    theta, E_history, C_history, G_history, T_history = learn(X_norm, y, theta, alpha, iterations)
    print("\nSUCCESS: Learning acomplished")
    print("RESULTS: Linear coefficients after learning" , theta)
    mean_error, max_error = E_history[-1]
    print("\nBonus 2 : Precision after learning:")
    print("            Mean error:  =  +-"+str(round(mean_error,0)))
    print("            Max error:  =  +-"+str(round(max_error, 0)))
    
    # visualisations
    print("Bonus 3 : Visualize linear regression" )
    visualizeRegression(X_norm, y, theta, name_X, name_Y)
    print("Bonus 4 : Visualize all iterations of coeficients search" )
    visualizeHistory(T_history, "Linear coefficients", 3)
    print("Bonus 5 : Visualize all iterations of gradient descend " )
    visualizeHistory(G_history, "Gradient", 4)
    print("Bonus 7 : Visualize all iterations of total cost function" )
    visualizeHistory(C_history, "Total cost", 6)
    print("Bonus 6 : Visualize all iterations of mean and max error" )
    visualizeHistory(E_history, "Precision", 5)
    print("\n----------------------------------------------------------------------")

    #saving results
    print("\nPhase IV : SAVING PARAMETERS ")
    save_parameters (theta, mean, stdev, mean_error, max_error)
    print("\nSUCCESS: Parameters saved to file parameters.txt")
    print("USAGE: Use predict.py [option:parameters.txt] to predict price")
    print("\n----------------------------------------------------------------------\n")

if __name__ == '__main__':
    filename = "data.csv"
    if len(sys.argv) == 1 :
        print("\nNo parameters. Default file 'data.csv' will be used.")
        main(filename)
    elif len(sys.argv) == 2 :
        filename = sys.argv[1]
        main(filename)
    else :
        sys.exit("ERROR: Wrong number of parameters. Expected one data file or nothing")
    