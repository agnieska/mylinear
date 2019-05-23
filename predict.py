import numpy as np
import sys


def predict (x, mean, stdev, theta):
    x = (x-mean)/stdev
    y = x * theta[1] + theta[0]
    if y < 0 : y = 0
    return y

def read_file (filename) :
    theta = [0,0]
    mean = 0
    stdev = 0.00001
    mean_error = 0
    max_error = 0
    parameters = []
    try : 
        with open (filename, "r", encoding="utf-8") as file :
            #print("with open")
            line = file.readline()
            #print(line)
    except :
        print ("WARNING: The file '",filename, "' with training results not found.")
        resp = input ("      Do you want to continue with parameters initialized as zero? Y/N \n")
        if resp in ["Y", "y", "Yes", "YES"] :
            return theta, mean, stdev , mean_error, max_error
        else :
            sys.exit("\nUSAGE: Execute learning.py to create parameters.txt file\n")
    try :
        parameters = line.split(",")
        #print(str(parameters))
    except :
        print ('''ERROR: Wrong format of parameters in the file. Expected line with 4 parameters separated by ","''')
        return theta, mean, stdev , mean_error, max_error
    try :
        theta[0] = float(parameters[0].strip(" "))
        theta[1] = float(parameters[1].strip(" "))
        mean = float(parameters[2].strip(" "))
        stdev = float(parameters[3].strip(" "))
        mean_error =  float(parameters[4].strip(" "))
        max_error =  float(parameters[5].strip(" "))
    except :
        print ("ERROR: Wrong number of parameters in first line. Expected 4.")
        return theta, mean, stdev , mean_error, max_error
    return theta, mean, stdev , mean_error, max_error


def main (filename) :
    print("\n-------------------------------------------------------------------")
    print("READ LEARNING PARAMETERS\n")
    theta, mean, stdev, mean_error, max_error = read_file(filename)
    print ("\ncoef X0 =", theta[0], ", coef X1 =", theta[1])
    print("mean error =", mean_error)
    print("\n-------------------------------------------------------------------")
    print("PREDICTION")
    kilometrage = input ("\nQuel kilometrage ?\n").strip(" ")
    while kilometrage not in ["q", "quit", ""]:
        try :
            km = float(kilometrage)
            prediction = round(predict(km, mean, stdev, theta), 2)
            print ("\nPrediction prix :" , prediction, " euro")
            print("Precision du resultat :")
            print ("         ecart moyen de prediction : +-"+str(round(mean_error,0))+" euro")
            print ("         ecart maximal de prediction : +-"+str(round(max_error,0))+" euro")
        except :
            print("\nERROR: Wrong format of input. Try again.")
        
        kilometrage = input("\nAutre kilometrage ?\n").strip(" ")


if __name__ == '__main__': 
    filename = "parameters.txt"
    if len(sys.argv) == 2 :
        filename = sys.argv[1]
    elif len(sys.argv) > 2 :
        sys.exit("ERROR: Wrong number of parameters. expected one data file or nothing.")
    main(filename)