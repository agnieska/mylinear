import numpy as np


theta_filename = "parameters.txt"

def predict (x, mean, stdev, theta):
    x = (x-mean)/stdev
    y = x * theta[1] + theta[0]
    if y < 0 : y = 0
    return y
def read_file (theta_filename) :
    theta = [0,0]
    mean = 0
    stdev = 0.00001
    #mean_error = 0
    parameters = []
    try : 
        with open (theta_filename, "r", encoding="utf-8") as file :
            #print("with open")
            line = file.readline()
            #print(line)
    except :
        print ("training results not found. they will be initialized as zero")
        return theta, mean, stdev
    try :
        parameters = line.split(",")
        print(str(parameters))
    except :
        print ('''wrong format of parameters in first line. Expected line with sep=","''')
        return theta, mean, stdev
    try :
        theta[0] = float(parameters[0].strip(" "))
        theta[1] = float(parameters[1].strip(" "))
        mean = float(parameters[2].strip(" "))
        stdev = float(parameters[3].strip(" "))
        #mean_error =  float(parameters[4].strip(" "))
    except :
        print ("wrong numbers of parameters in first line. Expected 4")
        return theta, mean, stdev #, mean_error
    return theta, mean, stdev #, mean_error


def main (theta_filename) :
    theta, mean, stdev = read_file(theta_filename)
    kilometrage = float(input ("Quel kilometrage ?\n").strip(" "))
    while kilometrage not in ["q", "quit", "\n"]:
        prediction = round(predict (kilometrage, mean, stdev, theta), 2)
        print ("Prediction prix :" , prediction, " euro")
        kilometrage = float(input("Autre kilometrage ?\n").strip(" "))


if __name__ == '__main__':
    #main(sys.argv[1])
    main(theta_filename)