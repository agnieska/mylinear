import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#import statistics
import sys


def predict (x, mean, stdev, theta):
    x = (x-mean)/stdev
    y = x * theta[1] + theta[0]
    if y < 0 : y = 0
    return y

def read_file (theta_filename) :
    theta = [0,0]
    try : 
        with open (theta_filename) as file :
            line = file.readline
    except :
        print ("file not found")
    try :
        [theta[0], theta[1], mean, stdev] = line.strip(" ").split(",")
    except :
        print ("wrong numbers of parameters in first line. Expected 4")
    return theta, mean, stdev


def main (theta_filename) :
    theta, mean, stdev = read_file (theta_filename)
    kilometrage = input ("Quel kilomatrage ?")
    while kilometrage not in ["q", "quit", "\n"]:
        prediction = predict (kilometrage, mean, stdev, theta)
        print ("predicted price :" , prediction, " euro")
        kilometrage = input ("Another  kilomatrage to predict ?")


if __name__ == '__main__':
    main(sys.argv[1])