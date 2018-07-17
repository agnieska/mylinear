package com.company;

import java.util.Vector;

public class SimpleLinearRegression {

    //  Une fonction fit qui prend en arguments le vecteur X
    //  et le vecteur y des données d'entraînement
    //  et renvoie le vecteur de paramètres theta qui a été appris.

    public static float[] fit (float[] X, float[] y, float[] theta, float alpha, int num_iters){

        int  N = X.length;
        float[] prediction = new float[X.length];
        float[] error = new float[X.length];
        float[] errorsqrt = new float[X.length];
        // Gradient descent qui boucle sur le nombre d'itérations
        for (int i = 1; i < num_iters; i++){
            prediction = predict(X, theta);
            error =  Vecteur.soustraction(prediction, y);
            errorsqrt = Vecteur.multiply(error, error);
            theta[0] = theta[0] - (alpha/N) * Vecteur.sumValues(errorsqrt);
            theta[1] = theta[1] - (alpha/N) * Vecteur.dot(error, X);
        }

        return theta;
    }


    public static float[] fit2 (float[] X, float[] y, float[] theta, float alpha, int num_iters){

        int  N = X.length;
        float[] prediction = new float[X.length];
        float[] error = new float[X.length];

        // Gradient descent qui boucle sur le nombre d'itérations
        for (int i = 1; i < num_iters; i++){
            prediction = predict(X, theta);
            error =  Vecteur.soustraction(prediction, y);
            theta = Vecteur.soustraction(theta, ((alpha/N) * Vecteur.sumValues(error)));
            theta[1] = theta[1] - (alpha/N) * Vecteur.dot(error, X);
        }
        return theta;
    }

    // Une fonction predict qui prend en argument une population (x)
    // ainsi que les parametres theta et prédit le profit (y) associé.

    public static float[] predict(float[] X, float[] theta){
        float[] result = new float[X.length];

        for (int i = 0; i < X.length; i++){
            result[i] = X[i] * theta[1] + theta[0];
        }
        return result;
    }


    public static double error (){

        double result = 0;
        return result;
    }


    // Fonction de coût du modele
    // resserre l'analyse de l'algorithme et calcule le coût (ou la perte, ou l'erreur) à chaque itération.

    public static double cost(float[] X, float[] y, float[] theta) {
        int N = X.length;
        float[] predict = new float[X.length];
        float[] error = new float[X.length];
        double cost = 0;
        predict = predict(X, theta);
        error = Vecteur.soustraction(predict , y);
        cost = 1 / (N * 2) * (Vecteur.sumValues(Vecteur.square(error)));
        return cost;
    }

}
