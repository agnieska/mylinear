package com.company;

public class Vecteur {

    public static float[] addition (float[] col1, float[] col2){

        if (col1.length == col2.length){
             float[] result = new float[col1.length];
             for(int i = 0; i < result.length; i++)
                result[i] = col1[i] + col2[i];
             return result;
        } else
            return null;
    }

    public static float[] soustraction (float[] col1, float[] col2){

        if (col1.length == col2.length){
            float[] result = new float[col1.length];
            for(int i = 0; i < result.length; i++)
                result[i] = col1[i] - col2[i];
            return result;
        } else
            return null;
    }


    public static float[] multiply (float[] col1, float[] col2){

        if (col1.length == col2.length){
            float[] result = new float[col1.length];
            for(int i = 0; i < result.length; i++)
                result[i] = col1[i] * col2[i];
            return result;
        } else
            return null;
    }
    public static float[] multiply (float[] col1, float c){

            float[] result = new float[col1.length];
            for(int i = 0; i < result.length; i++)
                result[i] = col1[i] * c;
            return result;

    }

    public static float[] divide (float[] col1, float[] col2){

        if (col1.length == col2.length){
            float[] result = new float[col1.length];
            for(int i = 0; i < result.length; i++) {
                if (col2[i] != 0)
                    result[i] = col1[i] / col2[i];
                else
                    result[i] = Math.signum(col1[i]) * (Float.POSITIVE_INFINITY - 1);
            }
            return result;
        } else
            return null;
    }
    public static float[] divide (float c, float[] col2){

            float[] result = new float[col2.length];
            for(int i = 0; i < result.length; i++) {
                if (col2[i] != 0)
                    result[i] = c / col2[i];
                else
                    result[i] = Math.signum(c) * (Float.POSITIVE_INFINITY - 1);
            }
            return result;
    }

    public static float[] square (float[] col1){
        return multiply(col1, col1);
    }

    public static float sumValues (float[] col1){
            float result = 0;
            for(int i = 0; i < col1.length; i++)
                result = result + col1[i];
            return result;

    }

    public static float dot (float[] col1, float[] col2 ){

        return sumValues(multiply(col1, col2));
    }

    public static float produit (float[] col1, float[] col2 ){

        return sumValues(multiply(col1, col2));
    }

}
