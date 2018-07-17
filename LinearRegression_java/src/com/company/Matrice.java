package com.company;


public class Matrice {

    private int ligne;
    private int colonne;

    private double coefficients[][];

    //ce multiple ne sert que pour l'affichage de matrice, lors de l'affichage d'inverse par exemple
    private double multiple;

    public Matrice(int ligne1, int colonne1) {
        ligne = ligne1;
        colonne = colonne1;
        coefficients = new double[ligne][colonne];
        multiple = 1;
    }

    //ici il faudra générer une exception si les lignes/colonnes ne sont pas compatibles
    public void remplir(double coefficients1[][]) throws Exception {
        if (coefficients1.length != ligne) {
            throw new Exception("incompatibilite de genre");
        }
        for (int a = 0; a < coefficients1.length; a++) {
            if (coefficients1[a].length != colonne)
                throw new Exception("incompatibilite de genre");
        }
        coefficients = coefficients1;
    }

    public void remplir(int numligne, int numcolonne, double coefficients1) {
        coefficients[numligne][numcolonne] = coefficients1;
    }

    public String toString() {
        String retour = "";
        if (multiple != 1)
            retour = "1/" + multiple + "\n\n  x\n\n";
        for (int a = 0; a < ligne; a++) {
            retour += "\n( ";
            for (int b = 0; b < colonne; b++) {
                retour += coefficients[a][b] + " ";
            }
            retour += ")";
        }

        return retour;
    }

    //générer exception si pas de même type
    public Matrice somme(Matrice B) throws Exception {

        if (ligne != B.getlig() || colonne != B.getcol())
            throw new Exception("incompatibilite de genre");

        Matrice C = new Matrice(ligne, colonne);
        for (int a = 0; a < ligne; a++) {
            for (int b = 0; b < colonne; b++) {
                C.remplir(a, b, ((coefficients[a][b] / multiple) + (B.getcoef(a, b) / B.getmultiple())));
            }
        }
        return C;
    }


    //générer une exception si non compatible
    public Matrice produit(Matrice B) throws Exception {

        if (colonne != B.getlig())
            throw new Exception("incompatibilite de genre");

        Matrice C = new Matrice(ligne, B.getcol());

        for (int a = 0; a < C.getlig(); a++) {
            for (int b = 0; b < C.getcol(); b++) {

                double coeftemp = 0;
                for (int c = 0; c < colonne; c++) {
                    coeftemp += coefficients[a][c] * B.getcoef(c, b);
                }
                C.remplir(a, b, coeftemp);
            }
        }

        C.setmultiple(multiple * B.getmultiple());
        C.arrondir();

        return C;
    }

    public Matrice produitreel(double reel) {
        Matrice C = new Matrice(ligne, colonne);

        double reel2 = reel / multiple;

        for (int a = 0; a < ligne; a++) {
            for (int b = 0; b < colonne; b++) {
                C.remplir(a, b, (reel2 * coefficients[a][b]));
            }
        }

        return C;
    }

    //calcul de l'inverse par produit de l'adjointe de la matrice avec l'inverse du déterminant de la matrice
    public Matrice inverser() throws Exception {
        double determinant = determinant();

        if (determinant == 0)
            throw new Exception("Matrice non inversible");

        Matrice inverse = new Matrice(ligne, colonne);

        inverse = adjointe().produitreel((1.0 / determinant));

        inverse = inverse.produitreel(multiple);

        boolean besoinmultiple = false;

        for (int a = 0; a < inverse.getlig() && !besoinmultiple; a++) {
            for (int b = 0; b < inverse.getcol() && !besoinmultiple; b++) {
                if ((int) inverse.getcoef(a, b) != inverse.getcoef(a, b)) {
                    besoinmultiple = true;
                }
            }
        }

        if (besoinmultiple) {
            Matrice inverse2 = new Matrice(ligne, colonne);
            for (int a = 0; a < inverse.getlig(); a++) {
                for (int b = 0; b < inverse.getcol(); b++) {
                    inverse2.remplir(a, b, (inverse.getcoef(a, b) * determinant));
                }
            }
            inverse2.setmultiple(determinant);
            inverse = inverse2;
        }

        inverse.arrondir();

        return inverse;
    }

    public Matrice puissance(int exposant) {
        if (exposant < 1)
            return null;

        try {
            Matrice C = new Matrice(ligne, colonne);
            C.remplir(coefficients);

            for (int a = 1; a < exposant; a++) {
                C = C.produit(C);
            }

            C.setmultiple(Math.pow(multiple, (double) exposant));

            return C;
        } catch (Exception e) {
            return null;
        }
    }

    //sert au calcul de l'inverse, c'est la transposée de la matrice des cofacteurs
    public Matrice adjointe() {
        try {
            if (ligne != colonne) {
                return null;
            }

            Matrice adjointe = new Matrice(ligne, colonne);

            for (int a = 0; a < ligne; a++) {
                for (int b = 0; b < colonne; b++) {
                    if (((a + b) % 2) == 0)
                        adjointe.remplir(a, b, (matsansligcol(a, b).determinant()));
                    if (((a + b) % 2) == 1)
                        adjointe.remplir(a, b, -(matsansligcol(a, b).determinant()));
                }
            }

            adjointe = adjointe.transposer();

            return adjointe;
        } catch (Exception e) {
            return null;
        }
    }

    public Matrice transposer() {
        Matrice C = new Matrice(colonne, ligne);
        for (int a = 0; a < colonne; a++) {
            for (int b = 0; b < ligne; b++) {
                C.remplir(a, b, coefficients[b][a]);
            }
        }
        C.setmultiple(multiple);
        return C;
    }

    //générer erreur si non carrée, méthode bezout sur première ligne suivie par sarrus
    public double determinant() throws Exception {

        if (ligne != colonne)
            throw new Exception("Matrice non inversible");

        produitreel(1.0 / multiple);
        multiple = 1;

        if (ligne != colonne)
            return 0;

        if (ligne == 1)
            return coefficients[0][0] * multiple;

        if (ligne == 2)
            return determinant2x2() * multiple;

        if (ligne == 3)
            return determinant3x3() * multiple;

        double determinant = 0;

        for (int a = 0; a < colonne; a++) {
            if (a % 2 == 0) {
                determinant += coefficients[0][a] * matsansligcol(0, a).determinant();
            } else {
                determinant -= coefficients[0][a] * matsansligcol(0, a).determinant();
            }
        }

        return determinant;
    }

    //méthode utilisée pour le calcul de déterminant
    public Matrice matsansligcol(int ligne1, int colonne1) {
        Matrice C = new Matrice((ligne - 1), (colonne - 1));

        int compteurL = -1;
        int compteurC = -1;

        for (int a = 0; a < ligne; a++) {
            if (a != ligne1) {
                compteurL++;
                for (int b = 0; b < colonne; b++) {
                    if (b != colonne1) {
                        compteurC++;
                        C.remplir(compteurL, compteurC, coefficients[a][b]);
                    }
                }
                compteurC = -1;
            }
        }

        return C;

    }

    //par la règle de Sarrus :
    public double determinant3x3() {
        if (ligne != 3 || colonne != 3)
            return 0;

        double determinant = coefficients[0][0] * coefficients[1][1] * coefficients[2][2];
        determinant += coefficients[0][1] * coefficients[1][2] * coefficients[2][0];
        determinant += coefficients[0][2] * coefficients[1][0] * coefficients[2][1];
        determinant -= coefficients[2][0] * coefficients[1][1] * coefficients[0][2];
        determinant -= coefficients[1][0] * coefficients[0][1] * coefficients[2][2];
        determinant -= coefficients[0][0] * coefficients[2][1] * coefficients[1][2];

        return determinant;
    }

    public double determinant2x2() {
        if (ligne != 2 || colonne != 2)
            return 0;

        double determinant = coefficients[0][0] * coefficients[1][1] - coefficients[0][1] * coefficients[1][0];
        return determinant;
    }

    public double[][] getcoef() {
        return coefficients;
    }

    //générer une exception si ligne ou colonne non encodée
    public double getcoef(int ligne1, int colonne1) {
        return coefficients[ligne1][colonne1];
    }

    public int getcol() {
        return colonne;
    }

    public int getlig() {
        return ligne;
    }

    public void setmultiple(double multiple1) {
        multiple = multiple1;
    }

    public double getmultiple() {
        return multiple;
    }

    //arrondit les valeurs très proches de 0 à 0 et supprimes les décimales très petites
    public void arrondir() {

        boolean divparmultiple = true;

        for (int a = 0; a < ligne; a++) {
            for (int b = 0; b < colonne; b++) {
                if (Math.abs(coefficients[a][b]) < Math.pow(10, -10))
                    coefficients[a][b] = 0;

                if (Math.abs(coefficients[a][b] - (int) coefficients[a][b]) < Math.pow(10, -10))
                    coefficients[a][b] = (int) coefficients[a][b];

                if (coefficients[a][b] % multiple != 0)
                    divparmultiple = false;

                if ((Math.abs(coefficients[a][b] - (int) coefficients[a][b]) + Math.pow(10, -10)) > 1) {
                    if (coefficients[a][b] > 0)
                        coefficients[a][b] = ((int) coefficients[a][b]) + 1;
                    else if (coefficients[a][b] < 0)
                        coefficients[a][b] = ((int) coefficients[a][b]) - 1;
                }

            }
        }

        if (divparmultiple) {
            for (int a = 0; a < ligne; a++) {
                for (int b = 0; b < colonne; b++) {
                    if (coefficients[a][b] != 0)
                        coefficients[a][b] = coefficients[a][b] / multiple;
                }
            }
            multiple = 1;
        }
    }

    //application du pivot de Gauss
    public Matrice echelonner(int etape1) {
        try {
            //je crée une copie, c'est celle-là que l'on va échelonner
            Matrice C = copy();

            int etape = etape1;

            //ici je permute jusqu'à avoir un pivot non nul mais de préférence 1
            boolean pivotOK = false;

            if (Math.abs(C.getcoef(etape, etape)) == 1)
                pivotOK = true;

            for (int a = (etape + 1); a < C.getlig() && !pivotOK; a++) {
                C.permuter(etape, a);
                if (Math.abs(C.getcoef(etape, etape)) == 1)
                    pivotOK = true;
            }

            if (C.getcoef(etape, etape) != 0)
                pivotOK = true;

            for (int a = (etape + 1); a < C.getlig() && !pivotOK; a++) {
                C.permuter(etape, a);
                if (C.getcoef(etape, etape) != 0)
                    pivotOK = true;
            }

            Matrice matpivot = C.matlig(etape);
            matpivot = matpivot.simplifier();
            for (int a = 0; a < colonne; a++) {
                C.remplir(etape, a, matpivot.getcoef(0, a));
            }
            for (int a = (etape + 1); a < ligne; a++) {
                Matrice matligne = C.matlig(a);
                double rapport = ((-1.0) * matligne.getcoef(0, etape)) / (matpivot.getcoef(0, etape));
                String txt = "" + rapport;
                if (txt.equals("NaN")) {
                    etape = ligne;
                    break;
                }
                //matligne = (matligne.somme(matpivot.produitreel(rapport)));
                matligne = (matligne.somme(matpivot.produitreel(rapport))).simplifier();
                for (int b = 0; b < colonne; b++) {
                    C.remplir(a, b, matligne.getcoef(0, b));
                }
            }

            etape++;
            if (etape < (ligne - 1))
                C = C.echelonner(etape);

            return C;
        } catch (Exception e) {
            return null;
        }
    }

    public Matrice copy() {
        Matrice C = new Matrice(ligne, colonne);

        for (int a = 0; a < ligne; a++) {
            for (int b = 0; b < colonne; b++) {
                C.remplir(a, b, coefficients[a][b]);
            }
        }

        return C;
    }

    //permute 2 lignes, sert à echelonner
    public void permuter(int L1, int L2) {
        double lig1;
        double lig2;

        for (int a = 0; a < colonne; a++) {
            lig1 = coefficients[L1][a];
            lig2 = coefficients[L2][a];
            coefficients[L1][a] = lig2;
            coefficients[L2][a] = lig1;
        }
    }

    public Matrice matlig(int ligne1) {

        Matrice C = new Matrice(1, colonne);

        for (int a = 0; a < colonne; a++) {
            C.remplir(0, a, coefficients[ligne1][a]);
        }

        return C;
    }

    //calcul du pgcd par l'algorithme d'euclide : a = qb + r
    static public int pgcd(int a1, int b1) {
        int a;
        int b;

        if (a1 < b1) {
            a = b1;
            b = a1;
        } else {
            a = a1;
            b = b1;
        }

        if (b == 0)
            return a;

        int reste = 1;
        while (reste != 0) {
            int q = 0;
            while ((b * q) < a) {
                q++;
            }
            if ((b * q) != a)
                q--;
            reste = a - (q * b);
            a = b;
            b = reste;
        }
        return a;
    }

    //cette méthode sert a simplifier les matrices lignes lors de l'échelonnement et donc d'avoir une matrice plus simple
    public Matrice simplifier() {
        int pgcd = 0;
        boolean lignenonnulle = false;
        for (int a = 0; a < colonne; a++) {
            if ((int) coefficients[0][a] != coefficients[0][a]) {
                pgcd = 1;
                a = colonne;
                break;
            }
            pgcd = pgcd(Math.abs((int) coefficients[0][a]), pgcd);
            if (coefficients[0][a] != 0)
                lignenonnulle = true;
        }

        if (!lignenonnulle)
            pgcd = 1;

        Matrice C = produitreel(1.0 / pgcd);
        return C;
    }

}


