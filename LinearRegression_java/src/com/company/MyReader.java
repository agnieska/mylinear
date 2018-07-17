package com.company;

import javax.swing.tree.RowMapper;
import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class MyReader {

    public static String readFile(File f) {
        BufferedInputStream inputStream = null;
        StringWriter outputWriter = null;

        try {
            inputStream = new BufferedInputStream(new FileInputStream(f));
            outputWriter = new StringWriter();
            int b;
            while ((b = inputStream.read()) != -1)

                outputWriter.write(b);
            outputWriter.flush();
            outputWriter.close();
            inputStream.close();

        } catch (FileNotFoundException e) {
            System.out.print("Couldn't find file " + f);
        } catch (IOException ie) {
            System.out.print("Cant read file or line");
        }
        return outputWriter.toString();
    }

    private static String[] load_And_ReadFile(String inputfilename) {
        String filecontent = null;
        try {
            File file = new File(inputfilename);
            filecontent = readFile(file);

        } catch (NullPointerException e) {
            System.out.print("The file " + inputfilename + " is null. ");
        }
        String[] lines = filecontent.split("\n");
        return lines;
    }


    public static Map<String, ArrayList<Double>> read_csv(String inputfilename, String delimiter, boolean header) {


        String[] lines  =  load_And_ReadFile(inputfilename);
        ArrayList<Double> columnX = new ArrayList<Double>();
        ArrayList<Double> columnY = new ArrayList<Double>();
        int l = 0;
        if (header == true) {
            String[] titles = lines[l].split(delimiter);
            l++;
        } else {
            String[] titles = new String[lines.length];
            for (int i =0; i < titles.length; i++)
                titles[i] = i+"";
        }
        while (l < lines.length) {
            String line = lines[l];
            String[] words = line.split(delimiter);
            columnX.add(Double.parseDouble(words[0]));
            columnY.add(Double.parseDouble(words[1]));
        }

        Map<String, ArrayList<Double>> dataframe = new HashMap<String, ArrayList<Double>>();
        dataframe.put("X", columnX);
        dataframe.put("Y", columnY);
        return dataframe;
    }


    public static Map<String, ArrayList<Double>> read_csv(String inputfilename) {
        return read_csv(inputfilename, ",", false);
    }



}
