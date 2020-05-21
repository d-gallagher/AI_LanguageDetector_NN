package ie.gmit.sw;

import java.awt.*;
import java.io.*;
import java.text.DecimalFormat;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class VectorProcessor extends Component {

    private final int ALL_LANGS = 235;
    private File wili_file = new File("./wili-2018-Small-11750-Edited.txt");
    private File user_file = null;
    private Language[] langs = Language.values();
    private int ngramSize;
    private int arraySize;
    private boolean isUser;
    private double [] liveData = new double[arraySize];
    int lineLenTotal = 0;
    int shortestLine = 1000000;
    int longestLine = -1;
    int totalLines = 0;

    public VectorProcessor(int ngramSize, int arraySize) {
        this.ngramSize = ngramSize;
        this.arraySize = arraySize;
        this.isUser = false;
    }

    public VectorProcessor(int ngramSize, int arraySize, File userFile) {
        this.ngramSize = ngramSize;
        this.arraySize = arraySize;
        this.isUser = true;
        this.user_file = userFile;
    }

    public void go(){
        ExecutorService es = Executors.newFixedThreadPool(2);
        File inFile = wili_file;
        if (isUser) inFile = new File (user_file.getName());
        try(BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(inFile)))){
            String line;
            while ((line = br.readLine()) != null){
                String finalLine = line;
                // Fire threads at the processing to decrease time
                Runnable runnable = () -> {
                    try {
                        if (!isUser){processWiliTxt(ngramSize, finalLine);}
//                        else{System.out.println("Processing ");}
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                };
                es.submit(runnable);
                if (isUser) liveData = getNormalizedVector(ngramSize, line);
            }
            // Shutdown Executor Service
            es.shutdown();
        }catch (Exception e){
            System.out.println("go exception "+e.toString());
        }

    }

    public double[] getLiveData() {
        return liveData;
    }

    /**
     * Take a line of text and process into ngrams.
     * Add to double array.
     * Generate second array (langValues) of 235 'Zeroes',
     *  -> except from a single 'One' representing expected language.
     * @param ngram
     * @param line
     */
    public void processWiliTxt(int ngram, String line) throws IOException {

        // Hold text values for text and language
        String text = "";
        String lang = "";

        // Split the input into Language text and Language Identifier
        String [] record = line.split("@");
        //System.out.println("Splitting a line..");
        if (record.length >2) return; // skip any line with no '@' symbol

        // Language text
        text = record[0].toUpperCase();

        // Language Identifier
        lang = record[1].toUpperCase();

        // Set the vector array
        double [] vector = getNormalizedVector(ngram, text);

        // Array of languages, see comments.
        double[] allLangValues = getLangs(lang);

//        printDArray(allLangValues); // Testing
        outputCsvFile(vector, allLangValues, arraySize);
    }//process



    /**
     * Take the array of vector values and the array of Languages
     * Output a file with all values in one row of a csv
     * @param arr1
     * @param langs
     * @throws IOException
     */
    public void outputCsvFile(double [] arr1, double [] langs, int num) throws IOException {
        DecimalFormat df = new DecimalFormat("###.###");
        FileWriter fileWriter;
        BufferedWriter bufferedWriter;
        String fileName = "";

        // filename = data + array/vector size .csv to easily id inputs for NN during manual testing
        fileName = "data" + num + ".csv";
        File TO_CSV_File = new File(fileName);
        fileWriter = new FileWriter(TO_CSV_File, true);
        bufferedWriter = new BufferedWriter(fileWriter);

        for (int i = 0; i < arr1.length; i++) {
            bufferedWriter.write(df.format(arr1[i]) + ", ");
        }

        for (int i = 0; i < langs.length; i++) {
            bufferedWriter.write(langs[i] + ", ");
        }

        bufferedWriter.newLine();
        bufferedWriter.close();
    }

    /**
     * Build array of languages where the identified lang is assigned a value of 1.
     * All other indexes are assigned a 0 (Zero).
     * @param lang index of this lang is assigned a 1
     * @return array of languages as 0's and a 1 for the language 'lang'
     */
    private double [] getLangs(String lang){
        double [] temp = new double[ALL_LANGS];
        for (int i = 0; i < langs.length; i++) {
            if (lang.equalsIgnoreCase(String.valueOf(langs[i]))) {
                temp[i] = 1;
//                System.out.println(i +" FOUND THE LANGUAGE " + allLangValues[i]); // Testing
            } else temp[i] = 0;
        }
        return temp;
    }

    /**
     * Take a ngram size and string input.
     * Convert the string into a vectorized array of ngrams.
     * @param ngram size of ngram.
     * @param text string to be split into ngrams.
     * @return vectorized array.
     */
    private double [] getNormalizedVector(int ngram, String text){
        // Array to hold normalized vector values
        double [] temp = new double[arraySize];
        // Iterate over the text and split into ngrams
        NgramProducer ng = new NgramProducer(ngram, text);
        while (ng.hasNext() && ng.getCount() != temp.length - ngram){
            int kmer = ng.hashCode();
            int vhash = kmer%temp.length;
            temp[vhash]++;
            //System.out.println("Vector "+ng.getCount()+": " + vector[ng.getCount()]);
            //System.out.println(ng.getCount());
        }
        temp = Utilities.normalize(temp, 0, 1);
        return temp;
    }

    /**
     * Print the contents of an array of Doubles
     * @param d array to print
     */
    public void printDArray(double[] d){
        for (int i = 0; i < d.length; i++) {
            System.out.print(" : " + d[i]+" : ");
        }
        System.out.println();
    }

    public static void main(String[] args) throws IOException {
//        int arraysize = 50;
//        for (int i = 0; i < 10; i++) {
//        File userFile = new File("engTest.txt");
//        VectorProcessor vp = new VectorProcessor(3, 200, userFile);
//        vp.go();
//        double [] livedata = vp.getLiveData();
//        vp.printDArray(livedata);
//        System.out.println(livedata.length);
//        Files.createDirectory(Paths.get("7gram"));
//            arraysize += 50;
//        System.out.println(vp.wili_file.getName());
//        }
    }


}
