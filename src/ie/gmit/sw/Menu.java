package ie.gmit.sw;

import org.encog.Encog;
import org.encog.engine.network.activation.*;

import java.awt.*;
import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;
import java.util.concurrent.TimeUnit;

public class Menu extends Component {
    private Scanner s = new Scanner(System.in);
    private boolean running = true;
    private ActivationFunction activationFunction = new ActivationElliottSymmetric();
    private int ngram = 2;
    private int vector = 500;
    private int epochs = 9;
    private Map<String, double[]> liveTestDataDir = new HashMap<>();
    private double [] liveTestDatafile;
    private String filepath = "./data" + vector + ".csv";

    public void doMenu() throws InterruptedException {
        while (running){
            String in;
            int option;
            printOptions();
            do{
                System.out.println("Only numeric values accepted..");
                in = s.next().toLowerCase();
            }while (in.isEmpty() || !in.matches("[1-9]"));
                option = Integer.parseInt(in);
                switch (option){
                    case 1:
                        getUserInputNgram();        // reset default ngram size
                        getUserInputVector();       // reset default vector size
                        getUserInputActFunction();  // reset default activation function
                        getUserInputEpoch();        // reset default epochs
                        printNetworkPreferences();
                        do{
                            System.out.println("Return to main Menu..? -> y/n");
                            in = s.next().toLowerCase();
                        }while (in.isEmpty() || !in.matches("[y]"));
                        break;
                    case 2: // Build and run (train test evaluate), and save neural network.
                        System.out.println("Building csv from Wili.txt file");
                        VectorProcessor defaultVP = new VectorProcessor(ngram, vector);
                        defaultVP.go();
                        // Sleep for a few seconds while the file generates.
                        TimeUnit.SECONDS.sleep(2);
                        System.out.println("Building Neural Network with:");
                        printNetworkPreferences();
                        String csvFile = "data"+vector+".csv";
                        System.out.println("Using csv: "+csvFile);
                        NN defaultNetwork = new NN(epochs);
                        defaultNetwork.trainNewNetwork(activationFunction, vector, csvFile);
                        Encog.getInstance().shutdown();
                        do{
                            System.out.println("Return to main Menu..? -> y/n");
                            in = s.next().toLowerCase();
                        }while (in.isEmpty() || !in.matches("[y]"));
                        break;
                    case 3:
                            do{
                                System.out.println("You must have created a Neural Network prior to testing data.\n" +
                                    "Do you wish to continue..? -> y/n");
                                in = s.next().toLowerCase();
                            }while (in.isEmpty() || !in.matches("[yn]"));
                        if (in.equalsIgnoreCase("y")) {
                            getUserInputNgram();
                            getUserInputVector();
                            getUserInputEpoch();
                            String testFile = getUserinputFile();
                            VectorProcessor userVP = new VectorProcessor(ngram, vector, testFile);
                            userVP.go();
                            liveTestDatafile = userVP.getLiveData();
                            NN userSingleTest = new NN(epochs);
                            userSingleTest.testExistingNetwork("test.nn", liveTestDatafile, testFile);
                            Encog.getInstance().shutdown();
                        }
                        do{
                            System.out.println("Return to main Menu..? -> y/n");
                            in = s.next().toLowerCase();
                        }while (in.isEmpty() || !in.matches("[y]"));
                        break;
                    case 4:
                        do{
                            System.out.println("You must have created a Neural Network prior to testing data.\n" +
                                    "Do you wish to continue..? -> y/n");
                            in = s.next().toLowerCase();
                        }while (in.isEmpty() || !in.matches("[yn]"));
                        if (in.equalsIgnoreCase("y")){
                            getUserInputNgram();
                            getUserInputVector();
                            getUserInputEpoch();
                            String directoryPath = getUserinputFile();
                            File [] testDirectory = new File(directoryPath).listFiles();
                            for (File file : testDirectory) {
                                VectorProcessor vp = new VectorProcessor(ngram, vector, file.getPath());
                                vp.go();
                                double [] dd = vp.getLiveData();
                                liveTestDataDir.put(file.getName(), dd);
                            }

                            NN nn = new NN(epochs);
                            // using for-each loop for iteration over Map.entrySet()
                            for (Map.Entry<String,double[]> e : liveTestDataDir.entrySet()){
                                nn.testExistingNetwork("test.nn",e.getValue(), e.getKey());
                            }

                            Encog.getInstance().shutdown();
                        }
                        do{
                            System.out.println("Return to main Menu..? -> y/n");
                            in = s.next().toLowerCase();
                        }while (in.isEmpty() || !in.matches("[y]"));
                        break;
                    case 5: printTopologyInfo();
                        do{
                            System.out.println("Return to main Menu..? -> y/n");
                            in = s.next().toLowerCase();
                        }while (in.isEmpty() || !in.matches("[y]"));
                        break;
                    case 6: printTrainingInfo();
                        do{
                            System.out.println("Return to main Menu..? -> y/n");
                            in = s.next().toLowerCase();
                        }while (in.isEmpty() || !in.matches("[y]"));
                        break;
                    case 7: printTestingInfo();
                        do{
                            System.out.println("Return to main Menu..? -> y/n");
                            in = s.next().toLowerCase();
                        }while (in.isEmpty() || !in.matches("[y]"));
                        break;
                    case 8: printVectorHashingInfo();
                        do{
                            System.out.println("Return to main Menu..? -> y/n");
                            in = s.next().toLowerCase();
                        }while (in.isEmpty() || !in.matches("[y]"));
                        break;
                    case 9:
                        do{
                            System.out.println("Exit Application..? -> y/n");
                            in = s.next().toLowerCase();
                        }while (in.isEmpty() || !in.matches("[yn]"));
                        if (in.equalsIgnoreCase("y")) {
                            running = false;
                            System.out.println("Exiting application.");
                        }
                        break;
                    default: System.out.println("Invalid Option");
                        break;
                }
        }
    }

    /**
     * Print menu options.
     */
    private void printOptions(){
        System.out.println("Language Detection Neural Network with Vector Hashing");
        System.out.println("================= DAVID GALLAGHER ===================");
        System.out.println("-------- Select one of the following options --------");
        System.out.println("1 - Customize Neural Network preferences.");
        System.out.println("2 - Build Neural Network with custom/default preferences.");
        System.out.println("3 - Test single live data file against the Network.");
        System.out.println("4 - Test directory containing test/live data against the Network.");
        System.out.println("5 - View network TOPOLOGY information.");
        System.out.println("6 - View network TRAINING information.");
        System.out.println("7 - View network TESTING  information.");
        System.out.println("8 - View generating CSV   information.");
        System.out.println("9 - Quit the program:");
    }
    /**
     * Allow user to choose activation function.
     * Perform minimal error handling for typos. (Not intended to be exhaustive).
     */
    private void getUserInputActFunction(){
        String in;
        //user select activation function
        do {
            System.out.println("Specify Activation function - 1-6.");
            System.out.println(" 1 - ElliottSymmetric (Best).");
            System.out.println(" 2 - LOG.");
            System.out.println(" 3 - TANH.");
            System.out.println(" 4 - ClippedLinear.");
            System.out.println(" 5 - BipolarSteepenedSigmoid.");
            System.out.println(" 6 - BiPolar.");
            in = s.next();
        }while (in.isEmpty() || !in.matches("[1-6]"));
        this.activationFunction = setActivation(Integer.parseInt(in));
        System.out.println("Act F: \n" + activationFunction.getLabel());
    }
    /**
     * Allow user to choose ngram size.
     * Perform minimal error handling for typos. (Not intended to be exhaustive).
     */
    private void getUserInputNgram(){
        //User enter ngrams
        String in;
        do {
            System.out.println("Specify ngram size - Recommended size is 2.");
            in = s.next();
        }while(in.isEmpty() || !in.matches("\\d"));
        this.ngram = Integer.parseInt(in);
        System.out.println("ngram: \n"+ngram);
    }
    /**
     * Allow user to choose vector size.
     * Perform minimal error handling for typos. (Not intended to be exhaustive).
     */
    private void getUserInputVector(){
        //user enter vectorsize
        String in;
        do {
            System.out.println("Specify vector size - Recommended size is 500.");
            in = s.next();
        }while(in.isEmpty() || !in.matches("\\d+"));
        this.vector = Integer.parseInt(in);
        System.out.println("vector: \n"+vector);
    }
    /**
     * User enter file for test data to be vectorised.
     * @return file test data.
     */
    private String getUserinputFile(){

        String selectedFile = null;
        //user enter option
        String in;
        do {
            System.out.println("Enter 1 for GUI select file\nEnter 2 to type filepath");
            in = s.next();
        }while(in.isEmpty() || !in.matches("\\d+"));
        int selection = Integer.parseInt(in);
        switch (selection) {
            case 1:
                    selectedFile = Utilities.getFileWithGUI();
                break;
            case 2:
                System.out.println("Type filepath: \n");
                in = s.next();
                selectedFile = in;
                break;
            default: break;
        }
        return  selectedFile;
    }
    private void getUserInputEpoch(){
        //user enter vectorsize
        String in;
        do {
            System.out.println("Specify epochs size - Recommended size is 9.");
            in = s.next();
        }while(in.isEmpty() || !in.matches("[1-9]"));
        this.epochs = Integer.parseInt(in);
        System.out.println("epochs: \n"+epochs);
    }
    /**
     * Print Topology Information.
     */
    private void printTopologyInfo(){
        System.out.println("Current Neural Network Topology Information");
        System.out.println("This network consists of 3 layers.");
        System.out.println("Input layer.\nNo activation function.\nHas bias - True.\nReceives "+vector+" input nodes, 500 is the " +
                "recommended size. May be changed by user input.");
        System.out.println("One hidden layer.\n"+activationFunction.getLabel().toUpperCase()+" activation function. May be changed by user input.\nHas bias - True." +
                "\nUses Geometric Pyramid Rule to calculate number of nodes.\nDropoff calculated as (0.8 * input vector)");
        System.out.println("One output layer.\nSoftmax activation function.\nHas bias - False." +
                "\n235 output nodes representing all languages present in dataset.");
    }
    /**
     * Print Training Information.
     */
    private void printTrainingInfo(){
        System.out.println("Neural Network Training Information");
        System.out.println("Training set is loaded in from a csv file.");
        System.out.println("This data is added to a folded data set.");
        System.out.println("The folded data set is added to a cross validation k fold.\n" +
                "- The cross validation k fold runs iterations over the data set for a" +
                " (currently) predetermined number of epochs to reduce the overall error.\n" +
                "The training is optimized for a ngram of size 2 and a vector of size 500.");

    }
    /**
     * Print Testing Information.
     */
    private void printTestingInfo(){
        System.out.println("Neural Network Testing Information");
        System.out.println("The testing examines the folded data and returns a comparison of" +
                " the expected result, which the neural net is aiming to achieve with its prediction" +
                ", and the actual result which the neural net predicts is the correct outcome.");
    }
    /**
     * Print Vector Hashing Information.
     */
    private void printVectorHashingInfo(){
        System.out.println("Neural Network Vector Hashing Information");
        System.out.println("The application reads in a text file with rows of text followed by a language identifier.\n" +
                "This is processed into a vector array and normalised to values of between a range, in this case of 0-1.");
    }
    /**
     * User chosen Activation function for the Network
     * @param choice function selector
     * @return selected function
     */
    private ActivationFunction setActivation(int choice){

        ActivationFunction activationFunction = null;
        switch (choice){
            case 1:
                activationFunction = new ActivationElliottSymmetric();
                break;
            case 2:
                activationFunction = new ActivationLOG();
                break;
            case 3:
                activationFunction = new ActivationTANH();
                break;
            case 4:
                activationFunction = new ActivationClippedLinear();
                break;
            case 5:
                activationFunction = new ActivationBipolarSteepenedSigmoid();
                break;
            case 6:
                activationFunction = new ActivationBiPolar();
                break;
            default:
                activationFunction = new ActivationElliottSymmetric();
                break;
        }
        return activationFunction;
    }

    private void printNetworkPreferences(){
        System.out.println("Preferences Set:\nNgram: " + ngram +"\nVector: " +vector
                +"\nEpocs: "+epochs+"\nActivation Function: "+activationFunction.getLabel().toUpperCase());
    }

}
