package ie.gmit.sw;

import java.io.File;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.concurrent.TimeUnit;

import org.encog.Encog;
import org.encog.app.analyst.EncogAnalyst;
import org.encog.engine.network.activation.*;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.data.buffer.MemoryDataLoader;
import org.encog.ml.data.buffer.codec.CSVDataCODEC;
import org.encog.ml.data.buffer.codec.DataSetCODEC;
import org.encog.ml.data.folded.FoldedDataSet;
import org.encog.ml.data.versatile.NormalizationHelper;
import org.encog.ml.train.MLTrain;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.cross.CrossValidationKFold;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.util.csv.CSVFormat;
import org.encog.util.simple.EncogUtility;


/**
 * NN - Modular Neural Network Class to allow building a new or testing an existing Neural Network.
 * * trainNewNetwork      - Build, Train, Test and Evaluate a network with the Wili.txt Dataset.
 * * testExistingNetwork  - Provide a dataset and Evaluate an existing network.
 */
public class NN {

    // An output exists for each language in the Language set
    final int LANGUAGES = 235;
    private final Language[] langs = Language.values(); //Only call this once...


    // Variables to calculate time elapsed and error difference
    private DecimalFormat df = new DecimalFormat("###.#######");
    private long totalTime = 0L;
    private long startTime = 0L;
    private long endTime = 0L;
    private long period = 0L;
    private double lastErVal = 0;
    private double currErVal;


    public NN(){
    }

    /**
     * Build a Neural Network with the given user options.
     * @param activationFunction Activation Function chosen by user.
     * @param vectorSize Vector size (input nodes) chosen by user.
     * @return Basic 3 layer Neural Network.
     */
    private BasicNetwork buildNetwork(ActivationFunction activationFunction, int vectorSize){
        // === SET NETWORK ATTRIBUTES ===
        // Geometric Pyramid Rule to calculate hidden layer nodes
        double hiddenGPR = Math.sqrt(vectorSize * LANGUAGES);
        // recommended dropout rate of 0.8 https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/
        int dropoutRate = (int) (hiddenGPR*0.8);

        // === BUILD NETWORK ===
        BasicNetwork network = new BasicNetwork();
        network.addLayer(new BasicLayer(null, true, vectorSize));
        network.addLayer(new BasicLayer(activationFunction, true, (int) hiddenGPR));
        network.addLayer(new BasicLayer(new ActivationSoftMax(), false, LANGUAGES));
        network.getStructure().finalizeStructure();
        network.reset();
        System.out.println("Network Built.");
        return network;
    }

    /**
     * Take a pre-generated vector hashed file input in csv format and convert to a ML dataset.
     * @param dataFilePath input csv file path.
     * @param vectorSize amount of input node columns in file.
     * @return ML dataset for use in network training/testing/evaluation.
     */
    private MLDataSet buildDataset(String dataFilePath, int vectorSize){
        //========================= Generate Training Set ==============================
        File dataFile = new File((dataFilePath));
//        dataFile = Utilities.getFileWithString(dataFilePath);
        //Read the CSV file "data.csv" into memory. Encog expects your CSV file to have input + output number of columns.
        DataSetCODEC dsc = new CSVDataCODEC(dataFile, CSVFormat.ENGLISH, false, vectorSize, LANGUAGES, false);
        MemoryDataLoader mdl = new MemoryDataLoader(dsc);
        MLDataSet dataSet = mdl.external2Memory();
        System.out.println("Dataset Built.");
        return dataSet;
    }

    /**
     * Take a dataset and train the network against the set using cross validation (hardcoded to 5 partitions).
     * Used to produce a Neural Network capable of testing and evaluation against other datasets.
     * Saves self to .nn file.
     * @param trainingSet
     * @param network
     */
    private void trainDataset(MLDataSet trainingSet, BasicNetwork network){
        //========================= Train Training Set ==============================
        //----- CrossValidation 5 partitions ---
        FoldedDataSet folded = new FoldedDataSet(trainingSet);
        MLTrain train = new ResilientPropagation(network, folded);
        CrossValidationKFold crossValidationKFold = new CrossValidationKFold(train, 5);
        System.out.println("Training Dataset.");
        //========= Train the neural network to error, for epochs =========
        // Trains to 9 epochs by default for recommended settings/vector size
        int epoch = 1;
        do {
            startTime = System.nanoTime();
            crossValidationKFold.iteration();
            currErVal = crossValidationKFold.getError();
            double diff = currErVal - lastErVal;
            lastErVal = currErVal;
            endTime = System.nanoTime();
            period = (endTime - startTime) / 1000000L;
            totalTime += period;
            System.out.println("Epoch #" + epoch + " Run Time: " + period + " ms"
                    +" Error:" + df.format(crossValidationKFold.getError())+" Diff: "+df.format(diff));
            epoch++;
        } while(crossValidationKFold.getError() > 0.0001 && epoch < 10);
        System.out.println("INFO: Training Complete.\nTotal Train Time: " + totalTime + " ms");

        // =============== Save the NN ========================
        Utilities.saveNeuralNetwork(network, "./test.nn");
        System.out.println("Network saved as test.nn");
    }

    /**
     * Test or Evaluate the given network against a dataset.
     * @param trainingSet test dataset.
     * @param network network providing evaluation.
     */
    private void testNetworkOnDataset(MLDataSet trainingSet, BasicNetwork network){
        // ================== Step 4: Test the NN =================
        double correct = 0;
        double total = 0;
        startTime = System.nanoTime();
        for(MLDataPair pair: trainingSet ){
            total++;
            //Get expected res
            double[]ideal=pair.getIdealArray();
            //Get actual index
            int actualCategoryIndex=network.classify(pair.getInput());
            //Get expected index
            int expectedCategoryIndex=getindexOf1InArray(ideal);
            //Compare
            if(actualCategoryIndex==expectedCategoryIndex) correct++;
        }

        endTime = System.nanoTime();
        period = (endTime - startTime) / 1000000L;
        totalTime += period;

        System.out.println("INFO: Correct Predictions: " + correct+" -- Total Predictions: "+ total + "\nTotal Test time: "+period);
        System.out.println("INFO: Testing Complete. Accuracy = " + (correct / total)*100);
        System.out.println("INFO: Total Run Time: " + totalTime + " ms");

    }

    /**
     * Build, train and evaluate a new Neural network.
     * @param activationFunction User selected or default suggested activation function.
     * @param vectorSize User selected or default suggested vector size.
     * @param dataFilePath Data to train, test and evaluate the network with.
     */
    public void trainNewNetwork(ActivationFunction activationFunction, int vectorSize, String dataFilePath){
        // Establish network
        BasicNetwork network = buildNetwork( activationFunction, vectorSize);
        // Establish dataset for training and testing.
        MLDataSet data = buildDataset(dataFilePath, vectorSize);
        // Train to epochs (hardcoded at 9)
        trainDataset(data, network);
        // Evaluate the network against the test data
        testNetworkOnDataset(data, network);
    }


    /**
     * Test a pre-generated Neural Network on a data set.
     * @param nnPath The network for classifying the data.
     */
    public void testExistingNetwork(String nnPath, double[] userTest, String filename){
        File nn = new File(nnPath);
        BasicNetwork network = Utilities.loadNeuralNetwork(nn.getName());
        // Build basic data from vectorised user test file
        // http://heatonresearch-site.s3-website-us-east-1.amazonaws.com/javadoc/encog-3.3/org/encog/ml/data/basic/BasicMLData.html
        BasicMLData testData = new BasicMLData(userTest);
        //Get actual index or predicted data
        int actualCategoryIndex = network.classify(testData);
        // Output the index
        System.out.println("TestFile: "+filename+ " - Predicted Language: " + langs[actualCategoryIndex]);
//        EncogAnalyst analyst = new EncogAnalyst();
//        analyst.load(dataFile);
    }

    /**
     * Return the index of a '1' in an array of all Zeroes and One '1'.
     * @param arr array to search.
     * @return index of the '1' in the array.
     */
    private int getindexOf1InArray(double [] arr){
        int pos = 0;
        for (int i = 0; i < arr.length; i++) {
            if(arr[i] == 1) pos = i;
        }
        return pos;
    }

    /**
     * get the language as evaluated by the network
     * @param index index of predicted language
     * @return the string language.
     */
    private String getLanguageAtIndex(int index){
		String predictedLang = null;
        for (int i = 0; i < langs.length; i++){
			predictedLang = langs[i].name();
		}
        return predictedLang;
    }

    public static void main(String[] args) throws InterruptedException {

////        File in = new File("frenchTest.txt");
//        ArrayList<double []> liveTestDataDir = new ArrayList<>();
//
////        File [] files = new File("./TestData").listFiles();
////        for (File file : files) {
////            VectorProcessor vp = new VectorProcessor(2, 500, file.getPath());
////            vp.go();
////            double [] dd = vp.getLiveData();
////            liveTestDataDir.add(dd);
////        }
////        File aaa = new File("./2gram/data500.csv");
////        File nnn = new File("test.nn");
//        VectorProcessor vp = new VectorProcessor(2, 1000);
//        vp.go();
//        TimeUnit.SECONDS.sleep(2);
//        NN nn = new NN();
//        ActivationFunction activationFunction = new ActivationElliottSymmetric();
//        nn.trainNewNetwork(activationFunction, 500, "data500.csv");
//////        nn.trainNewNetwork(activationFunction, 500, "data500.csv");
//////        for (double [] d: liveTestDataDir) {
//////            nn.testExistingNetwork("test.nn", d);
//////        }
////
////        // Shut down the Neural Network
//        Encog.getInstance().shutdown();
    }
}

