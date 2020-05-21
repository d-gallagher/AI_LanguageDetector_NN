package ie.gmit.sw;

import java.io.File;
import java.text.DecimalFormat;

import org.encog.Encog;
import org.encog.engine.network.activation.*;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.buffer.MemoryDataLoader;
import org.encog.ml.data.buffer.codec.CSVDataCODEC;
import org.encog.ml.data.buffer.codec.DataSetCODEC;
import org.encog.ml.data.folded.FoldedDataSet;
import org.encog.ml.train.MLTrain;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.cross.CrossValidationKFold;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.util.csv.CSVFormat;

public class NeuralNetwork {

	/*
	 * *************************************************************************************
	 * NB: READ THE FOLLOWING CAREFULLY AFTER COMPLETING THE TWO LABS ON ENCOG AND REVIEWING
	 * THE LECTURES ON BACKPROPAGATION AND MULTI-LAYER NEURAL NETWORKS! YOUR SHOULD ALSO
	 * RESTRUCTURE THIS CLASS AS IT IS ONLY INTENDED TO DEMO THE ESSENTIALS TO YOU.
	 * *************************************************************************************
	 *
	 * The following demonstrates how to configure an Encog Neural Network and train
	 * it using backpropagation from data read from a CSV file. The CSV file should
	 * be structured like a 2D array of doubles with input + output number of columns.
	 * Assuming that the NN has two input neurons and two output neurons, then the CSV file
	 * should be structured like the following:
	 *
	 *			-0.385,-0.231,0.0,1.0
	 *			-0.538,-0.538,1.0,0.0
	 *			-0.63,-0.259,1.0,0.0
	 *			-0.091,-0.636,0.0,1.0
	 *
	 * The each row consists of four columns. The first two columns will map to the input
	 * neurons and the last two columns to the output neurons. In the above example, rows
	 * 1 an 4 train the network with features to identify a category 2. Rows 2 and 3 contain
	 * features relating to category 1.
	 *
	 * You can normalize the data using the Utils class either before or after writing to
	 * or reading from the CSV file.
	 */
	public int getInputsFromFileName(String filename){
		return Integer.parseInt(filename.substring(4, filename.length()-4));
	}

	public NeuralNetwork(File dataFile) {
		//============== Initialise Variables =========================

		int inputs = getInputsFromFileName(dataFile.getName()); //Change this to the number of input neurons
		int outputs = 235; //Change this to the number of output neurons
		double hiddenGPR = Math.sqrt(inputs * outputs); // Geometric Pyramid Rule to calculate hidden layer nodes
		// recommended dropout rate of 0.8 https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/
		int dropoutRate = (int) ( inputs*0.8);

		DecimalFormat df = new DecimalFormat("###.#######");

		long totalTime = 0L;
		long startTime = 0L;
		long endTime = 0L;
		long period = 0L;
		double lastErVal= 0;
		double currErVal;

		//============= Configure the neural network topology. =========================
		BasicNetwork network = new BasicNetwork();
		network.addLayer(new BasicLayer(null, true, inputs));
		network.addLayer(new BasicLayer(new ActivationElliottSymmetric(), true, (int) hiddenGPR, dropoutRate)); 	// Hidden using GPR
//		network.addLayer(new BasicLayer(new ActivationLOG(), true, (int) hiddenGPR, dropoutRate)); 	// Hidden using GPR
//		network.addLayer(new BasicLayer(new ActivationTANH(), true, (int) hiddenGPR, dropoutRate)); 	// Hidden using GPR
//		network.addLayer(new BasicLayer(new ActivationClippedLinear(), true, (int) hiddenGPR, dropoutRate)); 	// Hidden using GPR
//		network.addLayer(new BasicLayer(new ActivationBipolarSteepenedSigmoid(), true, (int) hiddenGPR, dropoutRate)); 	// Hidden using GPR
//		network.addLayer(new BasicLayer(new ActivationBiPolar(), true, (int) hiddenGPR, dropoutRate)); 	// Hidden using GPR
//		network.addLayer(new BasicLayer(new ActivationCompetitive(), true, (int) hiddenGPR, dropoutRate)); 	// Hidden using GPR
		network.addLayer(new BasicLayer(new ActivationSoftMax(), false, outputs));
		network.getStructure().finalizeStructure();
		network.reset();

		//========================= Generate Training Set ==============================
		//Read the CSV file "data.csv" into memory. Encog expects your CSV file to have input + output number of columns.
		DataSetCODEC dsc = new CSVDataCODEC(dataFile, CSVFormat.ENGLISH, false, inputs, outputs, false);
		MemoryDataLoader mdl = new MemoryDataLoader(dsc);
		MLDataSet trainingSet = mdl.external2Memory();

		//========================= Train Training Set ==============================
		//----- CrossValidation no Dropout ---
		FoldedDataSet folded = new FoldedDataSet(trainingSet);
        MLTrain train = new ResilientPropagation(network, folded);
        CrossValidationKFold crossValidationKFold = new CrossValidationKFold(train, 5);

		//========= Train the neural network to error, for epochs =========
		int epoch = 1; //Use this to track the number of epochs
		do {
//			System.out.println("Epoch Started..");
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
		} while(crossValidationKFold.getError() > 0.001 && epoch < 20);
		System.out.println("INFO: Total Train Time: " + totalTime + " ms");

		// =============== Save the NN ========================
		Utilities.saveNeuralNetwork(network, "./test.nn");

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

		System.out.println("INFO: Correct: " + correct+" -- Total: "+ total + " Test time: "+period);
		System.out.println("INFO: Training Complete. Accuracy = " + (correct / total)*100 + " Train time: "+period);
		System.out.println("INFO: Total Run Time: " + totalTime + " ms");

//		EncogUtility.evaluate(network, trainingSet);
		// Step 5: Shutdown the NN
		Encog.getInstance().shutdown();
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

	public int getindexOf1InArray(double [] arr){
		int pos = 0;
		for (int i = 0; i < arr.length; i++) {
			if(arr[i] == 1) pos = i;
		}
		return pos;
	}

	public static void main(String[] args) {
//		String working_dir = System.getProperty("user.dir");
		int ngramSize = 2;
		int num = 300;
//		for (int i = 0; i < 8; i++) {
			String filepath = "data/" + ngramSize + "grams/" + "data" + num + ".csv";
			System.out.println("csv"+num+":");
			NeuralNetwork nn = new NeuralNetwork(new File("data300.csv"));
//			ngramSize ++;
//			num += 50;
//		}
//		String filename = "data500.csv";
//		System.out.println(50*.8);
	}
}

