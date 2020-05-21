package ie.gmit.sw;

import org.encog.Encog;
import org.encog.engine.network.activation.ActivationElliottSymmetric;
import org.encog.engine.network.activation.ActivationFunction;

import java.io.File;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

public class Runner {
	public static void main(String[] args) throws InterruptedException {
		/*
			Each of the languages in the enum Language can be represented as a number between 0 and 234. You can 
			map the output of the neural network and the training data label to / from the language using the
			following. Eg. index 0 maps to Achinese, i.e. langs[0].  
		*/
//		Language[] langs = Language.values(); //Only call this once...
//		for (int i = 0; i < langs.length; i++){
//			System.out.println(i + "-->" + langs[i]);
//		}

		//=========== Working Example ===========
		// Track time to parse file
		//Total Run Time: 3135 ms = NO THREADS
		//Total Run Time: 0236 ms = 10 THREADS
//		long totalTime = 0L;
//		long startTime = 0L;
//		long endTime = 0L;
//		long period = 0L;
//
//		System.out.println("Building Vector Started..");
//		VectorProcessor vp = new VectorProcessor(2, 500);
//		startTime = System.nanoTime();
//		vp.go();
//		endTime = System.nanoTime();
//		period = (endTime - startTime) / 1000000L;
//		totalTime += period;
//		System.out.println("Total Run Time: " + totalTime + " ms");
//		System.out.println("Building Vector Complete..");

//		NN nn = new NN();
//		Map<String, double[]> map = new HashMap<>();
//double [] array = new double[123];
//		map.put("filename", array);

		Menu menu = new Menu();
		menu.doMenu();
	}


}