package test;

import java.io.OutputStream;
import java.io.PrintStream;
import java.util.Random;

import base.Fit;
import base.Minimizer;
import base.sLBFGS;

public class MinimizerTest {
	public static int testLoops = 10;				//Number of random starts to test
	public static double xlim = 1, ylim = 1;		//Bounds of random starts
	public static int maxMemoryDepth = 10, maxIterations = 100;
	public static double convergence = 1E-5;
	public static double stochStepSize = 0.01;
	public static int dimensionality = 2;
	
	public static void main(String[] args) {
		double tStart, tEnd, distance, fitTime, iterations, dataLoops, successes;
		//Initialize random generator
		double[] seed, correctPos = null;
		Random generator = new Random();
		
		//Create null output stream
		PrintStream originalOut	= System.out;
		PrintStream originalErr = System.err;
		PrintStream nullStream  = new PrintStream(new NullOutputStream());  
		
		//Loop over function types
		for (int fType = 1; fType<=5; fType++) {
			//Initialize test functions
			MinimizerTestFunctions testFunc = new MinimizerTestFunctions(fType, dimensionality);
			Fit fit = null;
			Minimizer minimize = new Minimizer();
			
			//Define final position
			correctPos = new double[dimensionality];
			switch (fType) {
				case 1:		for (int i=0; i<dimensionality; i++) {
								correctPos[i] = 1;
							}
							break;
				case 2:		for (int i=0; i<dimensionality; i++) {
								correctPos[i] = 0;
							}
							break;
				case 3:		for (int i=0; i<dimensionality; i+=2) {
								correctPos[i] = 3;
								correctPos[i+1] = 0.5;
							}
							break;
				case 4:		for (int i=0; i<dimensionality; i++) {
								correctPos[i] = 0;
							}
							break;
				case 5:		for (int i=0; i<dimensionality; i+=2) {
								correctPos[i] = 0;
								correctPos[i+1] = -1;
							}
			}
			
			//Loop over multiple random starts
			System.setOut(nullStream);
			System.setErr(nullStream);
			fitTime = 0;
			iterations = 0;
			successes = 0;
			dataLoops = 0;
			for (int currLoop = 0; currLoop<testLoops; currLoop++) {
				//create new seed
				seed = new double[dimensionality];
				for (int i=0; i<dimensionality; i++) {
					seed[i] = generator.nextDouble()*xlim*2-xlim;
				}
				
				// Run sLBFGS
				minimize = new sLBFGS(testFunc, 1, 1, maxMemoryDepth, maxIterations, 10, 500, stochStepSize, convergence, 0, true);
				try {
					tStart = System.nanoTime();
					fit = minimize.doMinimize(seed, null);
					tEnd = System.nanoTime();
					distance = 0;
					for (int i=0; i<dimensionality; i++) {
						distance += (fit.finalPosition[i]-correctPos[i])*(fit.finalPosition[i]-correctPos[i]);
					}
					if (Math.sqrt(distance)<5E-3) {
						successes++;
						fitTime		+= (tEnd-tStart)/1E9;
						iterations	+= fit.fitSteps;
						dataLoops	+= fit.dataLoops;
					}
				} catch (Exception e) {
					//Do nothing
				}
			}
			System.setOut(originalOut);
			System.setErr(originalErr);
			System.out.println("Function Type:\t"+testFunc.fName);
			System.out.println("Successes:     \t"+successes);
			System.out.println("Fit Time:      \t"+fitTime);
			System.out.println("Iterations:    \t"+iterations);
			System.out.println("Data Loops:    \t"+dataLoops);
			System.out.println("\n");
		}
		
		// Run SVM test. Begin with a convergent L-BFGS run
		SVM svm = new SVM(0.001);
		Minimizer min = new Minimizer();
		Fit slbfgsFit = null;
		fitTime = 0;
		iterations = 0;
		successes = 0;
		dataLoops = 0;
		
		// Minimize with LBFGS
		System.setOut(nullStream);
		System.setErr(nullStream);
		for (int currLoop = 0; currLoop<testLoops; currLoop++) {			
			try {
				min = new sLBFGS(svm, 20, 200, maxMemoryDepth, maxIterations, 10, 500, stochStepSize, convergence, 0, false);
				tStart = System.nanoTime();
				slbfgsFit = min.doMinimize(null, null);
				tEnd = System.nanoTime();
				fitTime += (tEnd-tStart)/1E9;
				iterations += slbfgsFit.fitSteps;
				successes++;
				dataLoops += slbfgsFit.dataLoops;
			} catch (Exception e) {
				// Do nothing
			}
		}
		System.setOut(originalOut);
		System.setErr(originalErr);
		System.out.println("Function Type:\t"+svm.fName);
		System.out.println("Successes:     \t"+successes);
		System.out.println("Fit Time:      \t"+fitTime);
		System.out.println("Iterations:    \t"+iterations);
		System.out.println("Data Loops:    \t"+dataLoops);
		System.out.println("\n");
	}
	
	private static class NullOutputStream extends OutputStream {
	    @Override
	    public void write(int b){
	         return;
	    }
	    @Override
	    public void write(byte[] b){
	         return;
	    }
	    @Override
	    public void write(byte[] b, int off, int len){
	         return;
	    }
	    public NullOutputStream(){
	    }
	}
}
