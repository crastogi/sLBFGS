package test;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;

import base.Model;

public class SVM extends Model {
	public double lambda;								// Regularizer for svm
	public double[] classes;							// The class of every data point
	public double[][] data;								// The data for every point

	public SVM(double lambda) {							// Load data
		int maxCols;
        String csvFile = "./src/featurized_df.csv";
        String line;
        double[] temp;
        String[] split;
        ArrayList<double[]> parsed = new ArrayList<double[]>();

        // Read in data file
        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
        	// Ignore first line
        	line = br.readLine();
        	maxCols = line.split(",").length;
            // Read, split, and ensure that the remaining lines have the same number of columns
        	// Store the remaining values in a matrix
        	while ((line = br.readLine()) != null) {
            	split = line.split(",");
            	if (split.length != maxCols) {
            		throw new Exception("Variable number of columns!");
            	}
            	temp = new double[maxCols];
            	for (int i=0; i<maxCols; i++) {
            		temp[i] = Double.parseDouble(split[i]);
            	}
            	parsed.add(temp);
            }
        	// Store values in data
        	N = 10000; 
        	data = new double[N][maxCols-1];
        	classes = new double[N];
        	for (int i=0; i<N; i++) {
        		for (int j=0; j<maxCols-1; j++) {
        			data[i][j] = parsed.get(i)[j];
        		}
        		classes[i] = parsed.get(i)[maxCols-1];
        	}
        	nDim = maxCols-1;
        } catch (Exception e) {
            e.printStackTrace();
        }
        
        this.lambda = lambda;
        theta = new double[nDim];
        fName = "Simple SVM";
	}

	// Computes both the likelihood and the gradient
	public CompactGradientOutput evaluate() {
		double alpha, y;
		double likelihood = 0, temp;
		double x[];
		double gradient[] = new double[nDim];
		
		for (int i=0; i<N; i++) {
			x = data[i];
			y = classes[i];
			alpha = dotProduct(x, theta);
			temp = Math.max(0.0, 1 - y*alpha);
			likelihood += temp*temp;
			if (y*alpha < 1.0) {
				gradient = addScalarMultiply(gradient, -1, scalarMultiply(data[i], y*(1 - y*alpha)));
			}
		}
		likelihood /= 2*N;											// Normalize by the number of data points
		likelihood += lambda*dotProduct(theta, theta)/2;		// Add regularization
		gradient = scalarMultiply(gradient, 1/((double) N));
		gradient = addScalarMultiply(gradient, lambda, theta);
		evaluatedDataPoints += N;
		
		return (new CompactGradientOutput(likelihood, gradient));
	}
	
	public CompactGradientOutput stochasticEvaluate() {
		double alpha, y;
		double likelihood = 0, temp;
		double x[];
		double gradient[] = new double[nDim];
		
		for (int i : currBatchIdx) {
			x = data[i];
			y = classes[i];
			alpha = dotProduct(x, theta);
			temp = Math.max(0.0, 1 - y*alpha);
			likelihood += temp*temp;
			if (y*alpha < 1.0) {
				gradient = addScalarMultiply(gradient, -1, scalarMultiply(data[i], y*(1 - y*alpha)));
			}
		}
		likelihood /= 2*currBatchIdx.length;						// Normalize by the number of data points
		likelihood += lambda*dotProduct(theta, theta)/2;		// Add regularization
		gradient = scalarMultiply(gradient, 1/((double) currBatchIdx.length));
		gradient = addScalarMultiply(gradient, lambda, theta);
		evaluatedDataPoints += currBatchIdx.length;
		return (new CompactGradientOutput(likelihood, gradient));
	}
	
	public CompactGradientOutput evaluate(double[] theta) {			//Overloaded operator for easy function calling
		setParams(theta);
		return evaluate();
	}
	
	private double[] addScalarMultiply(double[] x, double c, double[] y) {
		double[] out = new double[x.length];
		
		for (int i=0; i<x.length; i++) {
			out[i] = x[i]+c*y[i];
		}
		return out;
	}
	
	private double dotProduct(double[]a, double[]b) {
		double out = 0;
		for (int i=0; i<a.length; i++) {
			out += a[i]*b[i];
		}
		return out;
	}
	
	private double[] scalarMultiply(double[] in, double scalar) {
		double[] out = new double[in.length];
		for (int i=0; i<in.length; i++) {
			out[i] = in[i]*scalar;
		}
		return out;
	}
}
