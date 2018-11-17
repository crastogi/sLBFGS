package base;

import java.util.ArrayList;

public class Fit {
	public int fitSteps, nDim, dataPoints;
	public double fitTime, likelihood, dataLoops;
	public String functionType;
	public double[] seed						= null;
	public double[] finalPosition				= null;
	public ArrayList<double[]> trajectory		= new ArrayList<double[]>();
	
	// Fit Constructor
	public Fit(String functionType, int nDim, double[] seed) {		
		this.functionType	= functionType;				// Stores the name of the function
		this.nDim			= nDim;						// Stores dimensionality
		this.seed			= seed;						// And seed (if it exists)
	}
	
	public void recordFit(int fitSteps, double fitTime, double likelihood, Model input) {
		this.fitSteps = fitSteps;
		this.fitTime = fitTime;
		this.likelihood = likelihood;
		this.finalPosition = input.getParams();
		dataPoints = input.N;
		dataLoops = ((double) input.evaluatedDataPoints)/((double) input.N);
	}
		
	public double[] positionVector() {
		double[] out = new double[nDim];
		
		for (int i=0; i<nDim; i++) {
			out[i] = finalPosition[i];
		}
		return out;
	}
}