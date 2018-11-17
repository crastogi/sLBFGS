package base;

import java.util.HashSet;
import java.util.Iterator;
import java.util.Random;

public abstract class Model {
	public int nDim = 0, N = 0, evaluatedDataPoints = 0;
	public String fName = "N/A";
	public int[] currBatchIdx;
	public double[] theta;
	private Random mtfast = new Random();
	
	public int getNFeatures() {
		return nDim;
	}

	public void setParams(double[] position) {
		for (int i=0; i<nDim; i++) {
			theta[i] = position[i];
		}
	}

	public double[] getParams() {
		double[] out = new double[nDim];
		
		for (int i=0; i<nDim; i++) {
			out[i] = theta[i];
		}
		return out;
	}
 
	public abstract CompactGradientOutput evaluate() throws Exception;
		
	public abstract CompactGradientOutput stochasticEvaluate() throws Exception;
	
	public Fit generateFit(double[] seed) {
		return new Fit(fName, nDim, seed);
	}
	
	public void sampleBatch(int k) {
		int idx = 0;
		currBatchIdx = new int[k];
		HashSet<Integer> h = new HashSet<Integer>();
		
		for (int i=0; i<k; i++) {
			h.add(mtfast.nextInt(N));
		}
		while(h.size()<k-1) {
			h.add(mtfast.nextInt(N));
		}
		Iterator<Integer> i = h.iterator(); 
        while (i.hasNext()) {
        	currBatchIdx[idx] = i.next();
        	idx++;
        }
		return;
	}
	
	public class CompactGradientOutput {
		public double functionValue = 0;
		public double[] gradientVector;

		public CompactGradientOutput(double functionValue, double[] gradientVector) {
			this.functionValue	= functionValue;
			this.gradientVector	= gradientVector;
		}
		
		public CompactGradientOutput(double[] gradientVector) {
			this.gradientVector	= gradientVector;
		}
	}
}
