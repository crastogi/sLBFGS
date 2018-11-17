package base;

import java.util.ArrayList;
import java.util.Random;

public class sLBFGS extends Minimizer{
	private boolean isVerbose, randomSelect = false;
	private int d, N, b, bH, M, m, L, currDepth, maxEpoch;
	private double eta, delta, fdHVPStepSize = 5E-2, gradientNormBound = 100;
	private double[] rho;
	private double[][] s, y;
	private Fit fitOutput;
	private Model.CompactGradientOutput fOut;
	
	//LBFGS object constructor; load basic minimization parameters. To be used for all subsequent minimizations using this object.
	public sLBFGS(Model model, int gradientBatch, int hessianBatch, int memorySize, 
			int maxEpoch, int hessianPeriod, int epochIterations, double stepsize,
			double epsilon, double delta, boolean isVerbose) {
		this.model		= model;
		d				= model.getNFeatures();
		N				= model.N;				// Number of data points 
		if (gradientBatch<1) throw new IllegalArgumentException("Gradient batch size must be greater than 1!");
		b				= gradientBatch;		// Batch size for stochastic gradient updates
		if (hessianBatch<1) throw new IllegalArgumentException("Hessian batch size must be greater than 1!");
		bH				= hessianBatch;			// Batch size for stochastic Hessian updates
		if (memorySize<=0)	throw new IllegalArgumentException("Memory size must be positive!");
		M				= memorySize;			// Maximum memory depth
		if (maxEpoch<1) throw new IllegalArgumentException("The maximum number of epochs must be positive!");
		this.maxEpoch	= maxEpoch;				// The maximum number of epochs before termination
		if (hessianPeriod<2) throw new IllegalArgumentException("The hessian update period must be larger than 1!");
		L				= hessianPeriod;		// Number of iterations before the inverse hessian is updated
		if (epochIterations<=hessianPeriod) throw new IllegalArgumentException("The number of iterations per epoch must be greater than the hessian period!");
		m				= epochIterations;		// Number of iterations to run per epoch
		if (stepsize<=0) throw new IllegalArgumentException("Eta (step size) must be greater than 0!");
		eta				= stepsize;				// The fixed step size value
		if (epsilon<0) throw new IllegalArgumentException("Epsilon cannot be negative!");
		this.epsilon	= epsilon;				// The accuracy with which the solution needs to be found
		if (delta<0) throw new IllegalArgumentException("Delta (inverse hessian regularization parameter) cannot be negative!");
		this.delta		= delta;				// An optional inverse hessian regularization parameter
		this.isVerbose	= isVerbose;
	}
	
	public Fit doMinimize(double[] seed, String trajectoryFile) throws Exception {
		int r = 0;									// Number of currently computed Hessian correction pairs
		currDepth = 0;
		double tStart, egNorm, divisor;
		// Full gradient, variance reduced gradient, and components of variance reduced gradient
		double[] mu_k = new double[d], v_t = new double[d], grad_f_xt = new double[d], grad_f_wk = new double[d];
		// Effective gradient
		double[] effGrad = new double[d];
		// Positions in current iterations
		double[] w_k = new double[d], w_k_prev = new double[d], x_t = new double[d];
		// Average of path travelled in the current and previous inverse hessian updates
		double[] u_r = new double[d], u_r_prev = new double[d];
		// Components of two-loop update
		double[] s_r = new double[d], y_r= new double[d];
		rho = new double[M];
		s = new double[M][d];
		y = new double[M][d];
		// Stores the history of all previous steps in the current epoch
		ArrayList<double[]> x_t_hist = new ArrayList<double[]>();
		Random generator = new Random();
		
		// Deal with a potential seed and initialize
		if (seed!=null) {
			try {
				w_k = clone(seed);
			} catch (Exception e) {
				throw new IllegalArgumentException("Improper seed!");
			}
		} else {
			for (int i=0; i<d; i++) {
				w_k[i] = randomSeedScale*generator.nextDouble();
			}
		}
		fitOutput = model.generateFit(seed);
		w_k_prev = clone(w_k);
			
		// Init the number of loops over data points
		model.evaluatedDataPoints = 0;
		tStart	= System.nanoTime();
		// Loop over epochs 
		for (int k=0; k<=maxEpoch; k++) {
			// Compute full gradient for variance reduction
			fOut = evaluate(w_k);
			// Print some information about the current epoch
			if (isVerbose) {
				if (k==0) {
					System.out.println("Starting Function Value: "+fOut.functionValue);
					System.out.println("    Epochs   Data Loops           Likelihood       Distance Moved"+
							"        Gradient Norm");
				} else {
					printStep(k, model.evaluatedDataPoints/N, fOut.functionValue, norm(subtract(w_k, w_k_prev)), 
							norm(fOut.gradientVector));					
				}
			}

			// Check for convergence, final epoch
			if (Double.isNaN(fOut.functionValue) || Double.isInfinite(fOut.functionValue)) {
				throw new Exception("sLBFGS Failure: NaN encountered! Try reducing the step size...");
			}
			if (norm(fOut.gradientVector)/Math.max(1, norm(w_k)) < epsilon) {
				fitOutput.recordFit(k, (System.nanoTime()-tStart)/1E9, fOut.functionValue, model);
				System.out.println("Convergence criteria met.");
				return fitOutput;
			}
			if (k==maxEpoch) {
				break;			// This allows convergence to be tested on the FINAL epoch iteration without computation
			}

			mu_k = clone(fOut.gradientVector);		// Assign variance reduced gradient
			x_t = clone(w_k);							// Set x_t to current value of w_k
						
			// Perform m stochastic iterations before a full gradient computation takes place
			for (int t=1; t<=m; t++) {
				// Compute the current stochastic gradient estimate; begin by sampling a minibatch
				model.sampleBatch(b);
				// Next, compute the reduced variance gradient
				grad_f_xt = stochasticEvaluate(x_t).gradientVector;
				grad_f_wk = stochasticEvaluate(w_k).gradientVector;
				for (int i=0; i<d; i++) {
					v_t[i] = grad_f_xt[i] - grad_f_wk[i] + mu_k[i];
				}
//				v_t = Array.subtract(Array.add(grad_f_xt, mu_k), grad_f_wk);
				
				// Update u_r with current position
				for (int i=0; i<d; i++) {
					u_r[i] += x_t[i];
				}
//				u_r = Array.add(u_r, x_t);			
				// Compute next iteration step; condition the gradient so as not to produce rapid oscilations (extreme function values)
				x_t_hist.add(clone(x_t));		// Need to store the history of iteration steps				
				if (r < 1) {						// Until a single hessian correction has taken place, H_0 = I
					effGrad = v_t;
				} else {							// Compute the two-loop recursion product
					effGrad = twoLoopRecursion(v_t);
				}			
				// Bound the effective gradient update step
				egNorm = norm(effGrad);
				divisor = 1;
				while (egNorm/divisor > gradientNormBound) {
					divisor *= 10;
				}
				x_t = addScalarMultiply(x_t, -eta, scalarMultiply(effGrad, 1/divisor));
				
				// Check to see if L iterations have passed (triggers hessian update)
				if (t % L == 0) {
					// Increment the number of hessian correction pairs
					r++;
					// Finish computing u_r
					u_r = scalarMultiply(u_r, 1.0/((double) L));
					
					// Use HVP to compute hessian updates. Begin by sampling a minibatch
					model.sampleBatch(bH);
					// Compute s_r update
					s_r = subtract(u_r, u_r_prev);
					// Compute y_r estimate using HVP
					y_r = subtract(stochasticEvaluate(addScalarMultiply(u_r, fdHVPStepSize, s_r)).gradientVector, 
							stochasticEvaluate(addScalarMultiply(u_r, -fdHVPStepSize, s_r)).gradientVector);
					y_r = scalarMultiply(y_r, 1.0/(2*fdHVPStepSize));
					// Store latest values of s_r and y_r
					add(s_r, y_r);
					
					// Resetting u_r for next evaluation
			        u_r_prev = clone(u_r);
			        u_r = new double[d];
				}
			}
			// Choose either the last position vector x_t or a random one from the previous epoch as the starting point for the next
			if (randomSelect) {
				w_k = clone(x_t_hist.get(generator.nextInt(m)));
			} else {
				w_k = clone(x_t);
			}
			x_t_hist = new ArrayList<double[]>();
		}
		throw new Exception("sLBFGS Failure: maximum epochs exceeded without convergence!");
	}
	
	private double[] twoLoopRecursion(double[] v_t) {
		double alpha, beta, gamma;
		double[] q = new double[d], r = new double[d];
		double[] alphas = new double[currDepth];
		
		// Begin by cloning the input gradient
		q = clone(v_t);
		
		// The first loop (starts from the latest entry and goes to the earliest)
		for (int i=0; i<currDepth; i++) {
			// Compute and store alpha_i = rho_u*s_i*q
			alpha = rho[i]*dotProduct(s[i], q);
			alphas[i] = alpha;
			// Update q: q = q - alpha_i*y_i
			q = addScalarMultiply(q, -alpha, y[i]);
		}		
		// Start computing R. To do so, begin by computing gamma_k = s_k*y_k/(y_k*y_k)
		gamma = dotProduct(s[currDepth-1], y[currDepth-1])/
				dotProduct(y[currDepth-1], y[currDepth-1]);
		// r = gamma_k*q/(1 + delta*gamma_k); the denominator includes the pseudo-hessian 
		// regularization parameter delta NOTE: There is no need to multiply by I here, 
		// as that will anyway produce a dot product 
		r = scalarMultiply(q, gamma/(1.0 + delta*gamma));
		
		// Second loop (goes in reverse, starting from the earliest entry)
		for (int i=currDepth-1; i>=0; i--) {
			// beta = rho_i*y_i*r
			beta = rho[i]*dotProduct(y[i], r);
		    // r = r + s_i*(alpha_i-beta)
			r = addScalarMultiply(r, alphas[i]-beta, s[i]);
		}
		return r;
	}
	
	private void add(double[] inS, double[] inY) {
		// Compute rho in the lbfgs two-loop method: rho_j = 1/s_j^T*y_j
		cycleDown(rho, 1.0/dotProduct(inS, inY));
		// Add values to structure
		cycleDown(s, inS);
		cycleDown(y, inY);
		currDepth = Math.min(currDepth+1, M);
	}
	
	// Cycle rows in matrix downwards and add a new row on top
	private void cycleDown(double[][] input, double[] newRow) {	
		for (int i=input.length-1; i>0; i--) {
			for (int j=0; j<input[0].length; j++) {
				input[i][j] = input[i-1][j];
			}
		}
		for (int i=0; i<input[0].length; i++) {
			input[0][i] = newRow[i];
		}
	}
	
	private void cycleDown(double[] input, double newValue) {	
		for (int i=input.length-1; i>0; i--) {
			input[i] = input[i-1];
		}
		input[0] = newValue;
	}
	
	private double[] addScalarMultiply(double[] x, double c, double[] y) {
		double[] out = new double[x.length];
		
		for (int i=0; i<x.length; i++) {
			out[i] = x[i]+c*y[i];
		}
		return out;
	}
	
	private double[] clone(double[] in) {
		double[] out = new double[in.length];
		
		for (int i = 0; i < in.length; i++){
			out[i] = in[i];
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
	
	private double norm(double[] in) {
		double out = 0;
		
		for (int i=0; i<in.length; i++) {
			out += in[i]*in[i];
		}
		return Math.sqrt(out);
	}
	
	private double[] scalarMultiply(double[] in, double scalar) {
		double[] out = new double[in.length];
		for (int i=0; i<in.length; i++) {
			out[i] = in[i]*scalar;
		}
		return out;
	}
	
	private double[] subtract(double[] a, double[] b) {
		double[] out = new double[a.length];
		
		for (int i=0; i<a.length; i++) {
			out[i] = a[i]-b[i];
		}
		return out;
	}
}