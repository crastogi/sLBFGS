package test;

import base.Model;

public class MinimizerTestFunctions extends Model {
	public int functionType;							//The type of function this Functions Object will represent
	private int maxFunctions = 5;						//The maximum number of functions within this class.

	public MinimizerTestFunctions(int functionType, int nDim) {		//Function constructor
		if (functionType<=0 || functionType>maxFunctions) {
			throw new IllegalArgumentException("Incorrect function type: "+functionType+"; there are only "+maxFunctions+" functions.");
		}
		if (nDim%2!=0) {
			throw new IllegalArgumentException("The number of dimensions must be a multiple of 2!");
		}
		this.functionType	= functionType;				//Stores values for the type of function
		this.nDim			= nDim;						//And the number of dimensions
		N					= 1;						//There is no 'stochastic' behavior here
		theta = new double[nDim];
		
		switch (functionType) {
		case 1:		fName = "Rosenbrock Function";
					break;
		case 2:		fName = "Multi-dimensional Sphere";
					break;
		case 3:		fName = "Beale's Function";
					break;
		case 4:		fName = "Matya's Function";
					break;
		case 5:		fName = "Goldstien-Price Function";
		}
	}
	
	public CompactGradientOutput evaluate() {
		CompactGradientOutput output = null;
		
		switch (functionType) {
			case 1:		output = sdrive();
						break;
			case 2:		output = sphere();
						break;
			case 3:		output = beales();
						break;
			case 4:		output = matyas();
						break;
			case 5:		output = goldstien();
		}
		evaluatedDataPoints++;
		return output;
	}
	
	public CompactGradientOutput stochasticEvaluate() throws Exception {
		return evaluate();
	}
	
	private CompactGradientOutput beales() {					//beale's function
		double a, b, t1, t2, t3;
		double functionValue		= 0;
		double[] functionGradient	= new double[nDim];
		
		for (int j=0; j<nDim; j+=2) {
			a = theta[j];
			b = theta[j+1];
			t1 = 1.5 - a + a*b;
			t2 = 2.25 - a + a*b*b;
			t3 = 2.625 - a + a*b*b*b;
			functionValue += t1*t1 + t2*t2 + t3*t3;
			functionGradient[j]		= 2*(b-1)*t1 + 2*(b*b-1)*t2 + 2*(b*b*b-1)*t3;
			functionGradient[j+1]	= 2*a*t1 + 4*a*b*t2 + 6*a*b*b*t3;
		}
		
		return (new CompactGradientOutput(functionValue, functionGradient));
	}
	
	private CompactGradientOutput goldstien() {					////Goldstien-Price function 			
		double a, b, t1, t2, t3, t4, t5, t6;
		double functionValue		= 0;
		double[] functionGradient	= new double[nDim];
		
		for (int j=0; j<nDim; j+=2) {
			a = theta[j];
			b = theta[j+1];
			t1 = a+b+1;
			t2 = 2*a-3*b;
			t3 = 19-14*a+3*a*a-14*b+6*a*b+3*b*b;
			t4 = 18-32*a+12*a*a+48*b-36*a*b+27*b*b;
			t5 = 1+t1*t1*t3;
			t6 = 30+t2*t2*t4;
			functionValue			+= t5*t6;
			functionGradient[j]		= t5*(t2*t2*(-32+24*a-36*b)+4*t2*t4)+(t1*t1*(-14+6*a+6*b)+2*t1*t3)*t6;
			functionGradient[j+1]	= t5*(t2*t2*(48-36*a+54*b)-6*t2*t4)+(t1*t1*(-14*6*a+6*b)+2*t1*t3)*t6;
		}
		
		return (new CompactGradientOutput(functionValue, functionGradient));
	}
	
	private CompactGradientOutput matyas() {					//matya's function
		double a, b;
		double functionValue		= 0;
		double[] functionGradient	= new double[nDim];
		
		for (int j=0; j<nDim; j+=2) {
			a = theta[j];
			b = theta[j+1];
			functionValue 			+= .26*(a*a + b*b) - .48*a*b;
			functionGradient[j]		= .52*a-.48*b;
			functionGradient[j+1]	= .52*b-.48*a;
		}
		
		return (new CompactGradientOutput(functionValue, functionGradient));
	}
	
	private CompactGradientOutput sdrive() {					//Evaluates the function and gradient for the Sdrive function.
		double t1, t2;
		double functionValue		= 0;
		double[] functionGradient	= new double[nDim];
		
		for (int j=0; j<nDim; j+=2) {					//Sdrive function: (1-x_j)^2 + 100(x_{j+1}-x_j^2)^2 for every x_j, x_{j+1} pair
			t1 = 1-theta[j];								//(Rosenbrock)
			t2 = 10*(theta[j+1]-theta[j]*theta[j]);
			functionGradient[j+1]	= 20*t2;
			functionGradient[j]		= -2*(theta[j]*functionGradient[j+1]+t1);
			functionValue			+= t1*t1+t2*t2;
		}
		
		return (new CompactGradientOutput(functionValue, functionGradient));
	}
	
	private CompactGradientOutput sphere() {					//sphere function 
		double functionValue		= 0;
		double[] functionGradient	= new double[nDim];
				
		for (int j=0; j<nDim; j++) {
			functionValue += theta[j]*theta[j];
			functionGradient[j] = 2*theta[j];
		}
		
		return (new CompactGradientOutput(functionValue, functionGradient));
	}
}
