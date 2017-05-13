//Class used to organize a data set into inputs, outputs, and bias vector

//Notes:
//passing by something else might speed up file processing
//limit number of samples used for training

#include "SetStructure.h"

using namespace std;

SetStructure::SetStructure() {
}

//Read in structure from text file
void SetStructure::process(ifstream &file, int inputCount, int outputCount) {
	std::vector< std::vector<double> > inputVector; //input
	std::vector< std::vector<double> > outputVector; //output
	std::vector< std::vector<double> > biasVector; //bias
	count = 0; 
	double nextValue;
	while (file >> nextValue) {
		//Create next row of input, output, bias
		std::vector<double> inputRow (inputCount);
		std::vector<double> outputRow (outputCount);
		std::vector<double> biasRow (1,1);
		//Read in input values
		for (int i = 0; i<inputCount; i++) {
			if (i!=0) {
				file >> nextValue;
			}
			inputRow[i] = nextValue;
			if (i==inputCount-1) {
				inputVector.push_back(inputRow);
			}
		}
		//Read in output values
		for (int i = 0; i<outputCount; i++) {
			file >> nextValue;
			outputRow[i] = nextValue;
			if (i==outputCount-1) {
				outputVector.push_back(outputRow);
			}
		}
		biasVector.push_back(biasRow);
		count++;
	}
	//Create matrices from input, output, bias vectors
	inputs = *(new Matrix(inputVector));
	outputs = *(new Matrix(outputVector));
	classes = outputs.matrixOutput();
	bias = *(new Matrix(biasVector));
}

SetStructure::~SetStructure() {
}
