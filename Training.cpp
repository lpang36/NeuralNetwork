//Main program for neural net
//Shallow learning neural net using tanh activation function
//Backpropagation implemented via matrices

//Notes:
//new vector creation inefficient
//make the limit customizable

#include "Training.h"

using namespace std;

Training::Training() {
	training = new SetStructure(); //training data set
	validation = new SetStructure(); //validation data set
	test = new SetStructure(); //test data set
	srand(time(NULL));
	validationLimit = 0.1; //limit required to terminate learning
}

//Load data from prompted files
void Training::loadData() {
	int a, b;
	cout << "Enter number of inputs: " << endl;
	cin >> a;
	cout << "Enter number of outputs: " << endl;
	cin >> b;
	string trainingName, validationName, testName;
	cout << "Enter training file name: " << endl;
	cin >> trainingName;
	cout << "Enter validation file name: " << endl;
	cin >> validationName;
	cout << "Enter test file name: " << endl;
	cin >> testName;
	loadData(a,b,trainingName,validationName,testName);
}

//Read data from files
void Training::loadData(int a, int b, string trainingName, string validationName, string testName) {
	//Initialize data stream for each file
	ifstream trainingFile, validationFile, testFile;
	ifstream &trainingReference=trainingFile, &validationReference=validationFile, &testReference=testFile;
	inputCount = a;
	outputCount = b;
	int placeholder;
	trainingFile.open(trainingName.c_str());
	validationFile.open(validationName.c_str());
	testFile.open(testName.c_str());
	//Create structure from file
	training->process(trainingReference, inputCount, outputCount);
	validation->process(validationFile, inputCount, outputCount);
	test->process(testFile, inputCount, outputCount);
	trainingFile.close();
	validationFile.close();
	testFile.close();
}

//Activation function
//tanh applied component wise 
Matrix Training::activation(Matrix M) {
	vector< vector<double> > newArray = M.array;
	for (int i = 0; i<M.array.size(); i++) {
		for (int j = 0; j<M.array[0].size(); j++) {
			newArray[i][j] = (tanh(newArray[i][j])+1)/2;
		}
	}
	Matrix output;
	output.array = newArray;
	return output;
}

//Activation derivative function
//Applied component wise
Matrix Training::activationDerivative(Matrix M) {
	vector< vector<double> > newArray = M.array;
	for (int i = 0; i<M.array.size(); i++) {
		for (int j = 0; j<M.array[0].size(); j++) {
			newArray[i][j] = (1-pow(tanh(newArray[i][j]),2))/2;
		}
	}
	Matrix output;
	output.array = newArray;
	return output;
}

//Generates final classification from given weight matrix
Matrix Training::feedForward(Matrix input, Matrix weight, Matrix bias, Matrix &net) {
	net = input.horizontalConcat(bias).matrixMultiplication(weight);
	Matrix output = activation(net);
	return output;
}

//Initialize random weight matrix
Matrix Training::weightInit(double weight, int width, int height) {
	vector< vector<double> > newArray(height,vector<double>(width));
	srand(time(NULL));
	for (int i = 0; i<height; i++) {
		for (int j = 0; j<width; j++) {
			newArray[i][j] = rand()/(RAND_MAX+1.)*weight*2-weight;
		}
	}
	Matrix output;
	output.array = newArray;
	return output;
}

//Evaluate error present in network
//Two types of error
//&error is the square error between the target output matrix and the generated output matrix
//cError is the classification error, i.e. the number of places where the generated classification differs from the target
float Training::evaluateNetwork(Matrix input, Matrix weight, Matrix output, Matrix classM, Matrix bias, float &error, int sampleNum) {
	Matrix &placeholder = *(new Matrix());
	Matrix Z = feedForward(input, weight, bias, placeholder);
	delete &placeholder;
	Matrix classO = Z.matrixOutput();
	output = output.scalarMultiplication(-1);
	Z = Z.add(output);
	Z = Z.hadamardMultiplication(Z);
	error = Z.sumAll()/(sampleNum*outputCount);
	float cError = classM.hamming(classO)/(sampleNum+0.0);
	return cError;
}

//Backpropagation algorithm
void Training::backpropagation(Matrix input, Matrix &weight, Matrix output, double rate, Matrix bias) {
	int rowNum = (int) rand()/(RAND_MAX+1.)*training->count;
	Matrix sampleInput = input.row(rowNum);
	Matrix sampleBias = bias.row(rowNum);
	Matrix sampleOutput = output.row(rowNum);
	Matrix &net = *(new Matrix());
	Matrix Z = feedForward(sampleInput,weight,sampleBias,net);
	Matrix Y = Z.scalarMultiplication(-1);
	Matrix error = Y.add(sampleOutput);
	Matrix delta = error.hadamardMultiplication(activationDerivative(net));
	Matrix weightsDelta = (delta.transpose()).kroneckerMultiplication(sampleInput.horizontalConcat(sampleBias));
	weightsDelta = weightsDelta.scalarMultiplication(rate);
	weight = weight.add(weightsDelta);
	delete &net;
} 

//Train neural net
Matrix Training::train(float &trainingE, float &trainingCE, float &validationE, float &validationCE, float &testE, float &testCE) {
	//Change these paths to analyze different data sets
	loadData(4,3,"iris_data_files/iris_training.dat","iris_data_files/iris_validation.dat","iris_data_files/iris_test.dat");
	Matrix weights = weightInit(0.5,outputCount,inputCount+1);
	validationE = validationLimit+1;
	//Train until validation error is less than the given limit
	while (validationE>validationLimit) {
		backpropagation(training->inputs,weights,training->outputs,0.1,training->bias);
		trainingCE = evaluateNetwork(training->inputs,weights,training->outputs,training->classes,training->bias,trainingE,training->count);
		validationCE = evaluateNetwork(validation->inputs,weights,validation->outputs,validation->classes,validation->bias,validationE,validation->count);
		testCE = evaluateNetwork(test->inputs,weights,test->outputs,test->classes,test->bias,testE,test->count);
	}
	return weights;
}

Training::~Training() {
	delete training;
	delete validation;
	delete test;
}
