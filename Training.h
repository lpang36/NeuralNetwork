//Header file for neural net

//Notes:
//make errors float array
//make loaddata take in string parameters rather than prompt

#pragma once
#include "SetStructure.h"
#include "Matrix.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <math.h>
#include <time.h>
#include <stdlib.h>

class Training {
	private:
		Matrix feedForward(Matrix input, Matrix weight, Matrix bias, Matrix &net);
		void loadData();
		void loadData(int a, int b, std::string trainingName, std::string validationName, std::string testName);
		float evaluateNetwork(Matrix input, Matrix weight, Matrix output, Matrix classM, Matrix bias, float &error, int sampleCount);
		void backpropagation(Matrix input, Matrix &weight, Matrix output, double rate, Matrix bias);
		Matrix activation(Matrix M);
		Matrix activationDerivative(Matrix M);
		Matrix weightInit(double weight, int width, int height);
		int inputCount;
		int outputCount;
		int sampleCount;
		SetStructure* training;
		SetStructure* validation;
		SetStructure* test;
		double validationLimit;
	public:
		Training();
		Matrix train(float &trainingE, float &trainingCE, float &validationE, float &validationCE, float &testE, float &testCE);
		~Training();
};


