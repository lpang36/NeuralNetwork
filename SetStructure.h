//Header file for set structure

#pragma once
#include "Matrix.h"
#include <vector>
#include <iostream>
#include <fstream>

class SetStructure {
	public:
		SetStructure();
		~SetStructure();
		void process(std::ifstream &file, int inputCount, int outputCount);
		Matrix inputs;
		Matrix outputs;
		Matrix classes;
		Matrix bias;
		int count;
};
