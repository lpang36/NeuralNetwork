//Matrix class - contains basic matrix operations necessary for neural net to function

//Notes:
//optimization possible: creation of new matrix requires duplication
//sometimes new vectors are created by = old ones - inefficient

#include "Matrix.h"

using namespace std;

Matrix::Matrix() {
}

//Create matrix from 2D vector
Matrix::Matrix(const vector< vector<double> > &b) {
	array = b;
}

//Create matrix from 1D vector
//Example: column vector 1 4 2 3 will create matrix
//1 0 0 0
//0 0 0 1
//0 1 0 0
//0 0 1 0
Matrix::Matrix(const vector<double> &b) {
	int max = 0; 
	for (int i = 0; i<b.size(); i++) {
		if (b[i]>max) {
			max = b[i];
		}
	}
	vector< vector<double> > array(b.size(),vector<double>(max,0));
	for (int i = 0; i<b.size(); i++) {
		array[i][(int)b[i]] = 1;
	}
}

//Matrix transpose
Matrix Matrix::transpose() const {
	vector< vector<double> > newArray(array[0].size(),vector<double> (array.size()));
	for (int i = 0; i<array.size(); i++) {
		for (int j = 0; j<array[0].size(); j++) {
			newArray[j][i] = array[i][j];
		}
	}
	Matrix output;
	output.array = newArray;
	return output;
}

//Matrix addition
Matrix Matrix::add(const Matrix &b) const {
	//Dimension mismatch
	if (array[0].size()!=b.array[0].size()||array.size()!=b.array.size()) {
		cout << "Matrix size mismatch";
	}
	vector< vector<double> > newArray = array;
	for (int i = 0; i<array.size(); i++) {
		for (int j = 0; j<array[0].size(); j++) {
			newArray[i][j] = b.array[i][j]+array[i][j];
		}
	}
	Matrix output;
	output.array = newArray;
	return output;
}

//Scalar multiplication
Matrix Matrix::scalarMultiplication(double factor) const {
	vector< vector<double> > newArray = array;
	for (int i = 0; i<array.size(); i++) {
		for (int j = 0; j<array[0].size(); j++) {
			newArray[i][j] = factor*array[i][j];
		}
	}
	Matrix output;
	output.array = newArray;
 	return output;
}

//Add all values in matrix
double Matrix::sumAll() const {
	double sum = 0;
	for (int i = 0; i<array.size(); i++) {
		for (int j = 0; j<array[0].size(); j++) {
			sum+=array[i][j];
		}
	}
	return sum;
}

//Hamming distance from this matrix to input matrix
//Number of positions where the two differ
int Matrix::hamming(const Matrix &b) const {
	//Dimension mismatch
	if (array[0].size()!=b.array[0].size()||array.size()!=b.array.size()) {
		cout << "Matrix size mismatch";
	}
	int sum = 0;
	for (int i = 0; i<array.size(); i++) {
		for (int j = 0; j<array[0].size(); j++) {
			if (array[i][j]!=b.array[i][j]) {
				sum++;
			}
		}
	}
	return sum;
}

//Standard matrix multiplication
Matrix Matrix::matrixMultiplication(const Matrix &b) const {
	//Inner dimension mismatch
	if (array[0].size()!=b.array.size()) {
		cout << "Matrix inner dimension mismatch";
	}
	vector< vector<double> > newArray(array.size(),vector<double> (b.array[0].size()));
	for (int i = 0; i<array.size(); i++) {
		for (int j = 0; j<b.array[0].size(); j++) {
			double sum = 0;
			for (int k = 0; k<array[0].size(); k++) {
				sum = sum+array[i][k]*b.array[k][j];
			}
			newArray[i][j] = sum;
		}
	} 
	Matrix output;
	output.array = newArray;
	return output;
}

//Hadamard matrix multiplication (i.e. component-wise)
Matrix Matrix::hadamardMultiplication(const Matrix &b) const {
	//Dimension mismatch
	if (array[0].size()!=b.array[0].size()||array.size()!=b.array.size()) {
		cout << "Matrix size mismatch";
	}
	vector< vector<double> > newArray = array;
	for (int i = 0; i<array.size(); i++) {
		for (int j = 0; j<array[0].size(); j++) {
			newArray[i][j] = b.array[i][j]*array[i][j];
		}
	}
	Matrix output;
	output.array = newArray;
	return output;
}

//Kronecker matrix multiplication for row and column vectors
Matrix Matrix::kroneckerMultiplication(const Matrix &b) const {
	if (array.size()==1&&b.array[0].size()==1) {
		vector< vector<double> > newArray(array[0].size(),vector<double> (b.array.size()));
		for (int i = 0; i<array[0].size(); i++) {
			for (int j = 0; j<b.array.size(); j++) {
				newArray[i][j] = array[0][i]*b.array[j][0];
			}
		}
		Matrix output;
		output.array = newArray;
		return output;
	}
	else if (b.array.size()==1&&array[0].size()==1) { //Reverse inputs if row and column in wrong order
		return b.kroneckerMultiplication(*this);
	}
	//Not row or column vectors
	cout << "Not row and column vector";
}

//Horizontal concatenation
Matrix Matrix::horizontalConcat(const Matrix &b) const {
	//Row mismatch
	if (array.size()!=b.array.size()) {
		cout << "Matrix row size mismatch";
	}
	vector< vector<double> > newArray(array.size(),vector<double> (array[0].size()+b.array[0].size()));
	for (int i = 0; i<array.size(); i++) {
		for (int j = 0; j<array[0].size()+b.array[0].size(); j++) {
			if (j<array[0].size()) {
				newArray[i][j] = array[i][j];
			}
			else {
				newArray[i][j] = b.array[i][j-array[0].size()];
			}
		}
	}
	Matrix output;
	output.array = newArray;
	return output;
}

//Vertical concatenation
Matrix Matrix::verticalConcat(const Matrix &b) const {
	//Column mismatch
	if (array[0].size()!=b.array[0].size()) {
		cout << "Matrix column size mismatch";
	}
	vector< vector<double> > newArray(array.size()+b.array.size(),vector<double> (array[0].size()));
	for (int i = 0; i<array[0].size(); i++) {
		for (int j = 0; j<array.size()+b.array.size(); j++) {
			if (j<array.size()) {
				newArray[j][i] = array[j][i];
			}
			else {
				newArray[j][i] = b.array[j-array.size()][i];
			}
		}
	}
	Matrix output;
	output.array = newArray;
	return output;
}

//Create string of values separated by spaces
string Matrix::toString () const {
	stringstream output; 
	for (int i = 0; i<array.size(); i++) {
		for (int j = 0; j<array[0].size(); j++) {
			output << " " << array[i][j];
		}
	}
	return output.str();
}

//Create 1D vector from matrix
//Example: 
//1 0 0 0
//0 0 0 1
//0 1 0 0
//0 0 1 0
//Creates column vector 1 4 2 3 
vector<double> Matrix::vectorOutput() const {
	vector<double> output(array.size());
	for (int i = 0; i<array.size(); i++) {
		double max = -1;
		int maxLoc = 0;
		for (int j = 0; j<array[0].size(); j++) {
			if (array[i][j]>max) {
				max = array[i][j];
				maxLoc = j;
			}
		}
		output[i] = maxLoc;
	}
	return output;
}

//Same as above, except column vector is in matrix form
Matrix Matrix::matrixOutput() const {
	vector<double> rowArray = vectorOutput();
	vector< vector<double> > columnArray(array.size(),vector<double>(1));
	for (int i = 0; i<array.size(); i++) {
		columnArray[i][0] = rowArray[i];
	}
	Matrix output;
	output.array = columnArray;
	return output;
}

//Return specific row of matrix, in matrix form
Matrix Matrix::row(int b) const {
	if (b<0||b>array.size()) {
		cout << "Out of row range";
	}
	vector< vector<double> > newArray(1,vector<double>(array[0].size()));
	for (int i = 0; i<array[0].size(); i++) {
		newArray[0][i] = array[b][i];
	}
	Matrix output;
	output.array = newArray;
	return output;
}

//Return specific column of matrix, in matrix form
Matrix Matrix::column(int b) const {
	if (b<0||b>array[0].size()) {
		cout << "Out of column range";
	}
	vector< vector<double> > newArray(array.size(),vector<double>(1));
	for (int i = 0; i<array.size(); i++) {
		newArray[i][0] = array[i][b];
	}
	Matrix output;
	output.array = newArray;
	return output;
}

Matrix::~Matrix() {
}
