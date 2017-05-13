//Header file for matrix 

#pragma once
#include <vector>
#include <iostream> 
#include <sstream>
#include <time.h>
#include <stdlib.h>

class Matrix {
	public: 
		std::vector< std::vector<double> > array; 
		Matrix();
		Matrix(const std::vector< std::vector<double> > &b);
		Matrix(const std::vector<double> &b);
		Matrix transpose() const;
		Matrix add(const Matrix &b) const;
		Matrix scalarMultiplication(double factor) const;
		double sumAll () const;
		int hamming(const Matrix &b) const;
		Matrix matrixMultiplication(const Matrix &b) const;
		Matrix hadamardMultiplication(const Matrix &b) const;
		Matrix kroneckerMultiplication(const Matrix &b) const;
		Matrix horizontalConcat(const Matrix &b) const;
		Matrix verticalConcat(const Matrix &b) const;
		std::string toString () const;
		std::vector<double> vectorOutput() const;
		Matrix matrixOutput() const;
		Matrix row(int b) const;
		Matrix column(int b) const;
		~Matrix();
};

