//Main function

#include "Training.h"

using namespace std;

int main(){
	//Create training object
	Training *T = new Training();
	float list[6];
	//Return and display weight matrix
	Matrix M = T->train(list[0],list[1],list[2],list[3],list[4],list[5]);
	delete T;
	cout << M.toString();
	return 0;
}
